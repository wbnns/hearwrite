"""The Coordinator: the only stateful component, and the actual product.

Everything upstream of here is a stateless model wrapper behind a Protocol.
Everything that determines whether the output *feels* good -- when text is final,
which speaker a word belongs to, whether a pause was the end of a thought --
lives in this package, in plain synchronous Python with no I/O.

That boundary is the design. It means models are swappable, and it means the
interesting behaviour is testable in milliseconds on a laptop with no GPU, no
network and no downloads.

Determinism is a hard requirement, not a nicety. The Coordinator never reads a
system clock; every timestamp comes from the StreamClock, which only moves when
audio is pushed. That is what makes a fixture replayed at 10x produce a
byte-identical event log to the same fixture at 1x.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..clock import StreamClock
from ..engines.base import ASREngine, Hypothesis, Word
from ..events import Event, EventKind, EventLog, commit_payload
from ..speakers.base import SpeakerFrontend
from ..turn.base import TurnDetector
from ..vad.base import VAD, SpeechState
from .backpressure import BackpressureGate
from .commit import CommitPolicy
from .endpoint import EndpointGate
from .policy import Policy
from .speakers import SpeakerTracker

#: The label every word gets in solo mode. Solo does not mean "one cluster", it
#: means the clustering never runs.
SOLO_LABEL = "A"


class EngineTimestampError(RuntimeError):
    """An engine reported a word ending after the audio it was given.

    Always an adapter bug, never a runtime condition. Raised rather than clamped
    because a clamped timestamp produces a plausible-looking delay metric that is
    quietly wrong, which is far more expensive to find later.
    """


class Coordinator:
    """Drives the models and produces the event log."""

    def __init__(
        self,
        policy: Policy,
        *,
        engine: ASREngine,
        vad: VAD | None = None,
        speakers: SpeakerFrontend | None = None,
        turn: TurnDetector | None = None,
    ) -> None:
        self.policy = policy
        self.log = EventLog()

        self._engine = engine
        self._vad = vad
        self._turn = turn
        # Solo mode is a BYPASS, not a special case of clustering. Running a
        # diarizer over a single voice occasionally splits that person into two
        # labels, which is worse than emitting no distinction at all -- and it
        # costs two models on the hot path for the privilege.
        self._frontend = None if policy.is_solo else speakers

        self._clock = StreamClock(policy.sample_rate)
        self._commit = CommitPolicy(
            confidence_gate=policy.confidence_gate,
            slow_commit_seconds=policy.slow_commit_seconds,
            early_commit_confidence=policy.early_commit_confidence,
        )
        self._endpoint = EndpointGate(policy.endpoint)
        self._backpressure = BackpressureGate(policy.max_lag_seconds)
        self._tracker = None if policy.is_solo else SpeakerTracker(policy.speakers)

        self._turn_open = False
        self._turn_speaker: str | None = None
        self._turn_index = 0
        #: Audio position of the most recent endpoint. A word whose audio starts
        #: before this belongs to the turn that already closed, however late it
        #: arrives.
        self._closed_through = 0.0
        self._spoke_yet = False
        self._last_completeness = 0.0
        #: Rolling audio for the turn detector. An audio native detector needs
        #: the speech leading up to the pause, and wrappers are stateless, so
        #: the buffer lives here with the rest of the state.
        self._recent = bytearray()
        self._scored_at = -1.0
        self._recent_limit = int(policy.turn_context_seconds * policy.sample_rate) * 2
        #: Commit seq numbers still waiting for a speaker label, with the audio
        #: span needed to resolve them later.
        self._unlabeled: list[tuple[int, float, float]] = []

    @property
    def position(self) -> float:
        return self._clock.position

    @property
    def speaker_count(self) -> int:
        return 1 if self.policy.is_solo else (self._tracker.speaker_count if self._tracker else 0)

    def push(self, pcm: bytes, *, lag: float = 0.0) -> tuple[Event, ...]:
        """Feed one chunk of PCM and return the events it produced.

        `lag` is how far behind real time the caller is running. It is the only
        wall-clock-derived input, it is optional, and it never reaches a
        timestamp -- it only gates whether partials are emitted.
        """
        first = len(self.log)
        at = self._clock.advance_bytes(len(pcm))

        self._observe_lag(lag, at)

        if self._turn is not None:
            self._recent += pcm
            if len(self._recent) > self._recent_limit:
                del self._recent[: len(self._recent) - self._recent_limit]

        state = self._read_vad(pcm, at)
        if state is not None and state.speaking and not self._spoke_yet:
            self._spoke_yet = True
            self.log.emit(EventKind.SPEECH_ONSET, at, {"at": at})

        self._ingest_segments(pcm, at)

        hypothesis = self._engine.push(pcm, at)
        # None is the transducer's blank: audio was processed, nothing to say.
        if hypothesis is not None:
            self._emit_words(self._commit.take(hypothesis, at), at)
            self._emit_partial(hypothesis, at)

        self._fill_unlabeled()
        self._close_endpoint(state, at)

        return self.log.events[first:]

    def finish(self) -> tuple[Event, ...]:
        """End of stream. Commit everything outstanding and close the turn."""
        first = len(self.log)
        at = self._clock.position

        for segment in self._drain_frontend():
            self._assign(segment)

        self._emit_words(self._commit.flush(self._engine.flush()), at)
        self._fill_unlabeled(final=True)

        endpoint = self._endpoint.flush(at, self._completeness())
        if endpoint is not None:
            self._emit_endpoint(endpoint.at, endpoint.reason, endpoint.completeness)

        return self.log.events[first:]

    # -- internals ---------------------------------------------------------

    def _observe_lag(self, lag: float, at: float) -> None:
        changed = self._backpressure.observe(lag)
        if changed is not None:
            self.log.emit(
                EventKind.DEGRADED,
                at,
                {"degraded": changed, "lag": lag, "dropping": "partial"},
            )

    def _read_vad(self, pcm: bytes, at: float) -> SpeechState | None:
        if self._vad is None:
            # With no VAD there is no acoustic gate, so treat all audio as
            # speech. Endpointing then relies entirely on the flush path.
            return None
        return self._vad.push(pcm, at)

    def _ingest_segments(self, pcm: bytes, at: float) -> None:
        if self._frontend is None:
            return
        for segment in self._frontend.push(pcm, at):
            self._assign(segment)

    def _drain_frontend(self) -> Iterable:
        if self._frontend is None:
            return ()
        return self._frontend.flush()

    def _assign(self, segment) -> None:
        if self._tracker is None:
            return
        self._tracker.assign(segment)
        for survivor, absorbed in self._tracker.merge_candidates():
            # Merges apply to open and future turns. Labels already emitted in a
            # commit stand -- the append-only rule forbids rewriting them.
            self._tracker.merge(survivor, absorbed)

    def _emit_words(self, words: tuple[Word, ...], at: float) -> None:
        for word in words:
            # A word cannot end after the audio that has been pushed. If it
            # does, the engine adapter is reporting timestamps in the wrong
            # frame of reference, and every emission-delay number downstream is
            # silently wrong. Fail here, where the cause is still obvious.
            if word.audio_end > at + 1e-6:
                raise EngineTimestampError(
                    f"engine emitted {word.text!r} ending at {word.audio_end:.3f}s "
                    f"but only {at:.3f}s of audio has been pushed; the adapter is "
                    f"probably reporting buffer-relative rather than stream-relative times"
                )
            speaker = self._speaker_for(word)
            self._maybe_open_turn(speaker, at, word.audio_start)
            event = self.log.emit(
                EventKind.COMMIT,
                at,
                commit_payload(
                    text=word.text,
                    audio_start=word.audio_start,
                    audio_end=word.audio_end,
                    emitted_at=at,
                    speaker=speaker,
                    confidence=word.confidence,
                ),
            )
            if speaker is None:
                self._unlabeled.append((event.seq, word.audio_start, word.audio_end))

    def _speaker_for(self, word: Word) -> str | None:
        if self.policy.is_solo:
            return SOLO_LABEL
        if self._tracker is None:
            return None
        return self._tracker.label_for(word.audio_start, word.audio_end)

    def _maybe_open_turn(self, speaker: str | None, at: float, audio_start: float) -> None:
        """A turn starts on the first word after an endpoint, or on a new voice.

        `at` is the emission position, so the log stays ordered in time.
        `audio_start` is where the turn's first word actually begins, which is
        what a consumer wants to seek to. They are different numbers and both
        matter -- stamping the event with the audio position is what once let a
        turn_start claim a moment earlier than the endpoint before it.

        An unlabeled word never starts a turn on its own -- we do not know whose
        turn it would be, and inventing a boundary is exactly the kind of guess
        the append-only rule punishes.
        """
        # A streaming transducer releases its last words only after it hears
        # trailing audio, so words routinely arrive after the endpoint that
        # followed them. Such a word trails the closed turn; opening a new turn
        # for it would invent a boundary in the middle of a finished sentence.
        if not self._turn_open and audio_start + 1e-6 < self._closed_through:
            return

        changed = speaker is not None and self._turn_open and speaker != self._turn_speaker
        if self._turn_open and not changed:
            return
        if not self._turn_open or changed:
            self._turn_index += 1
            self._turn_open = True
            self._turn_speaker = speaker
            self.log.emit(
                EventKind.TURN_START,
                at,
                {
                    "speaker": speaker,
                    "turn": self._turn_index,
                    "audio_start": audio_start,
                },
            )

    def _emit_partial(self, hypothesis: Hypothesis, at: float) -> None:
        if not self._backpressure.allows_partial():
            return
        pending = [w for w in hypothesis.tentative if w.audio_start >= self._commit.committed_to]
        if not pending:
            return
        self.log.emit(
            EventKind.PARTIAL,
            at,
            {
                "text": " ".join(w.text for w in pending),
                "audio_start": pending[0].audio_start,
                "audio_end": pending[-1].audio_end,
            },
        )

    def _fill_unlabeled(self, *, final: bool = False) -> None:
        """Emit `speaker` events for words that have since become resolvable.

        A commit that carried `speaker: null` made no claim, so supplying the
        label later contradicts nothing. This is the mechanism that lets the
        clustering abstain safely.
        """
        if self._tracker is None or not self._unlabeled:
            return
        still: list[tuple[int, float, float]] = []
        for seq, start, end in self._unlabeled:
            label = self._tracker.label_for(start, end)
            if label is None:
                if not final:
                    still.append((seq, start, end))
                continue
            self.log.emit(
                EventKind.SPEAKER,
                self._clock.position,
                {"seq": seq, "speaker": label, "audio_start": start, "audio_end": end},
            )
        self._unlabeled = still

    def _completeness(self) -> float:
        if self._turn is None:
            # With no semantic gate the conjunction would never be satisfied and
            # every endpoint would come from the timeout. Treat "no detector" as
            # "no semantic objection" so the acoustic gate alone decides.
            return 1.0
        self._last_completeness = self._turn.completeness(
            self.log.committed_text, bytes(self._recent)
        )
        return self._last_completeness

    def _close_endpoint(self, state: SpeechState | None, at: float) -> None:
        if state is None:
            return
        completeness = self._maybe_score(state, at)
        endpoint = self._endpoint.observe(state, completeness)
        if endpoint is not None:
            self._emit_endpoint(endpoint.at, endpoint.reason, endpoint.completeness)

    def _maybe_score(self, state: SpeechState, at: float) -> float:
        """Run the semantic gate only when its answer could change the outcome.

        Two throttles, and both matter. The score is irrelevant until the
        ACOUSTIC gate is satisfied, because the conjunction cannot fire before
        then. And once it is satisfied, re-running an eight second model every
        20ms is pure waste: scoring at frame rate cost nine times the real time
        factor of everything else in the pipeline combined.
        """
        if self._turn is None:
            # No semantic gate installed, so no semantic objection. The acoustic
            # side decides alone. Returning 0.0 here instead would veto every
            # conjunction and leave nothing but the timeout, which is a much
            # slower endpoint for everyone who has not installed the turn extra.
            return 1.0
        if not self._endpoint.wants_completeness:
            return 0.0
        if self._endpoint.silence_held(at) < self.policy.endpoint.silence_seconds:
            return 0.0
        if at - self._scored_at < self.policy.turn_interval:
            return self._last_completeness
        self._scored_at = at
        return self._completeness()

    def _emit_endpoint(self, at: float, reason, completeness: float) -> None:
        self.log.emit(
            EventKind.ENDPOINT,
            at,
            {"reason": str(reason), "completeness": completeness, "turn": self._turn_index},
        )
        # The turn is now closed. Unlabeled words in it get one last chance at a
        # label, then stay null forever; that bounds memory over a long session.
        self._fill_unlabeled(final=True)
        self._turn_open = False
        self._turn_speaker = None
        self._closed_through = max(self._closed_through, at)
