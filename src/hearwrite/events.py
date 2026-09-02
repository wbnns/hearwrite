"""The event model: one merged, ordered, append-only token stream.

Muse Voice Transcribe emits transcription, speaker identity and endpoints as
special tokens in one shared vocabulary. HearWrite gets those three signals from
separate models, but presents them the same way -- a single ordered log where
speaker and endpoint events interleave with text.

That choice is deliberate and load-bearing. It means the client contract does not
encode how many models are behind it, so replacing three models with one joint
model later is a Coordinator change rather than a protocol break.

    HearWrite event   Muse Voice Transcribe token
    speech_onset      <|speech_onset|>
    turn_start        <|start_of_turn|>
    speaker           <|speaker_A|>
    endpoint          <|speech_endpoint|>
    partial / commit  transcription tokens
    degraded          -- (HearWrite-specific; see coordinator.backpressure)

THE ONE RULE: once a `commit` is emitted, nothing may contradict it. `partial`
may be revised or withdrawn at any time. A consumer that ignores every `partial`
must still receive a correct, complete transcript.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# A speaker label the Coordinator could not assign unambiguously. Committing
# `None` is a supported outcome, not a failure: under the append-only rule a
# wrong speaker label can never be corrected, so abstaining and filling the gap
# with a later `speaker` event is the only safe move.
UNASSIGNED: str | None = None


class EventKind(StrEnum):
    SPEECH_ONSET = "speech_onset"
    TURN_START = "turn_start"
    PARTIAL = "partial"
    COMMIT = "commit"
    SPEAKER = "speaker"
    ENDPOINT = "endpoint"
    DEGRADED = "degraded"


#: Kinds a consumer may never see revised. Everything else is provisional.
FINAL_KINDS = frozenset({EventKind.COMMIT, EventKind.ENDPOINT, EventKind.TURN_START})


@dataclass(frozen=True)
class Event:
    """One entry in the log. `at` is a stream position in seconds, never a clock time."""

    seq: int
    kind: EventKind
    at: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError(f"seq must be non-negative, got {self.seq}")
        if self.at < 0:
            raise ValueError(f"at must be non-negative, got {self.at}")


class AppendOnlyViolation(RuntimeError):
    """Raised when code tries to contradict something already committed.

    This is a programming error in the Coordinator, not a runtime condition to
    handle. It exists so the invariant fails loudly in tests rather than quietly
    in a user's transcript.
    """


class EventLog:
    """Assigns sequence numbers and enforces the append-only rule.

    The log is the only thing allowed to mint a `seq`, so ordering is total and
    gap-free by construction rather than by convention.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._committed_words: int = 0
        self._last_commit_end: float = 0.0
        self._last_at: float = 0.0

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, i: int) -> Event:
        return self._events[i]

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    @property
    def committed_text(self) -> str:
        """Everything committed so far, in order. This only ever grows."""
        return " ".join(str(e.payload["text"]) for e in self._events if e.kind is EventKind.COMMIT)

    @property
    def last_commit_end(self) -> float:
        """Audio position at the end of the most recent commit."""
        return self._last_commit_end

    def emit(self, kind: EventKind, at: float, payload: Mapping[str, Any]) -> Event:
        # `at` is when the event was EMITTED, not what audio it describes. The
        # audio span a commit refers to lives in its payload. Conflating the two
        # is how `turn_start` once ended up claiming a position earlier than the
        # endpoint that preceded it.
        if at + 1e-6 < self._last_at:
            raise AppendOnlyViolation(
                f"{kind} at {at:.3f}s goes backwards; the log is already at {self._last_at:.3f}s"
            )
        self._last_at = max(self._last_at, at)

        if kind is EventKind.COMMIT:
            audio_start = float(payload["audio_start"])
            # Commits must tile the audio timeline forward. A commit that starts
            # before the previous one ended means two engines disagreed about
            # word order, which would surface to the user as scrambled text.
            if audio_start + 1e-6 < self._last_commit_end:
                raise AppendOnlyViolation(
                    f"commit at {audio_start:.3f}s starts before the previous "
                    f"commit ended at {self._last_commit_end:.3f}s"
                )
            self._last_commit_end = float(payload["audio_end"])
            self._committed_words += 1

        event = Event(seq=len(self._events), kind=kind, at=at, payload=dict(payload))
        self._events.append(event)
        return event


def commit_payload(
    *,
    text: str,
    audio_start: float,
    audio_end: float,
    emitted_at: float,
    speaker: str | None,
    confidence: float,
) -> dict[str, Any]:
    """Build a commit payload carrying both clocks.

    `delay = emitted_at - audio_end` is audio-relative, so it stays meaningful
    under accelerated replay. Wall-clock lag is recorded elsewhere and is only
    meaningful at 1x pacing.
    """
    return {
        "text": text,
        "audio_start": audio_start,
        "audio_end": audio_end,
        "emitted_at": emitted_at,
        "delay": emitted_at - audio_end,
        "speaker": speaker,
        "confidence": confidence,
    }
