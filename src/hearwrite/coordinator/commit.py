"""The commit policy: what text is final, and when.

An engine reports which words it will stand behind. This module decides which of
those actually get committed, and that is a separate question -- committing is
irreversible, so it is worth holding a word a little longer when the engine is
unsure, and worth taking one early when it is very sure.

C1 confidence gating approximates the reference system's learned per word delay
with policy instead of training, and it runs in both directions:

  * `confidence_gate` HOLDS a stable word the engine is unsure about, for
    `slow_commit_seconds` of further audio. Costs latency, buys accuracy.
  * `early_commit_confidence` TAKES a tentative word the engine is very sure
    about, without waiting for it to settle. Buys latency, costs accuracy.

Which one helps depends entirely on the engine. A greedy transducer settles a
word the moment it emits it, so there is no tentative state to skip and early
commit does nothing. LocalAgreement over Whisper holds every word until two
passes agree, so early commit can remove a whole pass of latency -- which is
where the measured win is.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import takewhile

from ..engines.base import Hypothesis, Word


@dataclass
class _Held:
    word: Word
    first_seen_at: float


class CommitPolicy:
    """Turns a stream of hypotheses into an append-only sequence of words."""

    def __init__(
        self,
        *,
        confidence_gate: float = 0.0,
        slow_commit_seconds: float = 0.35,
        early_commit_confidence: float = 1.01,
    ) -> None:
        self._gate = confidence_gate
        self._slow = slow_commit_seconds
        self._early = early_commit_confidence
        self._committed_to: float = 0.0
        self._held: dict[tuple[float, str], _Held] = {}

    @property
    def committed_to(self) -> float:
        """Audio position through which text has been committed."""
        return self._committed_to

    def reset(self) -> None:
        self._committed_to = 0.0
        self._held.clear()

    def take(self, hypothesis: Hypothesis, now: float) -> tuple[Word, ...]:
        """Return the words that become final on this push, in order.

        THE COMMITTED SEQUENCE MUST HAVE NO HOLES. Every candidate is considered
        in time order, and the first one that is not ready stops the rest: a word
        held back blocks everything after it, because emitting word five while
        word four is still pending would leave a gap that the append-only rule
        makes permanent.

        That is not hypothetical. Without it, an early commit of a confident
        later word advanced the frontier past two earlier words and dropped them
        from the transcript entirely.
        """
        candidates = sorted(hypothesis.stable, key=lambda w: w.audio_start)
        if self._early <= 1.0:
            # Only a contiguous PREFIX of the tentative words may be taken.
            # Filtering them individually by confidence picks a scattered
            # subset, and committing a confident later word advances the
            # frontier past an unconfident earlier one, which deletes it. That
            # bug ate "The build" from a test transcript.
            tentative = sorted(hypothesis.tentative, key=lambda w: w.audio_start)
            candidates += list(takewhile(lambda w: w.confidence >= self._early, tentative))

        ready: list[Word] = []
        still_held: dict[tuple[float, str], _Held] = {}

        for word in candidates:
            # Anything at or behind the commit frontier is already out the door.
            if word.audio_start + 1e-6 < self._committed_to:
                continue

            key = (round(word.audio_start, 4), word.text)

            if word.confidence >= self._gate:
                ready.append(word)
                continue

            held = self._held.get(key)
            if held is None:
                # First sighting of a word we are not confident about. Hold it
                # and see whether the engine still says the same thing shortly.
                still_held[key] = _Held(word, now)
                break

            if now - held.first_seen_at >= self._slow:
                ready.append(word)
            else:
                still_held[key] = held
                break

        self._held = still_held

        for word in ready:
            self._committed_to = max(self._committed_to, word.audio_end)
        return tuple(ready)

    def flush(self, hypothesis: Hypothesis) -> tuple[Word, ...]:
        """End of stream: everything still pending becomes final."""
        ready = [
            w
            for w in hypothesis.stable + hypothesis.tentative
            if w.audio_start + 1e-6 >= self._committed_to
        ]
        ready.sort(key=lambda w: w.audio_start)
        for word in ready:
            self._committed_to = max(self._committed_to, word.audio_end)
        self._held.clear()
        return tuple(ready)
