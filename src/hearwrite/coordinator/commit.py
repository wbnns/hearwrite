"""The commit policy: what text is final, and when.

An engine reports which words it will stand behind. This module decides which of
those actually get committed, and that is a separate question -- committing is
irreversible, so it is worth holding a word a little longer when the engine is
unsure.

C1 confidence gating (the `confidence_gate` policy field) is the cheap
approximation of the reference system's learned per-word delay. A high-confidence
word commits immediately; a low-confidence one waits for `slow_commit_seconds` of
further audio to see whether the engine changes its mind. It is pure policy, it
needs no training, and it captures a useful part of the benefit for none of the
cost. Set the gate to 0.0 to commit every stable word immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engines.base import Hypothesis, Word


@dataclass
class _Held:
    word: Word
    first_seen_at: float


class CommitPolicy:
    """Turns a stream of hypotheses into an append-only sequence of words."""

    def __init__(self, *, confidence_gate: float = 0.0, slow_commit_seconds: float = 0.35) -> None:
        self._gate = confidence_gate
        self._slow = slow_commit_seconds
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
        """Return the words that become final on this push, in order."""
        ready: list[Word] = []
        still_held: dict[tuple[float, str], _Held] = {}

        for word in hypothesis.stable:
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
                continue

            if now - held.first_seen_at >= self._slow:
                ready.append(word)
            else:
                still_held[key] = held

        self._held = still_held

        ready.sort(key=lambda w: w.audio_start)
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
