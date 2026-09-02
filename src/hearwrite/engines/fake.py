"""A scripted ASR engine for Tier 1.

The point of a fake here is not to avoid a slow dependency, it is to make the
engine's behaviour an INPUT to the test. Real recognizers are the wrong tool for
asserting that the commit policy is append-only, because you cannot ask one to
produce a specific pathological hypothesis on demand.

ScriptedEngine lets a test say exactly that: at this audio position the engine
stabilises these words, at that one it changes its mind about a tentative word,
here it returns the blank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Hypothesis, Word


def words(
    spec: str, *, start: float = 0.0, each: float = 0.3, confidence: float = 1.0
) -> tuple[Word, ...]:
    """Build evenly spaced words from a sentence. Convenience for fixtures."""
    out = []
    at = start
    for token in spec.split():
        out.append(Word(token, at, at + each, confidence))
        at += each
    return tuple(out)


@dataclass
class ScriptedEngine:
    """Replays a canned sequence of hypotheses, keyed by stream position.

    `script` maps a stream position (seconds) to what the engine returns once the
    stream reaches it. A value of None is the transducer's blank.
    """

    script: dict[float, Hypothesis | None] = field(default_factory=dict)
    sample_rate: int = 16_000
    final: Hypothesis | None = None
    _fired: set[float] = field(default_factory=set, init=False)
    _last: Hypothesis | None = field(default=None, init=False)
    #: Every (pcm_len, at) push this engine received. Solo-mode tests assert the
    #: speaker frontend was never called; this is the same idea for the engine.
    calls: list[tuple[int, float]] = field(default_factory=list, init=False)

    def push(self, pcm: bytes, at: float) -> Hypothesis | None:
        self.calls.append((len(pcm), at))
        due = [t for t in self.script if t <= at + 1e-9 and t not in self._fired]
        if not due:
            return None
        for t in sorted(due):
            self._fired.add(t)
            self._last = self.script[t]
        return self._last

    def flush(self) -> Hypothesis:
        if self.final is not None:
            return self.final
        if self._last is not None:
            return Hypothesis(stable=self._last.all_words, consumed_to=self._last.consumed_to)
        return Hypothesis()

    def reset(self) -> None:
        self._fired.clear()
        self._last = None
        self.calls.clear()
