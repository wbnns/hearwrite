"""A scripted VAD for Tier 1.

Speech is described as spans, so a test states "silence from 2.0s to 3.5s"
directly instead of synthesising audio quiet enough to trip a real detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import SpeechState


@dataclass
class ScriptedVAD:
    """Reports speech according to a list of (start, end) spans."""

    speech: tuple[tuple[float, float], ...] = ()
    sample_rate: int = 16_000
    _since: float = field(default=0.0, init=False)
    _last: bool | None = field(default=None, init=False)

    def push(self, pcm: bytes, at: float) -> SpeechState:
        speaking = any(start <= at < end for start, end in self.speech)
        if self._last is None or speaking != self._last:
            self._since = at
            self._last = speaking
        return SpeechState(speaking=speaking, at=at, since=self._since)

    def reset(self) -> None:
        self._since = 0.0
        self._last = None
