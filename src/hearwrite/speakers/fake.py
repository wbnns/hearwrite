"""A scripted speaker frontend for Tier 1.

Records every call, so a test can assert the strongest property of solo mode:
that the frontend is not merely ignored but never invoked at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Segment


def embedding(*values: float, dim: int = 8) -> tuple[float, ...]:
    """Pad a short vector out to `dim`, so fixtures stay readable."""
    out = list(values) + [0.0] * (dim - len(values))
    return tuple(out[:dim])


@dataclass
class ScriptedFrontend:
    """Emits canned segments once the stream passes their end position."""

    segments: tuple[Segment, ...] = ()
    sample_rate: int = 16_000
    _emitted: int = field(default=0, init=False)
    calls: list[tuple[int, float]] = field(default_factory=list, init=False)

    def push(self, pcm: bytes, at: float) -> tuple[Segment, ...]:
        self.calls.append((len(pcm), at))
        out = []
        while self._emitted < len(self.segments) and self.segments[self._emitted].end <= at + 1e-9:
            out.append(self.segments[self._emitted])
            self._emitted += 1
        return tuple(out)

    def flush(self) -> tuple[Segment, ...]:
        out = self.segments[self._emitted :]
        self._emitted = len(self.segments)
        return tuple(out)

    def reset(self) -> None:
        self._emitted = 0
        self.calls.clear()
