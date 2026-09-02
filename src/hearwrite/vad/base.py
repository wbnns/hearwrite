"""The acoustic gate: is anyone talking right now.

A VAD times silence. It does not judge whether a thought is finished -- "what's
the weather in--" and "what's the weather in Menlo Park" look identical to it if
the pause lengths match. That is why endpointing is conjunctive: this interface
is the necessary half, and turn/base.py is the sufficient half.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SpeechState:
    """Whether the most recent audio was speech, and since when."""

    speaking: bool
    at: float
    #: Stream position where the current run of speech/silence began.
    since: float

    @property
    def held_for(self) -> float:
        return self.at - self.since


@runtime_checkable
class VAD(Protocol):
    sample_rate: int

    def push(self, pcm: bytes, at: float) -> SpeechState: ...

    def reset(self) -> None: ...
