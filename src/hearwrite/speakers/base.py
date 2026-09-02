"""The speaker frontend interface: segmentation plus embeddings, nothing more.

Diarization splits into two halves that fail in completely different ways, so
HearWrite splits them into two testable pieces:

  * The MODEL half (this interface) answers "someone spoke from t1 to t2, and
    here is a vector for their voice". It has no notion of speaker identity,
    speaker count, or history.
  * The LOGIC half (coordinator/speakers.py) turns those vectors into stable
    speaker labels over time. That is our code, it is pure, and it is where the
    real bugs live.

Keeping the count-agnostic half in the model is what lets HearWrite handle more
than four speakers. Nothing in this interface takes a speaker count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Segment:
    """A stretch of speech with a voice embedding.

    `overlap` marks a region where more than one person is talking. HearWrite
    detects overlap and degrades gracefully; it does not separate speakers (a
    stated non-goal). An overlapping segment is labeled `None` rather than
    assigned to whichever voice happened to win.
    """

    start: float
    end: float
    embedding: tuple[float, ...]
    overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"segment ends ({self.end}) before it starts ({self.start})")
        if not self.embedding:
            raise ValueError("segment has an empty embedding")


@runtime_checkable
class SpeakerFrontend(Protocol):
    """Emits (start, end, embedding) over a rolling window."""

    sample_rate: int

    def push(self, pcm: bytes, at: float) -> tuple[Segment, ...]:
        """Feed audio. Return any segments that closed in this chunk."""
        ...

    def flush(self) -> tuple[Segment, ...]: ...

    def reset(self) -> None: ...
