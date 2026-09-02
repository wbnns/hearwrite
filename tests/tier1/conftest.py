"""Shared fixtures for Tier 1.

Everything here is scripted. No models, no downloads, no network, no GPU. The
whole suite has to stay fast enough that running it is never a decision.
"""

from __future__ import annotations

import pytest

from hearwrite.engines.base import Hypothesis, Word
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.speakers.base import Segment
from hearwrite.speakers.fake import ScriptedFrontend, embedding
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

SAMPLE_RATE = 16_000


def pcm(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Silent PCM of a given duration. Contents never matter to a fake."""
    return b"\x00\x00" * int(seconds * sample_rate)


def drive(coordinator, seconds: float, chunk: float = 0.02, lag: float = 0.0):
    """Push `seconds` of audio in `chunk`-sized pieces, then finish."""
    frame = pcm(chunk, coordinator.policy.sample_rate)
    events = []
    for _ in range(round(seconds / chunk)):
        events.extend(coordinator.push(frame, lag=lag))
    events.extend(coordinator.finish())
    return events


def committed(events):
    """The committed prefix as comparable tuples, ignoring emission timing."""
    return [
        (
            e.payload["text"],
            round(e.payload["audio_start"], 4),
            round(e.payload["audio_end"], 4),
        )
        for e in events
        if e.kind == "commit"
    ]


def kinds(events):
    return [str(e.kind) for e in events]


@pytest.fixture
def two_speaker_segments():
    """Two clearly distinct voices, alternating."""
    a = embedding(1.0, 0.0, 0.0)
    b = embedding(0.0, 1.0, 0.0)
    return (
        Segment(0.0, 1.2, a),
        Segment(1.3, 2.4, b),
        Segment(2.5, 3.6, a),
    )


@pytest.fixture
def sentence_engine():
    """An engine that stabilises a sentence in two steps, never ahead of the audio."""
    return ScriptedEngine(
        script={
            1.6: Hypothesis(stable=words("what's the weather in", each=0.4), consumed_to=1.6),
            2.6: Hypothesis(
                stable=words("what's the weather in Menlo Park.", each=0.4), consumed_to=2.6
            ),
        }
    )


__all__ = [
    "SAMPLE_RATE",
    "Hypothesis",
    "ScriptedEngine",
    "ScriptedFrontend",
    "ScriptedTurnDetector",
    "ScriptedVAD",
    "Segment",
    "Word",
    "committed",
    "drive",
    "embedding",
    "kinds",
    "pcm",
    "words",
]
