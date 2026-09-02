"""The same guarantees, whichever engine is behind the interface.

This is the file that justifies the Phase 0 sequencing bet. The ASR interface
was defined against a transducer's shape before any real engine existed, on the
argument that designing to the easier case first guarantees a rewrite. These
tests run one scenario through two engines with completely different internals
-- a transducer that emits monotonically, and an offline model that transcribes
a growing buffer and revises itself -- and assert the Coordinator's contract
holds identically for both.

If a future engine needs a change to `coordinator/` or `protocol.py` to pass
these, the abstraction has leaked and the fix belongs in the adapter.
"""

from __future__ import annotations

import pytest

from hearwrite import DICTATION, Coordinator
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.engines.whisper import WhisperStreamingEngine
from hearwrite.events import EventKind
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive
from .test_whisper_adapter import StubModel

SR = 16_000

#: The same sentence, arriving the way each engine would actually deliver it.
SENTENCE = "the build is green"


def transducer_engine():
    """Monotonic: tokens appear and never change."""
    return ScriptedEngine(
        script={
            1.6: Hypothesis(stable=words("the build", each=0.4), consumed_to=1.6),
            2.6: Hypothesis(stable=words(SENTENCE, each=0.4), consumed_to=2.6),
        }
    )


def offline_engine():
    """Revises itself: the first pass is wrong about the last word."""
    passes = [
        [("the", 0.0, 0.4), ("build", 0.4, 0.8)],
        [("the", 0.0, 0.4), ("build", 0.4, 0.8), ("is", 0.8, 1.2)],
        [("the", 0.0, 0.4), ("build", 0.4, 0.8), ("is", 0.8, 1.2), ("grey", 1.2, 1.6)],
        [("the", 0.0, 0.4), ("build", 0.4, 0.8), ("is", 0.8, 1.2), ("green", 1.2, 1.6)],
        [("the", 0.0, 0.4), ("build", 0.4, 0.8), ("is", 0.8, 1.2), ("green", 1.2, 1.6)],
    ]
    return WhisperStreamingEngine(StubModel(passes), sample_rate=SR, interval=1.0)


ENGINES = pytest.mark.parametrize(
    "build_engine", [transducer_engine, offline_engine], ids=["transducer", "offline"]
)


def run(build_engine, seconds=6.0):
    coordinator = Coordinator(
        DICTATION,
        engine=build_engine(),
        vad=ScriptedVAD(speech=((0.0, 3.0),)),
    )
    return coordinator, drive(coordinator, seconds)


@ENGINES
def test_committed_output_is_append_only(build_engine):
    _, events = run(build_engine)
    frontier = 0.0
    for event in events:
        if event.kind is EventKind.COMMIT:
            assert event.payload["audio_start"] + 1e-6 >= frontier
            frontier = event.payload["audio_end"]


@ENGINES
def test_event_times_never_go_backwards(build_engine):
    _, events = run(build_engine)
    times = [e.at for e in events]
    assert times == sorted(times)


@ENGINES
def test_no_word_is_emitted_before_its_audio(build_engine):
    _, events = run(build_engine)
    for event in events:
        if event.kind is EventKind.COMMIT:
            assert event.payload["delay"] >= -1e-6


@ENGINES
def test_a_retracted_word_never_reaches_a_commit(build_engine):
    """The offline engine says "grey" once and then corrects itself to "green".
    Neither engine may leak a word it withdrew.
    """
    coordinator, _ = run(build_engine)
    assert "grey" not in coordinator.log.committed_text


@ENGINES
def test_the_transcript_is_the_same_either_way(build_engine):
    coordinator, _ = run(build_engine)
    assert coordinator.log.committed_text == SENTENCE


@ENGINES
def test_turns_and_endpoints_are_produced(build_engine):
    _, events = run(build_engine)
    kinds = {str(e.kind) for e in events}
    assert "turn_start" in kinds
    assert "endpoint" in kinds


def test_the_offline_engine_is_slower_to_commit():
    """Not a defect, the documented trade. A transducer decides per chunk; an
    offline model must see a word twice before it will stand behind it.
    """
    _, fast = run(transducer_engine)
    _, slow = run(offline_engine)

    def worst(events):
        return max(e.payload["delay"] for e in events if e.kind is EventKind.COMMIT)

    assert worst(slow) > worst(fast)
