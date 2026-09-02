"""Solo mode is a bypass, not clustering with one cluster.

The distinction matters twice. For correctness: a diarizer run over a single
voice occasionally splits that person into two labels, which is worse than
making no distinction at all. For cost: it drops the segmentation and embedding
models off the hot path entirely, which is most of the CPU budget on a laptop.

So the assertion is not "solo mode produces one label". It is "the speaker
frontend was never called".
"""

from __future__ import annotations

from hearwrite import AGENT, CONVERSATION, DICTATION, Coordinator
from hearwrite.coordinator.policy import SpeakerMode
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.speakers.fake import ScriptedFrontend
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive


def _engine():
    return ScriptedEngine(
        script={2.1: Hypothesis(stable=words("one two three four five", each=0.4), consumed_to=2.1)}
    )


def test_solo_never_calls_the_speaker_frontend(two_speaker_segments):
    frontend = ScriptedFrontend(segments=two_speaker_segments)
    coord = Coordinator(
        DICTATION,
        engine=_engine(),
        vad=ScriptedVAD(speech=((0.0, 2.2),)),
        speakers=frontend,
        turn=ScriptedTurnDetector(),
    )
    drive(coord, 3.0)

    assert frontend.calls == [], "solo mode invoked the speaker frontend"


def test_solo_labels_every_word_the_same(two_speaker_segments):
    """Even when the audio genuinely contains two people, solo says one."""
    coord = Coordinator(
        DICTATION,
        engine=_engine(),
        vad=ScriptedVAD(speech=((0.0, 2.2),)),
        speakers=ScriptedFrontend(segments=two_speaker_segments),
    )
    events = drive(coord, 3.0)
    speakers = {e.payload["speaker"] for e in events if e.kind == "commit"}
    assert speakers == {"A"}


def test_solo_never_emits_a_null_label():
    """Nothing to abstain about, so no word should ever need filling later."""
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 2.2),)))
    events = drive(coord, 3.0)
    assert all(e.payload["speaker"] is not None for e in events if e.kind == "commit")
    assert not [e for e in events if e.kind == "speaker"]


def test_solo_still_emits_turns_and_endpoints():
    """One speaker still has utterance boundaries. Only identity is fixed."""
    coord = Coordinator(
        DICTATION,
        engine=_engine(),
        vad=ScriptedVAD(speech=((0.0, 2.2),)),
        turn=ScriptedTurnDetector(fixed=1.0),
    )
    events = drive(coord, 4.0)
    kinds = {str(e.kind) for e in events}
    assert "turn_start" in kinds
    assert "endpoint" in kinds


def test_solo_works_with_no_speaker_frontend_supplied():
    """The common case: a caller who never installed the diarization extra."""
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 2.2),)))
    events = drive(coord, 3.0)
    assert [e.payload["text"] for e in events if e.kind == "commit"] == [
        "one",
        "two",
        "three",
        "four",
        "five",
    ]


def test_presets_pair_the_axes_as_documented():
    """Speaker mode and endpoint mode are orthogonal; presets are combinations."""
    assert DICTATION.speakers.mode is SpeakerMode.SOLO
    assert AGENT.speakers.mode is SpeakerMode.SOLO
    assert CONVERSATION.speakers.mode is SpeakerMode.AUTO

    # A solo voice agent is impatient; solo dictation is not. Same speaker axis,
    # opposite endpoint axis -- which three opaque presets could not express.
    assert AGENT.endpoint.silence_seconds < DICTATION.endpoint.silence_seconds
    assert AGENT.is_solo and DICTATION.is_solo
