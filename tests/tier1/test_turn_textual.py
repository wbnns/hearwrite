"""Endpointing from the recogniser's own punctuation.

A recogniser that writes "shells." has already judged that a sentence ended,
using the whole context it was trained on. On some material that judgement is
markedly better than a separate turn detector's, and it is free.
"""

from __future__ import annotations

import pytest

from hearwrite.turn.textual import UNKNOWN, PunctuationTurnDetector


@pytest.mark.parametrize("text", ["that is done.", "is it?", "stop!", 'he said "yes."'])
def test_terminal_punctuation_means_finished(text):
    assert PunctuationTurnDetector().completeness(text) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "the muttering",
        "a patient etherized upon a",
        "let us go through certain half deserted streets,",
        "to an overwhelming question",
    ],
)
def test_no_terminal_punctuation_means_mid_thought(text):
    """The exact fragments smart-turn scored 0.70 and higher on a real reading.

    Verse falls in pitch at every line end, so an audio native model reads each
    one as finality. The transcript does not.
    """
    assert PunctuationTurnDetector().completeness(text) == 0.0


def test_a_comma_is_not_the_end_of_anything():
    assert PunctuationTurnDetector().completeness("well, then,") == 0.0


def test_an_empty_transcript_abstains():
    """Vetoing on no evidence would leave nothing but the timeout forever."""
    assert PunctuationTurnDetector().completeness("") == UNKNOWN
    assert PunctuationTurnDetector().completeness("   ") == UNKNOWN


def test_it_ignores_the_audio_it_is_handed():
    detector = PunctuationTurnDetector()
    assert detector.completeness("done.", b"\x00\x00" * 1000) == 1.0


def test_auto_picks_it_behind_a_punctuating_recogniser():
    from hearwrite.pipeline import Backends, _recogniser_punctuates

    assert _recogniser_punctuates(Backends(model="nemotron-3.5-160ms"))
    assert not _recogniser_punctuates(Backends(model="zipformer-en"))


def test_auto_falls_back_where_there_is_no_punctuation_to_read():
    """Ask this detector about block capitals with no full stops and it vetoes
    every endpoint forever, so it must not be chosen there."""
    from hearwrite.pipeline import Backends, _recogniser_punctuates

    assert not _recogniser_punctuates(Backends(model="zipformer-en"))
