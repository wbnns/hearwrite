"""Inverse text normalisation.

The whole risk here is over eagerness. Under conversion is invisible; over
conversion corrupts a transcript, and the corrupting case turned up in a real
microphone test rather than in imagination.
"""

from __future__ import annotations

import pytest

from hearwrite.polish.itn import normalise
from hearwrite.polish.normalisation import NormalisationStage


@pytest.mark.parametrize(
    "spoken,written",
    [
        ("january third two thousand nine", "january 3rd 2009"),
        ("there were three hundred and forty two people", "there were 342 people"),
        ("that costs twenty five dollars", "that costs $25"),
        ("we grew ten percent last quarter", "we grew 10% last quarter"),
        ("it cost five pounds", "it cost £5"),
        ("about fifteen hundred", "about 1500"),
    ],
)
def test_spoken_numbers_become_figures(spoken, written):
    assert normalise(spoken) == written


@pytest.mark.parametrize(
    "text",
    [
        # The case that matters most: a count-off is not the number 123.
        "test one two three",
        "one two three four",
        # A lone number in prose reads better as a word.
        "call me at four",
        "ten people came",
        # Ordinals away from a month are ordinary words.
        "he came second in the race",
        "no numbers here at all",
    ],
)
def test_what_must_not_be_converted_is_not(text):
    assert normalise(text) == text


def test_a_year_said_as_two_values_is_left_alone():
    """ "nineteen eighty four" is 1984 to a person and two numbers to a rule.

    Converting it needs year specific heuristics that would also mangle
    "nineteen eighty four people". Leaving it is the safe half of the trade, and
    it is a documented limitation rather than an oversight.
    """
    assert normalise("in nineteen eighty four") == "in nineteen eighty four"


def test_punctuation_around_a_number_survives():
    assert normalise("we hired twenty five, then more") == "we hired 25, then more"


def test_the_stage_rejects_a_rule_that_touched_a_non_number():
    """The stage's own check: number words may change, nothing else may."""
    stage = NormalisationStage()
    assert stage.verify("we hired twenty five people", "we hired 25 people")
    assert not stage.verify("we hired twenty five people", "we fired 25 people")


def test_the_stage_runs_after_the_models():
    """Run it first and the punctuation model uppercases the digits: "3RD"."""
    from hearwrite.polish.punctuation import PunctuationStage

    assert NormalisationStage().order > PunctuationStage.order
