"""The polish pass, and the invariant that makes it safe.

A second model rewriting committed text is a direct threat to the one rule, so
the whole design rests on one check: a polish may change rendering and nothing
else. These tests are that check.
"""

from __future__ import annotations

from hearwrite import DICTATION, Coordinator
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.punctuate.base import preserves_words, words_of
from hearwrite.punctuate.fake import ScriptedPunctuator
from hearwrite.transcript import committed_text, display_text
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive

# -- the invariant -----------------------------------------------------------


def test_punctuation_and_casing_are_not_content():
    assert preserves_words("the build is green", "The build is green.")
    assert preserves_words("hello there", "Hello, there!")


def test_a_dropped_or_added_word_is_content():
    assert not preserves_words("the build is green", "The build is.")
    assert not preserves_words("the build", "The build is green.")


def test_a_substituted_word_is_content():
    assert not preserves_words("their build", "There build.")


def test_word_order_matters():
    assert not preserves_words("green is build", "Build is green.")


def test_words_of_ignores_punctuation_and_case():
    assert words_of("Charlie's, running. UP!") == ["charlie's", "running", "up"]


# -- through the Coordinator -------------------------------------------------


def _run(punctuator, text="the build is green"):
    coord = Coordinator(
        DICTATION,
        engine=ScriptedEngine(
            script={2.4: Hypothesis(stable=words(text, each=0.4), consumed_to=2.4)}
        ),
        vad=ScriptedVAD(speech=((0.0, 2.4),)),
        turn=ScriptedTurnDetector(fixed=1.0),
        punctuator=punctuator,
    )
    return coord, drive(coord, 5.0)


def test_a_polish_is_emitted_for_a_finished_utterance():
    _, events = _run(ScriptedPunctuator())
    polished = [e for e in events if e.kind == "polished"]
    assert polished, "no polish was emitted"
    assert polished[0].payload["text"] == "The build is green."


def test_the_committed_words_are_untouched_by_a_polish():
    """The whole point. The polish is a second document, not an edit."""
    _, events = _run(ScriptedPunctuator())
    assert committed_text(events) == "the build is green"
    assert display_text(events) == "The build is green."


def test_a_polish_that_changes_the_words_is_thrown_away():
    """A model that rewrites content must not reach a consumer at all."""
    liar = ScriptedPunctuator(by_text={"the build is green": "The build is red."})
    _, events = _run(liar)
    assert liar.calls, "the punctuator was never called"
    assert not [e for e in events if e.kind == "polished"], "a rewrite was emitted"
    assert display_text(events) == "the build is green"


def test_a_polish_that_drops_a_word_is_thrown_away():
    thief = ScriptedPunctuator(by_text={"the build is green": "The build is."})
    _, events = _run(thief)
    assert not [e for e in events if e.kind == "polished"]


def test_no_polish_without_a_punctuator():
    _, events = _run(None)
    assert not [e for e in events if e.kind == "polished"]
    assert display_text(events) == committed_text(events)


def test_a_polish_names_the_span_it_replaces():
    _, events = _run(ScriptedPunctuator())
    polished = next(e for e in events if e.kind == "polished")
    commits = [e for e in events if e.kind == "commit"]
    assert polished.payload["from_seq"] == commits[0].seq
    assert polished.payload["to_seq"] == commits[-1].seq


def test_the_previous_utterance_is_offered_as_context():
    """An utterance is a fragment. Without context every one is punctuated as
    though it began a sentence, which capitalises the middle of a clause."""
    punctuator = ScriptedPunctuator()
    engine = ScriptedEngine(
        script={
            1.6: Hypothesis(stable=words("the build is", each=0.4), consumed_to=1.6),
            4.4: Hypothesis(
                stable=words("the build is", each=0.4) + words("green today", start=3.2, each=0.4),
                consumed_to=4.4,
            ),
        }
    )
    coord = Coordinator(
        DICTATION,
        engine=engine,
        vad=ScriptedVAD(speech=((0.0, 1.6), (3.2, 4.4))),
        turn=ScriptedTurnDetector(fixed=1.0),
        punctuator=punctuator,
    )
    drive(coord, 7.0)
    assert len(punctuator.contexts) >= 2, punctuator.calls
    assert punctuator.contexts[0] == "", "the first utterance has no predecessor"
    assert punctuator.contexts[1], "the second was polished with no context"


def test_display_text_falls_back_to_commits_for_unpolished_spans():
    """A consumer that only partly polishes still gets every word."""
    from hearwrite.events import Event, EventKind

    events = [
        Event(0, EventKind.COMMIT, 1.0, {"text": "one", "audio_start": 0.0, "audio_end": 0.3}),
        Event(1, EventKind.COMMIT, 1.0, {"text": "two", "audio_start": 0.3, "audio_end": 0.6}),
        Event(2, EventKind.POLISHED, 1.2, {"text": "One.", "from_seq": 0, "to_seq": 0}),
    ]
    assert display_text(events) == "One. two"


# -- when the polish should run at all ---------------------------------------


def test_polish_is_off_behind_a_recogniser_that_punctuates():
    """Not merely redundant: measured on a real recording, running the
    punctuation model over Nemotron's output lowercased "Times" and
    "Chancellor" and capitalised "Brink". It makes good text worse.
    """
    from hearwrite.pipeline import Backends, _wants_punctuation

    assert not _wants_punctuation(Backends(model="nemotron-3.5-160ms"))


def test_polish_is_on_behind_a_recogniser_that_does_not():
    from hearwrite.pipeline import Backends, _wants_punctuation

    assert _wants_punctuation(Backends(model="zipformer-en"))


def test_polish_is_off_behind_whisper():
    from hearwrite.pipeline import Backends, _wants_punctuation

    assert not _wants_punctuation(Backends(engine="whisper"))


def test_an_explicit_setting_beats_the_default_either_way():
    from hearwrite.pipeline import Backends, _wants_punctuation

    assert _wants_punctuation(Backends(model="nemotron-3.5-160ms", punctuate=True))
    assert not _wants_punctuation(Backends(model="zipformer-en", punctuate=False))


def test_the_registry_records_which_models_punctuate():
    """The answer belongs with the model, not in a branch somewhere."""
    from hearwrite.models import REGISTRY

    assert REGISTRY["nemotron-3.5-160ms"].punctuates
    assert not REGISTRY["zipformer-en"].punctuates
