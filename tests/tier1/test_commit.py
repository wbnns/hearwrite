"""The commit policy, including C1 confidence gating.

C1 is the cheap approximation of the reference system's learned per-word delay:
commit a confident word immediately, hold an unconfident one briefly to see
whether the engine changes its mind. No training, pure policy, and it trades
latency for accuracy per word rather than as one global constant.
"""

from __future__ import annotations

from dataclasses import replace

from hearwrite import DICTATION, Coordinator
from hearwrite.coordinator.commit import CommitPolicy
from hearwrite.engines.base import Hypothesis, Word
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive


def test_ungated_policy_commits_every_stable_word_at_once():
    policy = CommitPolicy(confidence_gate=0.0)
    hypothesis = Hypothesis(stable=words("a b c", each=0.2), consumed_to=0.6)
    assert [w.text for w in policy.take(hypothesis, 0.6)] == ["a", "b", "c"]


def test_a_word_is_never_committed_twice():
    policy = CommitPolicy()
    first = Hypothesis(stable=words("a b", each=0.2), consumed_to=0.4)
    grown = Hypothesis(stable=words("a b c", each=0.2), consumed_to=0.6)
    assert [w.text for w in policy.take(first, 0.4)] == ["a", "b"]
    assert [w.text for w in policy.take(grown, 0.6)] == ["c"]


def test_confident_words_commit_immediately_under_the_gate():
    policy = CommitPolicy(confidence_gate=0.8, slow_commit_seconds=0.5)
    sure = Hypothesis(stable=(Word("sure", 0.0, 0.3, confidence=0.95),), consumed_to=0.3)
    assert [w.text for w in policy.take(sure, 0.3)] == ["sure"]


def test_unconfident_words_are_held_then_released():
    policy = CommitPolicy(confidence_gate=0.8, slow_commit_seconds=0.5)
    shaky = Hypothesis(stable=(Word("maybe", 0.0, 0.3, confidence=0.4),), consumed_to=0.3)

    assert policy.take(shaky, 0.3) == (), "an unconfident word was committed immediately"
    assert policy.take(shaky, 0.6) == (), "released before the hold elapsed"
    assert [w.text for w in policy.take(shaky, 0.9)] == ["maybe"]


def test_a_held_word_that_changes_is_never_committed_as_the_old_text():
    """The reason holding is worth doing at all."""
    policy = CommitPolicy(confidence_gate=0.8, slow_commit_seconds=0.5)
    wrong = Hypothesis(stable=(Word("their", 0.0, 0.3, confidence=0.4),), consumed_to=0.3)
    right = Hypothesis(stable=(Word("there", 0.0, 0.3, confidence=0.95),), consumed_to=0.3)

    assert policy.take(wrong, 0.3) == ()
    assert [w.text for w in policy.take(right, 0.4)] == ["there"]


def test_flush_releases_everything_still_pending():
    policy = CommitPolicy(confidence_gate=0.9, slow_commit_seconds=5.0)
    hypothesis = Hypothesis(
        stable=(Word("held", 0.0, 0.3, confidence=0.1),),
        tentative=(Word("tail", 0.3, 0.6, confidence=0.1),),
        consumed_to=0.6,
    )
    policy.take(hypothesis, 0.3)
    assert [w.text for w in policy.flush(hypothesis)] == ["held", "tail"]


def test_gating_delays_commits_but_does_not_change_the_transcript():
    """C1 trades latency for confidence. It must not alter what is said."""
    stable = words("the quick brown fox", each=0.4, confidence=0.5)
    script = {1.6: Hypothesis(stable=stable, consumed_to=1.6)}

    plain = Coordinator(
        DICTATION, engine=ScriptedEngine(script=dict(script)), vad=ScriptedVAD(speech=((0.0, 2.0),))
    )
    gated = Coordinator(
        replace(DICTATION, confidence_gate=0.9, slow_commit_seconds=0.4),
        engine=ScriptedEngine(script=dict(script)),
        vad=ScriptedVAD(speech=((0.0, 2.0),)),
    )

    plain_events = drive(plain, 3.0)
    gated_events = drive(gated, 3.0)

    def text_of(events):
        return [e.payload["text"] for e in events if e.kind == "commit"]

    def worst_delay(events):
        return max(e.payload["delay"] for e in events if e.kind == "commit")

    assert text_of(gated_events) == text_of(plain_events)
    assert worst_delay(gated_events) > worst_delay(plain_events), (
        "gating did not actually hold anything back"
    )


# -- C1 early commit ---------------------------------------------------------


def test_early_commit_takes_a_confident_tentative_word():
    """The other direction of C1: skip the wait when the engine is sure.

    Only meaningful for an engine with a real tentative state. A greedy
    transducer settles a word as it emits it, so there is nothing to skip.
    """
    policy = CommitPolicy(early_commit_confidence=0.9)
    hypothesis = Hypothesis(tentative=(Word("sure", 0.0, 0.3, confidence=0.98),))
    assert [w.text for w in policy.take(hypothesis, 0.3)] == ["sure"]


def test_early_commit_is_off_by_default():
    policy = CommitPolicy()
    hypothesis = Hypothesis(tentative=(Word("maybe", 0.0, 0.3, confidence=1.0),))
    assert policy.take(hypothesis, 0.3) == ()


def test_early_commit_takes_only_a_contiguous_prefix():
    """Filtering tentative words individually deletes the ones it skips.

    A confident later word advances the commit frontier past an unconfident
    earlier one, and the earlier word is then behind the frontier forever. This
    really happened: it ate "The build" from the front of a test transcript
    while leaving "is green." intact.
    """
    policy = CommitPolicy(early_commit_confidence=0.95)
    hypothesis = Hypothesis(
        tentative=(
            Word("The", 0.0, 0.20, confidence=0.89),  # below the gate
            Word("build", 0.22, 0.36, confidence=0.95),  # above, but after it
            Word("is", 0.38, 0.60, confidence=0.99),
            Word("green", 0.62, 0.76, confidence=1.0),
        )
    )
    assert policy.take(hypothesis, 1.0) == ()
    assert policy.committed_to == 0.0, "the frontier moved past an uncommitted word"


def test_early_commit_stops_at_the_first_unconfident_word():
    policy = CommitPolicy(early_commit_confidence=0.9)
    hypothesis = Hypothesis(
        tentative=(
            Word("one", 0.0, 0.2, confidence=0.99),
            Word("two", 0.2, 0.4, confidence=0.95),
            Word("three", 0.4, 0.6, confidence=0.10),
            Word("four", 0.6, 0.8, confidence=0.99),
        )
    )
    assert [w.text for w in policy.take(hypothesis, 1.0)] == ["one", "two"]


def test_a_held_word_blocks_the_words_after_it():
    """The same no-holes rule, from the other direction.

    If an unconfident word is held while a later confident one commits, the held
    word can never be emitted without contradicting the order already sent.
    """
    policy = CommitPolicy(confidence_gate=0.9, slow_commit_seconds=5.0)
    hypothesis = Hypothesis(
        stable=(
            Word("hold", 0.0, 0.3, confidence=0.10),
            Word("me", 0.3, 0.6, confidence=0.99),
        )
    )
    assert policy.take(hypothesis, 0.6) == ()
    assert policy.committed_to == 0.0


def test_the_committed_sequence_never_has_a_gap():
    """Property check over a stream of mixed confidence hypotheses."""
    policy = CommitPolicy(early_commit_confidence=0.8)
    confidences = [0.99, 0.5, 0.95, 0.99, 0.3, 0.99]
    emitted: list[Word] = []
    for i in range(1, len(confidences) + 1):
        tentative = tuple(
            Word(f"w{j}", j * 0.2, j * 0.2 + 0.18, confidence=confidences[j]) for j in range(i)
        )
        emitted.extend(policy.take(Hypothesis(tentative=tentative), i * 0.2))

    names = [w.text for w in emitted]
    assert names == sorted(names, key=lambda n: int(n[1:])), names
    expected = [f"w{j}" for j in range(len(names))]
    assert names == expected, f"a hole appeared: {names}"


def test_a_fragment_is_never_committed_early():
    """A transducer's trailing tentative entry is half a word, not a whole one.

    Committing it early does not deliver a word sooner, it delivers "Ja" instead
    of "January", permanently. Measured before this guard existed: "The Times
    January 3rd 2009 Chancellor" came back as "The Time Ja third 2009 Ch".
    """
    policy = CommitPolicy(early_commit_confidence=0.5)
    fragment = Hypothesis(
        tentative=(Word("Ja", 0.0, 0.2, confidence=0.99),),
        tentative_is_fragment=True,
    )
    assert policy.take(fragment, 1.0) == ()
    assert policy.committed_to == 0.0


def test_a_whole_tentative_word_is_still_eligible():
    """LocalAgreement yields whole words, so early commit still applies there."""
    policy = CommitPolicy(early_commit_confidence=0.5)
    whole = Hypothesis(tentative=(Word("January", 0.0, 0.5, confidence=0.99),))
    assert [w.text for w in policy.take(whole, 1.0)] == ["January"]


def test_the_transducer_adapter_declares_its_tentative_is_a_fragment():
    """The guard is only as good as the engine that sets the flag."""
    import inspect

    from hearwrite.engines import sherpa

    assert "tentative_is_fragment=True" in inspect.getsource(sherpa)
