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
