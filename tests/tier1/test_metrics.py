"""Diarization metrics.

The metric has to be right before any number it produces means anything, and two
things could silently make it lie: scoring arbitrary labels without mapping them
onto ground truth, and folding abstention into error.
"""

from __future__ import annotations

from hearwrite.events import Event, EventKind
from hearwrite.metrics import Turn, evaluate, load_turns


def commit(seq, text, start, end, speaker):
    return Event(
        seq=seq,
        kind=EventKind.COMMIT,
        at=end + 0.4,
        payload={
            "text": text,
            "audio_start": start,
            "audio_end": end,
            "speaker": speaker,
            "delay": 0.4,
        },
    )


TURNS = (Turn("alice", 0.0, 2.0), Turn("bob", 2.0, 4.0))


def test_arbitrary_labels_are_mapped_before_scoring():
    """The pipeline invents A and B; ground truth says alice and bob.

    Without a mapping step a perfect system scores zero.
    """
    events = [
        commit(0, "one", 0.0, 1.0, "A"),
        commit(1, "two", 1.0, 2.0, "A"),
        commit(2, "three", 2.1, 3.0, "B"),
        commit(3, "four", 3.0, 3.9, "B"),
    ]
    report = evaluate(events, TURNS)
    assert report.confusion_rate == 0.0
    assert report.correct == 4


def test_the_mapping_cannot_be_gamed_by_label_order():
    """Same transcript, labels swapped. The score must not move."""
    swapped = [
        commit(0, "one", 0.0, 1.0, "B"),
        commit(1, "two", 1.0, 2.0, "B"),
        commit(2, "three", 2.1, 3.0, "A"),
        commit(3, "four", 3.0, 3.9, "A"),
    ]
    assert evaluate(swapped, TURNS).confusion_rate == 0.0


def test_confusion_counts_only_labelled_words():
    """Abstention is not error. A system that declines to guess must not be
    scored as though it guessed wrong -- that would punish exactly the behaviour
    the append-only rule requires.
    """
    events = [
        commit(0, "one", 0.0, 1.0, "A"),
        commit(1, "two", 1.0, 2.0, None),
        commit(2, "three", 2.1, 3.0, "B"),
        commit(3, "four", 3.0, 3.9, None),
    ]
    report = evaluate(events, TURNS)
    assert report.labelled == 2
    assert report.unlabelled == 2
    assert report.confusion_rate == 0.0
    assert report.null_rate == 0.5


def test_a_wrong_label_is_confusion():
    events = [
        commit(0, "one", 0.0, 1.0, "A"),
        commit(1, "two", 1.0, 2.0, "A"),
        commit(2, "three", 2.1, 3.0, "A"),  # bob's turn, called A
        commit(3, "four", 3.0, 3.9, "B"),
    ]
    report = evaluate(events, TURNS)
    assert report.confused == 1
    assert report.confusion_rate == 0.25


def test_a_later_speaker_event_fills_a_null_and_counts():
    """The fill mechanism is part of the pipeline, so it is part of the score."""
    events = [
        commit(0, "one", 0.0, 1.0, None),
        Event(
            seq=1,
            kind=EventKind.SPEAKER,
            at=2.0,
            payload={"seq": 0, "speaker": "A", "audio_start": 0.0, "audio_end": 1.0},
        ),
        commit(2, "two", 2.1, 3.0, "B"),
    ]
    report = evaluate(events, TURNS)
    assert report.unlabelled == 0
    assert report.confusion_rate == 0.0


def test_words_outside_every_turn_are_not_scored():
    """Ground truth cannot say whether a word in unlabelled audio is right."""
    events = [commit(0, "one", 0.0, 1.0, "A"), commit(1, "stray", 90.0, 91.0, "A")]
    assert evaluate(events, TURNS).words == 1


def test_turn_label_latency_is_measured_from_the_turn_start():
    """A diarizer that is eventually right but late feels broken while scoring
    well, so latency is tracked separately from accuracy."""
    events = [commit(0, "one", 1.0, 1.5, "A"), commit(1, "two", 2.5, 3.0, "B")]
    report = evaluate(events, TURNS)
    assert len(report.turn_label_latency) == 2
    assert all(v >= 0 for v in report.turn_label_latency)


def test_load_turns_reads_the_fixture_format():
    turns = load_turns({"turns": [{"speaker": "x", "start": 0, "end": 1.5}]})
    assert turns[0].speaker == "x"
    assert turns[0].covers(0.7)
    assert not turns[0].covers(2.0)


def test_empty_input_does_not_divide_by_zero():
    report = evaluate([], TURNS)
    assert report.null_rate == 0.0
    assert report.confusion_rate == 0.0
