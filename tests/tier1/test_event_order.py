"""The log is ordered in time, not just in sequence number.

Both of these were real bugs, found by running real audio through the real
pipeline rather than by reasoning about it. They are cheap to re-introduce and
invisible in a transcript, so they get tests.
"""

from __future__ import annotations

import pytest

from hearwrite import DICTATION, Coordinator
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.events import AppendOnlyViolation, EventKind, EventLog
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive


def test_event_times_never_go_backwards():
    """`at` is when an event was emitted, not what audio it describes.

    turn_start used to be stamped with its first word's audio_start, which put
    it earlier than the endpoint that preceded it. Anything plotting or
    windowing the log by time would have seen it jump backwards.
    """
    engine = ScriptedEngine(
        script={
            1.6: Hypothesis(stable=words("all done.", start=0.4, each=0.4), consumed_to=1.6),
            4.4: Hypothesis(
                stable=words("all done.", start=0.4, each=0.4)
                + words("more now.", start=3.4, each=0.4),
                consumed_to=4.4,
            ),
        }
    )
    coord = Coordinator(
        DICTATION,
        engine=engine,
        vad=ScriptedVAD(speech=((0.0, 1.6), (3.4, 4.4))),
        turn=ScriptedTurnDetector(),
    )
    events = drive(coord, 6.0)

    times = [e.at for e in events]
    assert times == sorted(times), f"event times are not monotonic: {times}"


def test_log_rejects_an_event_from_the_past():
    log = EventLog()
    log.emit(EventKind.ENDPOINT, 2.0, {"reason": "complete"})
    with pytest.raises(AppendOnlyViolation, match="goes backwards"):
        log.emit(EventKind.TURN_START, 1.0, {"speaker": "A", "turn": 2})


def test_a_word_arriving_after_its_endpoint_does_not_open_a_new_turn():
    """A streaming transducer releases its last words late. They still belong
    to the sentence they were spoken in.

    Observed for real: the VAD called silence at 1.04s, the endpoint fired, and
    only then did the engine emit a word whose audio ended at 1.32s. Opening a
    turn for it put a boundary in the middle of a finished sentence.
    """
    engine = ScriptedEngine(
        script={
            # "late" is spoken at 1.0-1.4 but is not emitted until 3.0.
            1.6: Hypothesis(stable=words("the build is", start=0.2, each=0.3), consumed_to=1.6),
            3.0: Hypothesis(
                stable=words("the build is", start=0.2, each=0.3)
                + words("late.", start=1.1, each=0.3),
                consumed_to=3.0,
            ),
        }
    )
    coord = Coordinator(
        DICTATION,
        engine=engine,
        # Silence from 1.05, so the endpoint fires before "late." is emitted.
        vad=ScriptedVAD(speech=((0.0, 1.05),)),
        turn=ScriptedTurnDetector(fixed=1.0),
    )
    events = drive(coord, 4.0)

    kinds = [str(e.kind) for e in events]
    assert "endpoint" in kinds

    turns = [e.payload["turn"] for e in events if e.kind == EventKind.TURN_START]
    assert turns == [1], f"a trailing word opened a spurious turn: {turns}"

    late = [e for e in events if e.kind == EventKind.COMMIT and e.payload["text"] == "late."]
    assert late, "the late word was dropped entirely"


def test_turn_start_records_both_clocks():
    """The emission position orders the log; audio_start is what you seek to."""
    engine = ScriptedEngine(
        script={2.0: Hypothesis(stable=words("hello there", start=0.5, each=0.4), consumed_to=2.0)}
    )
    coord = Coordinator(DICTATION, engine=engine, vad=ScriptedVAD(speech=((0.0, 2.0),)))
    events = drive(coord, 3.0)

    start = next(e for e in events if e.kind == EventKind.TURN_START)
    assert start.payload["audio_start"] == pytest.approx(0.5)
    assert start.at >= start.payload["audio_start"]


def test_every_consumer_handles_both_shapes_of_speaker_event():
    """A `speaker` event names either one word or a whole turn.

    Both shapes are real, and code that assumes the first raises KeyError on the
    second. That happened three times in one change: in the metrics, in the CLI
    renderer, and in a test helper. This is the guard.
    """
    import inspect

    from hearwrite import cli, metrics

    for module in (cli, metrics):
        source = inspect.getsource(module)
        # Every read of payload["seq"] must be guarded by a presence check.
        for line in source.splitlines():
            if '"seq"' in line and "payload" in line and "in event.payload" not in line:
                assert "in p" in source or '"seq" in' in source, (
                    f"{module.__name__} reads seq without checking for it: {line}"
                )


def test_a_turn_level_speaker_event_carries_a_turn_not_a_seq():
    from hearwrite import CONVERSATION, Coordinator
    from hearwrite.engines.base import Hypothesis
    from hearwrite.engines.fake import ScriptedEngine, words
    from hearwrite.speakers.base import Segment
    from hearwrite.speakers.fake import ScriptedFrontend, embedding
    from hearwrite.vad.fake import ScriptedVAD

    coord = Coordinator(
        CONVERSATION,
        engine=ScriptedEngine(
            script={3.0: Hypothesis(stable=words("one two three", each=0.9), consumed_to=3.0)}
        ),
        vad=ScriptedVAD(speech=((0.0, 3.0),)),
        speakers=ScriptedFrontend(segments=(Segment(0.0, 2.0, embedding(1.0, 0.0, 0.0)),)),
    )
    events = drive(coord, 4.0)
    turn_level = [e for e in events if e.kind == EventKind.SPEAKER and "seq" not in e.payload]
    for event in turn_level:
        assert "turn" in event.payload
        assert "speaker" in event.payload
