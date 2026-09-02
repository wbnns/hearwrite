"""THE ONE RULE: committed output is append-only.

If any test in this file fails, HearWrite is broken in the way that matters
most. A consumer that ignored every `partial` and trusted every `commit` would
have been lied to.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from hearwrite import CONVERSATION, DICTATION, Coordinator
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.events import AppendOnlyViolation, EventKind, EventLog
from hearwrite.speakers.fake import ScriptedFrontend
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

from .conftest import committed, drive, pcm


def test_committed_text_only_grows(sentence_engine):
    coord = Coordinator(
        DICTATION,
        engine=sentence_engine,
        vad=ScriptedVAD(speech=((0.0, 2.8),)),
        turn=ScriptedTurnDetector(),
    )
    frame = pcm(0.02)
    seen = [""]
    for _ in range(200):
        coord.push(frame)
        seen.append(coord.log.committed_text)
    coord.finish()

    for earlier, later in pairwise(seen):
        assert later.startswith(earlier), f"{earlier!r} is not a prefix of {later!r}"


def test_engine_retraction_never_reaches_a_commit():
    """An engine that changes its mind about a STABLE word must not corrupt the log.

    This is the nightmare case: the adapter promised a word, then withdrew it.
    The commit frontier has already moved past that audio, so the retraction is
    dropped rather than emitted as a contradiction.
    """
    engine = ScriptedEngine(
        script={
            1.0: Hypothesis(stable=words("the cat sat", each=0.3), consumed_to=1.0),
            2.0: Hypothesis(stable=words("the dog sat down", each=0.3), consumed_to=2.0),
        }
    )
    coord = Coordinator(DICTATION, engine=engine, vad=ScriptedVAD(speech=((0.0, 2.5),)))
    events = drive(coord, 3.0)

    text = coord.log.committed_text
    assert text.startswith("the cat sat"), text
    # "dog" contradicted an already-committed word, so it never appears.
    assert "dog" not in text
    starts = [e.payload["audio_start"] for e in events if e.kind == EventKind.COMMIT]
    assert starts == sorted(starts), "commits went backwards in time"


def test_log_rejects_a_backwards_commit():
    """The EventLog itself refuses a contradiction, even if a caller tries."""
    log = EventLog()
    log.emit(
        EventKind.COMMIT,
        1.0,
        {"text": "one", "audio_start": 0.0, "audio_end": 0.5, "speaker": None},
    )
    with pytest.raises(AppendOnlyViolation):
        log.emit(
            EventKind.COMMIT,
            1.1,
            {"text": "two", "audio_start": 0.1, "audio_end": 0.6, "speaker": None},
        )


def test_partials_never_overlap_committed_audio(sentence_engine):
    coord = Coordinator(DICTATION, engine=sentence_engine, vad=ScriptedVAD(speech=((0.0, 2.8),)))
    events = drive(coord, 3.2)
    frontier = 0.0
    for event in events:
        if event.kind == EventKind.COMMIT:
            frontier = max(frontier, event.payload["audio_end"])
        elif event.kind == EventKind.PARTIAL:
            assert event.payload["audio_start"] >= frontier - 1e-6, (
                "a partial re-proposed audio that was already committed"
            )


def test_speaker_events_only_fill_nulls_never_change_a_label(two_speaker_segments):
    """A `speaker` event may supply a missing label; it may never replace one."""
    engine = ScriptedEngine(
        script={3.7: Hypothesis(stable=words("one two three four", each=0.5), consumed_to=3.7)}
    )
    coord = Coordinator(
        CONVERSATION,
        engine=engine,
        vad=ScriptedVAD(speech=((0.0, 3.8),)),
        speakers=ScriptedFrontend(segments=two_speaker_segments),
    )
    events = drive(coord, 4.5)

    labelled = {e.seq: e.payload["speaker"] for e in events if e.kind == EventKind.COMMIT}
    for event in events:
        if event.kind == EventKind.SPEAKER and "seq" in event.payload:
            target = event.payload["seq"]
            assert labelled[target] is None, (
                f"speaker event tried to overwrite an existing label on seq {target}"
            )


def test_committed_prefix_survives_a_degraded_stretch(sentence_engine):
    """Backpressure drops partials. It must never drop or reorder a commit."""
    clean = Coordinator(DICTATION, engine=sentence_engine, vad=ScriptedVAD(speech=((0.0, 2.8),)))
    baseline = committed(drive(clean, 3.2))

    engine2 = ScriptedEngine(script=dict(sentence_engine.script))
    stressed = Coordinator(DICTATION, engine=engine2, vad=ScriptedVAD(speech=((0.0, 2.8),)))
    # Run the whole session far behind real time.
    degraded = committed(drive(stressed, 3.2, lag=9.0))

    assert degraded == baseline
