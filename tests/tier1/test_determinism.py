"""Determinism: the same audio must produce the same log, however it is fed.

Two separate properties, often confused:

  * CHUNK-SIZE INVARIANCE -- the committed text does not depend on how the audio
    was sliced. A caller sending 1s buffers and one sending 20ms frames get the
    same transcript.
  * REPLAY DETERMINISM -- the log does not depend on how fast the audio arrived.
    A fixture replayed at 10x in CI produces byte-identical output to real time.

The second is the one that quietly rots. It holds only because nothing in the
Coordinator reads a system clock; the moment someone adds a `time.time()` to
compute a timestamp, tests start passing on a fast laptop and failing in CI.
"""

from __future__ import annotations

import time

from hearwrite import CONVERSATION, DICTATION, Coordinator
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.protocol import encode
from hearwrite.speakers.fake import ScriptedFrontend
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.fake import ScriptedVAD

from .conftest import committed, drive, pcm

SCRIPT = {
    1.6: Hypothesis(stable=words("the quick brown fox", each=0.4), consumed_to=1.6),
    3.2: Hypothesis(stable=words("the quick brown fox jumped over it.", each=0.4), consumed_to=3.2),
}


def _fresh(chunk_aware_speech=(0.0, 3.4)):
    return Coordinator(
        DICTATION,
        engine=ScriptedEngine(script=dict(SCRIPT)),
        vad=ScriptedVAD(speech=(chunk_aware_speech,)),
        turn=ScriptedTurnDetector(),
    )


def test_chunk_size_invariance():
    """200ms, 500ms and 1s buffers must yield an identical committed prefix."""
    results = {}
    for chunk in (0.02, 0.2, 0.5, 1.0):
        results[chunk] = committed(drive(_fresh(), 4.0, chunk=chunk))

    baseline = results[0.02]
    assert baseline, "fixture produced no commits at all"
    for chunk, got in results.items():
        assert got == baseline, f"chunk size {chunk}s changed the transcript"


def test_replay_speed_does_not_change_the_log():
    """Wall-clock pacing must not appear anywhere in the serialized log."""
    fast = [encode(e) for e in drive(_fresh(), 4.0)]

    slow_coord = _fresh()
    frame = pcm(0.02)
    slow: list[str] = []
    for i in range(200):
        slow.extend(encode(e) for e in slow_coord.push(frame))
        if i % 50 == 0:
            time.sleep(0.002)  # real elapsed time between pushes
    slow.extend(encode(e) for e in slow_coord.finish())

    assert slow == fast, "the event log changed when audio arrived at a different rate"


def test_lag_never_reaches_a_timestamp():
    """Backpressure input must alter which events fire, never their times."""
    calm = drive(_fresh(), 4.0, lag=0.0)
    busy = drive(_fresh(), 4.0, lag=0.4)  # below the threshold, so no degradation

    assert [encode(e) for e in calm] == [encode(e) for e in busy]


def test_speaker_labels_are_chunk_size_invariant(two_speaker_segments):
    """Alignment must not depend on where chunk boundaries happen to fall."""
    results = {}
    for chunk in (0.02, 0.1, 0.4):
        coord = Coordinator(
            CONVERSATION,
            engine=ScriptedEngine(
                script={3.7: Hypothesis(stable=words("a b c d e f g", each=0.5), consumed_to=3.7)}
            ),
            vad=ScriptedVAD(speech=((0.0, 3.8),)),
            speakers=ScriptedFrontend(segments=two_speaker_segments),
        )
        events = drive(coord, 4.4, chunk=chunk)
        results[chunk] = [
            (e.payload["text"], e.payload["speaker"]) for e in events if e.kind == "commit"
        ]

    baseline = results[0.02]
    assert baseline
    for chunk, got in results.items():
        assert got == baseline, f"chunk size {chunk}s changed speaker assignment"
