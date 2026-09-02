"""Session bookkeeping and admission control.

Both are pure logic, so neither needs a socket or a model to test.
"""

from __future__ import annotations

import pytest

from hearwrite import DICTATION
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.server.session import Admission, Rejected, Session
from hearwrite.vad.fake import ScriptedVAD

from .conftest import pcm


def test_admission_refuses_past_the_limit():
    """Never oversubscribe. Latency under contention is what users notice."""
    gate = Admission(limit=2)
    gate.acquire()
    gate.acquire()
    with pytest.raises(Rejected, match="at capacity"):
        gate.acquire()


def test_admission_frees_a_slot_on_release():
    gate = Admission(limit=1)
    gate.acquire()
    gate.release()
    gate.acquire()
    assert gate.active == 1


def test_release_below_zero_is_harmless():
    """A double release must not silently create capacity."""
    gate = Admission(limit=1)
    gate.release()
    assert gate.active == 0


def _session():
    engine = ScriptedEngine(
        script={1.6: Hypothesis(stable=words("hello there", each=0.4), consumed_to=1.6)}
    )
    return Session(DICTATION, engine=engine, vad=ScriptedVAD(speech=((0.0, 2.0),)))


def test_session_produces_events():
    session = _session()
    events = []
    for _ in range(100):
        events.extend(session.push(pcm(0.02)))
    events.extend(session.finish())
    assert [e.payload["text"] for e in events if e.kind == "commit"] == ["hello", "there"]


def test_lag_is_not_negative_when_ahead_of_real_time():
    """Replaying a file pushes audio far faster than it was recorded."""
    session = _session()
    for _ in range(50):
        list(session.push(pcm(0.02)))
    assert session.lag == 0.0


def test_lag_does_not_reach_any_timestamp():
    """The one wall clock number in the system stays out of the event log."""
    session = _session()
    events = []
    for _ in range(100):
        events.extend(session.push(pcm(0.02)))
    events.extend(session.finish())
    # Every timestamp is a clean multiple of the 20ms frame, which it could not
    # be if wall clock time had leaked in anywhere.
    for event in events:
        assert abs(event.at * 50 - round(event.at * 50)) < 1e-6, event
