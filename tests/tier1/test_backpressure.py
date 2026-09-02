"""Backpressure: shed provisional work, never final work.

Growing a buffer silently is the worst option and the easiest one to write by
accident, so the degraded state is both explicit and announced.
"""

from __future__ import annotations

from hearwrite import DICTATION, Coordinator
from hearwrite.coordinator.backpressure import BackpressureGate
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive, pcm


def _engine():
    return ScriptedEngine(
        script={
            0.8: Hypothesis(tentative=words("thinking about", each=0.4), consumed_to=0.8),
            1.6: Hypothesis(stable=words("settled text here", each=0.4), consumed_to=1.6),
        }
    )


def test_gate_reports_only_transitions():
    gate = BackpressureGate(max_lag_seconds=1.0)
    assert gate.observe(0.2) is None
    assert gate.observe(2.0) is True  # entered degradation
    assert gate.observe(2.5) is None  # still degraded, no repeat
    assert gate.observe(0.1) is False  # recovered
    assert gate.observe(0.1) is None


def test_partials_are_dropped_while_degraded():
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 2.0),)))
    events = drive(coord, 2.4, lag=9.0)
    assert not [e for e in events if e.kind == "partial"], "partials survived degradation"


def test_commits_are_never_dropped_while_degraded():
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 2.0),)))
    events = drive(coord, 2.4, lag=9.0)
    assert [e.payload["text"] for e in events if e.kind == "commit"] == [
        "settled",
        "text",
        "here",
    ]


def test_client_is_told_when_quality_drops():
    """Silently getting worse is its own failure mode."""
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 2.0),)))
    events = drive(coord, 2.4, lag=9.0)
    degraded = [e for e in events if e.kind == "degraded"]
    assert degraded, "the pipeline degraded without telling anyone"
    assert degraded[0].payload["degraded"] is True
    assert degraded[0].payload["dropping"] == "partial"


def test_recovery_is_announced_too():
    coord = Coordinator(DICTATION, engine=_engine(), vad=ScriptedVAD(speech=((0.0, 3.0),)))
    frame = pcm(0.02)
    events = []
    for i in range(150):
        events.extend(coord.push(frame, lag=9.0 if i < 60 else 0.0))
    events.extend(coord.finish())

    flags = [e.payload["degraded"] for e in events if e.kind == "degraded"]
    assert flags == [True, False]
