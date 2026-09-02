"""The conjunctive endpoint gate.

The behaviour worth protecting: a speaker pausing mid-sentence is not cut off,
and a speaker who trails off entirely does not hang the session. Those two pull
in opposite directions, which is why the gate is a state machine with a fallback
rather than a threshold.
"""

from __future__ import annotations

from hearwrite import DICTATION, Coordinator
from hearwrite.coordinator.endpoint import EndpointGate, Reason
from hearwrite.coordinator.policy import EndpointPolicy
from hearwrite.engines.base import Hypothesis
from hearwrite.engines.fake import ScriptedEngine, words
from hearwrite.turn.fake import ScriptedTurnDetector
from hearwrite.vad.base import SpeechState
from hearwrite.vad.fake import ScriptedVAD

from .conftest import drive

POLICY = EndpointPolicy(silence_seconds=0.5, completeness_threshold=0.6, max_silence_seconds=2.0)


def feed(gate, spans, completeness, step=0.1, until=6.0):
    """Drive the gate over a timeline, returning every endpoint it fired."""
    fired = []
    at = 0.0
    while at < until:
        speaking = any(s <= at < e for s, e in spans)
        state = SpeechState(speaking=speaking, at=at, since=at)
        score = completeness(at) if callable(completeness) else completeness
        result = gate.observe(state, score)
        if result:
            fired.append(result)
        at = round(at + step, 6)
    return fired


def test_silence_alone_is_not_an_endpoint():
    """The whole point: a mid-thought pause must not end the turn."""
    gate = EndpointGate(POLICY)
    fired = feed(gate, [(0.0, 1.0)], completeness=0.0, until=1.9)
    assert fired == [], "an incomplete utterance was endpointed on silence alone"


def test_both_gates_agreeing_fires_an_endpoint():
    gate = EndpointGate(POLICY)
    fired = feed(gate, [(0.0, 1.0)], completeness=1.0, until=1.9)
    assert len(fired) == 1
    assert fired[0].reason == Reason.COMPLETE
    assert fired[0].at >= 1.5 - 1e-9


def test_timeout_fallback_rescues_a_speaker_who_trails_off():
    """Without this the session hangs forever on an unfinished sentence."""
    gate = EndpointGate(POLICY)
    fired = feed(gate, [(0.0, 1.0)], completeness=0.0, until=4.0)
    assert len(fired) == 1
    assert fired[0].reason == Reason.TIMEOUT
    assert fired[0].at >= 3.0 - 1e-9


def test_only_one_endpoint_per_run_of_silence():
    gate = EndpointGate(POLICY)
    fired = feed(gate, [(0.0, 1.0)], completeness=1.0, until=5.0)
    assert len(fired) == 1


def test_a_mid_thought_pause_does_not_end_the_turn():
    """The scenario the whole conjunctive design exists for.

    Someone says "what's the weather in", pauses to think, then says "Menlo
    Park". The pause is long enough for the acoustic gate, so an acoustic-only
    endpointer would cut them off. The semantic gate says the utterance is
    unfinished, so nothing fires until they actually finish.
    """
    gate = EndpointGate(POLICY)
    finished_at = 2.6

    def completeness(at):
        return 1.0 if at >= finished_at else 0.0

    fired = feed(gate, [(0.0, 1.0), (1.8, finished_at)], completeness=completeness, until=4.0)

    assert len(fired) == 1, f"expected one endpoint at the real end, got {fired}"
    assert fired[0].at > finished_at
    assert fired[0].reason == Reason.COMPLETE


def test_silence_before_any_speech_is_not_an_endpoint():
    gate = EndpointGate(POLICY)
    fired = feed(gate, [], completeness=1.0, until=5.0)
    assert fired == []


def test_flush_closes_an_open_utterance():
    gate = EndpointGate(POLICY)
    feed(gate, [(0.0, 1.0)], completeness=0.0, until=1.2)
    result = gate.flush(1.2)
    assert result is not None
    assert result.reason == Reason.FLUSH


def test_flush_does_not_double_fire():
    gate = EndpointGate(POLICY)
    feed(gate, [(0.0, 1.0)], completeness=1.0, until=2.0)
    assert gate.flush(2.0) is None


def test_turn_detector_is_not_consulted_while_speech_continues():
    """At 20ms frames, scoring every push would dominate the CPU budget."""
    detector = ScriptedTurnDetector(fixed=1.0)
    coord = Coordinator(
        DICTATION,
        engine=ScriptedEngine(
            script={1.6: Hypothesis(stable=words("hello there", each=0.4), consumed_to=1.6)}
        ),
        vad=ScriptedVAD(speech=((0.0, 3.0),)),
        turn=detector,
    )
    frame = b"\x00\x00" * 320
    for _ in range(100):  # 2.0s, all of it speech
        coord.push(frame)
    assert detector.calls == [], "the semantic gate ran while the speaker was still talking"


def test_endpoint_closes_the_turn_so_the_next_word_opens_a_new_one():
    engine = ScriptedEngine(
        script={
            1.2: Hypothesis(stable=words("done.", start=0.4, each=0.4), consumed_to=1.2),
            4.0: Hypothesis(
                stable=words("done.", start=0.4, each=0.4) + words("again.", start=3.0, each=0.4),
                consumed_to=4.0,
            ),
        }
    )
    coord = Coordinator(
        DICTATION,
        engine=engine,
        vad=ScriptedVAD(speech=((0.0, 1.2), (3.0, 3.6))),
        turn=ScriptedTurnDetector(),
    )
    events = drive(coord, 5.0)
    turns = [e.payload["turn"] for e in events if e.kind == "turn_start"]
    assert turns == [1, 2], f"expected two turns, got {turns}"
