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


# -- the server must not drift from the CLI ----------------------------------


def test_the_server_builds_its_pipeline_through_the_shared_builder():
    """The server ran without diarization or a semantic gate for two phases.

    Nothing failed. It was written before either existed and was never updated,
    so `serve --policy conversation` quietly delivered a worse pipeline than
    `transcribe` did on the same audio. Both entrances now call one builder, and
    this is what stops them separating again.
    """
    import inspect

    from hearwrite.server import app

    source = inspect.getsource(app)
    assert "from ..pipeline import" in source, "the server builds its own pipeline"
    assert "components.as_kwargs()" in source, (
        "the server does not hand the built components to its Session"
    )

    # And it must not construct components directly any more.
    for constructed in ("SherpaStreamingEngine", "SileroVAD", "SmartTurnDetector"):
        assert constructed not in source, f"the server still constructs {constructed} itself"


def test_the_builder_skips_the_speaker_frontend_in_solo_mode():
    """Solo does not merely ignore the frontend, it never builds it."""
    import inspect

    from hearwrite import pipeline

    source = inspect.getsource(pipeline.build)
    assert "policy.is_solo" in source


def test_backends_defaults_match_the_cli_defaults():
    """A default that differs between entrances is the same drift in miniature."""
    from hearwrite.cli import build_parser
    from hearwrite.pipeline import Backends

    defaults = Backends()
    for command in ("transcribe", "serve"):
        argv = [command] + (["x"] if command == "transcribe" else [])
        args = build_parser().parse_args(argv)
        assert args.engine == defaults.engine
        assert args.speaker_model == defaults.speaker_model
        assert args.threads == defaults.threads
        assert args.no_turn is False
        # The one that actually bit: `serve --model` carried a literal default
        # that silently overrode the engine's, so the service ran a different
        # recogniser from `transcribe` on the same machine for days.
        assert args.model is None, (
            f"`{command} --model` hardcodes {args.model!r} instead of deferring "
            "to the engine default"
        )
        assert args.provider == defaults.provider
        assert args.normalise == defaults.normalise


def test_a_session_accepts_exactly_what_the_builder_produces():
    """The contract between the pipeline and the session, checked directly.

    Renaming a component from `punctuator` to `polish` updated the builder and
    not the Session, so every WebSocket connection died with a TypeError while
    the page still served a clean 200. Nothing caught it: the existing tests
    construct a Session with explicit arguments, which is precisely the path
    that cannot drift.

    This is the fourth time a component has been wired in two places and only
    one updated, so the test compares the two signatures rather than any
    particular field.
    """
    import inspect

    from hearwrite.pipeline import Components
    from hearwrite.server.session import Session

    produced = set(Components(engine=object()).as_kwargs())
    accepted = set(inspect.signature(Session.__init__).parameters) - {"self", "policy"}
    missing = produced - accepted
    assert not missing, f"Session cannot accept what build() returns: {missing}"


def test_the_coordinator_accepts_it_too():
    """The same contract one layer down."""
    import inspect

    from hearwrite import Coordinator
    from hearwrite.pipeline import Components

    produced = set(Components(engine=object()).as_kwargs())
    accepted = set(inspect.signature(Coordinator.__init__).parameters) - {"self", "policy"}
    assert not produced - accepted
