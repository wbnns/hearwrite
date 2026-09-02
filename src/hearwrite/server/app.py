"""The WebSocket service: binary PCM up, JSON events down.

Deliberately not part of the application server. Audio frames should never pass
through a request/response app: the client connects here directly, which means
this process can be restarted without touching the app and the app is not asked
to do a job it is bad at.

Protocol, in full:

    server -> client   one `hello` frame (protocol version, sample rate, policy)
    client -> server   binary frames of 16kHz mono signed 16 bit PCM
    server -> client   one JSON line per event, in order, forever append only
    client -> server   the text "finish" to flush and close

Anything the client sends that is neither binary audio nor `finish` is ignored
rather than fatal, so a chatty client cannot kill its own session.
"""

from __future__ import annotations

import asyncio
import os
import sys

from ..coordinator import Policy, preset
from ..pipeline import Backends, build
from ..protocol import encode, hello_frame
from .session import Admission, Rejected, Session


def default_max_sessions() -> int:
    """An admission limit derived from the machine, not from a guess.

    Measured on a 10 core Apple M4 with all four models: one stream runs at a
    real time factor of 0.047, eight concurrent streams at 0.133 each. So the
    CPU has a great deal of headroom and memory is the binding constraint --
    about 340MB for the shared models plus a few MB per session.

    Defaulting to the core count is therefore conservative by a wide margin,
    which is the right side to be on: latency degradation under contention is
    the failure users notice first, so refusing a connection is kinder than
    accepting it and being slow for everybody. Raise it deliberately, after
    measuring on the machine you actually deploy to.
    """
    return max(2, min(16, os.cpu_count() or 2))


async def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    policy: Policy | None = None,
    backends: Backends | None = None,
    max_sessions: int | None = None,
) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the server extra is not installed.\n  pip install 'hearwrite[server]'"
        ) from exc

    policy = policy or preset("dictation")
    backends = backends or Backends()
    limit = max_sessions if max_sessions is not None else default_max_sessions()
    admission = Admission(limit)

    async def handler(websocket) -> None:
        try:
            admission.acquire()
        except Rejected as exc:
            await websocket.close(code=1013, reason=str(exc))
            return
        try:
            await _run_session(websocket, policy, backends)
        finally:
            admission.release()

    async with websockets.serve(handler, host, port, max_size=None):
        speakers = "solo" if policy.is_solo else "auto"
        print(
            f"hearwrite: listening on ws://{host}:{port} "
            f"(engine {backends.engine}, speakers {speakers}, "
            f"semantic gate {'on' if backends.turn else 'off'}, "
            f"max {limit} sessions)",
            file=sys.stderr,
        )
        await asyncio.Future()


async def _run_session(websocket, policy, backends) -> None:
    loop = asyncio.get_running_loop()
    # Building the pipeline touches disk and can take a second, so keep it off
    # the event loop: one connecting client must not stall the others.
    #
    # It goes through the SAME builder the CLI uses. When it did not, the server
    # silently ran without diarization or a semantic gate for two whole phases,
    # because nothing fails when a pipeline is merely worse.
    components = await loop.run_in_executor(None, build, policy, backends)
    session = Session(policy, **components.as_kwargs())

    await websocket.send(
        hello_frame(sample_rate=policy.sample_rate, policy=str(policy.speakers.mode))
    )

    async for message in websocket:
        if isinstance(message, bytes):
            for event in session.push(message):
                await websocket.send(encode(event))
        elif message == "finish":
            break
        # Anything else is ignored: a chatty client should not kill its session.

    for event in session.finish():
        await websocket.send(encode(event))
