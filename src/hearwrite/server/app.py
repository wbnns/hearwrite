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
import sys

from ..coordinator import Policy, preset
from ..protocol import encode, hello_frame
from .session import Admission, Rejected, Session

DEFAULT_MAX_SESSIONS = 4


async def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    policy: Policy | None = None,
    model: str = "zipformer-en",
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    build_engine=None,
    build_vad=None,
) -> None:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the server extra is not installed.\n  pip install 'hearwrite[server]'"
        ) from exc

    policy = policy or preset("dictation")
    admission = Admission(max_sessions)

    if build_engine is None:
        from ..engines.sherpa import SherpaStreamingEngine

        def build_engine():
            return SherpaStreamingEngine.from_model(model)

    if build_vad is None:
        from ..vad.silero import SileroVAD

        def build_vad():
            return SileroVAD.from_model()

    async def handler(websocket) -> None:
        try:
            admission.acquire()
        except Rejected as exc:
            await websocket.close(code=1013, reason=str(exc))
            return
        try:
            await _run_session(websocket, policy, build_engine, build_vad)
        finally:
            admission.release()

    async with websockets.serve(handler, host, port, max_size=None):
        print(
            f"hearwrite: listening on ws://{host}:{port} "
            f"(policy {policy.speakers.mode}/{policy.endpoint.silence_seconds}s, "
            f"max {max_sessions} sessions)",
            file=sys.stderr,
        )
        await asyncio.Future()


async def _run_session(websocket, policy, build_engine, build_vad) -> None:
    loop = asyncio.get_running_loop()
    # Model construction touches disk and can take a second; keep it off the
    # event loop so one connecting client does not stall the others.
    engine = await loop.run_in_executor(None, build_engine)
    vad = await loop.run_in_executor(None, build_vad)
    session = Session(policy, engine=engine, vad=vad)

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
