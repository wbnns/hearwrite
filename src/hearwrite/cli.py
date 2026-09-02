"""The `hearwrite` command.

Exit codes: 0 on success, 1 on a runtime failure, 2 on a usage problem.

Phase 0 ships `demo` and `policies`, which need no models at all. `transcribe`
and `serve` exist but report clearly which extra to install rather than failing
with an ImportError traceback, because "pip install hearwrite && hearwrite
transcribe" is the first thing anyone tries.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from . import __version__
from .coordinator import PRESETS, Coordinator, preset
from .engines.base import Hypothesis
from .engines.fake import ScriptedEngine, words
from .events import Event
from .protocol import encode
from .speakers.base import Segment
from .speakers.fake import ScriptedFrontend, embedding
from .turn.fake import ScriptedTurnDetector
from .vad.fake import ScriptedVAD

_MISSING_BACKEND = (
    "the ONNX backend is not installed.\n"
    "  pip install 'hearwrite[onnx]'     # the default: CPU, no torch, nothing gated"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hearwrite",
        description="Real-time transcription, speaker labels and endpointing.",
    )
    parser.add_argument("--version", action="version", version=f"hearwrite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a scripted session; needs no models")
    demo.add_argument("--policy", default="conversation", choices=sorted(PRESETS))
    demo.add_argument("--chunk", type=float, default=0.02, help="chunk size in seconds")
    demo.add_argument("--json", action="store_true", help="emit the raw event log")

    sub.add_parser("policies", help="show the built-in policy presets")

    transcribe = sub.add_parser("transcribe", help="transcribe a 16kHz mono WAV file")
    transcribe.add_argument("path")
    transcribe.add_argument("--policy", default="dictation", choices=sorted(PRESETS))
    transcribe.add_argument("--model", default="zipformer-en")
    transcribe.add_argument("--chunk", type=float, default=0.02)
    transcribe.add_argument("--threads", type=int, default=2)
    transcribe.add_argument("--json", action="store_true", help="emit the raw event log")
    transcribe.add_argument("--no-vad", action="store_true", help="skip the acoustic gate")

    sub.add_parser("models", help="list known models and their licences")

    serve = sub.add_parser("serve", help="run the WebSocket service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--policy", default="dictation", choices=sorted(PRESETS))
    serve.add_argument("--model", default="zipformer-en")
    serve.add_argument(
        "--max-sessions",
        type=int,
        default=4,
        help="admission limit; never oversubscribe, latency is what users notice",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        return _demo(args)
    if args.command == "policies":
        return _policies()
    if args.command == "models":
        return _models()
    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "serve":
        return _serve(args)
    return 2


def _serve(args) -> int:
    import asyncio

    try:
        from .server.app import serve
    except ImportError as exc:
        print(f"hearwrite serve: {exc}", file=sys.stderr)
        return 1
    try:
        asyncio.run(
            serve(
                host=args.host,
                port=args.port,
                policy=preset(args.policy),
                model=args.model,
                max_sessions=args.max_sessions,
            )
        )
    except KeyboardInterrupt:
        return 0
    except ImportError as exc:
        print(f"hearwrite serve: {exc}", file=sys.stderr)
        return 1
    return 0


def _models() -> int:
    from .models import REGISTRY, cache_root, resolve

    print(f"cache: {cache_root()}")
    print()
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        try:
            resolve(name, download=False)
            state = "downloaded"
        except Exception:
            state = "not downloaded"
        print(f"{name}")
        print(f"  {spec.summary}")
        print(
            f"  {spec.licence} | {spec.languages} | ~{spec.approx_mb}MB | "
            f"{state} | sha256 {'pinned' if spec.sha256 else 'UNPINNED'}"
        )
    return 0


def _transcribe(args) -> int:
    """Run a real file through the real pipeline."""
    from . import audio
    from .models import ModelError, UnknownModel

    try:
        pcm = audio.read_wav(args.path)
    except audio.AudioError as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 2

    from .engines.sherpa import SherpaStreamingEngine
    from .vad.silero import SileroVAD

    policy = preset(args.policy)
    try:
        engine = SherpaStreamingEngine.from_model(args.model, num_threads=args.threads)
        vad = None if args.no_vad else SileroVAD.from_model()
    except ImportError:
        # sherpa-onnx is imported lazily inside from_model, so a missing backend
        # surfaces here rather than at module import. This is the first command
        # anyone runs; it must not answer with a traceback.
        print(f"hearwrite transcribe: {_MISSING_BACKEND}", file=sys.stderr)
        return 1
    except UnknownModel as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 1

    coordinator = Coordinator(policy, engine=engine, vad=vad)

    started = time.perf_counter()
    events: list[Event] = []
    for chunk in audio.chunks(pcm, args.chunk):
        events.extend(coordinator.push(chunk))
    events.extend(coordinator.finish())
    elapsed = time.perf_counter() - started
    seconds = audio.duration(pcm)

    if args.json:
        for event in events:
            print(encode(event))
        return 0

    for event in events:
        print(_render(event))

    delays = [e.payload["delay"] for e in events if e.kind == "commit"]
    print()
    print(f"transcript: {coordinator.log.committed_text}")
    if delays:
        ordered = sorted(delays)
        p50 = ordered[len(ordered) // 2]
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        print(
            f"delay:      p50 {p50:.2f}s  p90 {p90:.2f}s  max {ordered[-1]:.2f}s "
            f"({len(delays)} words)"
        )
    print(f"rtf:        {elapsed / seconds:.3f}  ({seconds:.1f}s audio in {elapsed:.2f}s)")
    return 0


def _policies() -> int:
    for name in sorted(PRESETS):
        p = PRESETS[name]
        print(f"{name}:")
        print(f"  speakers  {p.speakers.mode}")
        print(
            f"  endpoint  silence>={p.endpoint.silence_seconds}s "
            f"completeness>={p.endpoint.completeness_threshold} "
            f"timeout={p.endpoint.max_silence_seconds}s"
        )
    return 0


def _demo(args) -> int:
    """A canned two-speaker exchange, run through the real Coordinator.

    Everything downstream of the fakes is production code, so this exercises the
    commit policy, the clustering, the turn logic and the endpoint gate exactly
    as a real session would.
    """
    policy = preset(args.policy)

    voice_a = embedding(1.0, 0.0, 0.0)
    voice_b = embedding(0.0, 1.0, 0.0)
    engine = ScriptedEngine(
        script={
            2.1: Hypothesis(stable=words("is the build green", each=0.5), consumed_to=2.1),
            4.6: Hypothesis(
                stable=words("is the build green", each=0.5)
                + words("it is green.", start=2.6, each=0.5),
                consumed_to=4.6,
            ),
        }
    )
    coordinator = Coordinator(
        policy,
        engine=engine,
        vad=ScriptedVAD(speech=((0.0, 2.1), (2.6, 4.2))),
        speakers=ScriptedFrontend(
            segments=(Segment(0.0, 2.1, voice_a), Segment(2.6, 4.2, voice_b))
        ),
        turn=ScriptedTurnDetector(),
    )

    frame = b"\x00\x00" * int(args.chunk * policy.sample_rate)
    events: list[Event] = []
    for _ in range(int(6.0 / args.chunk)):
        events.extend(coordinator.push(frame))
    events.extend(coordinator.finish())

    if args.json:
        for event in events:
            print(encode(event))
        return 0

    for event in events:
        print(_render(event))
    print()
    print(f"transcript: {coordinator.log.committed_text}")
    print(f"speakers:   {coordinator.speaker_count}")
    return 0


def _render(event) -> str:
    p = event.payload
    head = f"{event.seq:>3}  {event.at:6.2f}s  {event.kind!s:<13}"
    if event.kind == "commit":
        who = p["speaker"] or "-"
        return f"{head} [{who}] {p['text']!r}  delay={p['delay']:+.2f}s"
    if event.kind == "partial":
        return f"{head} {p['text']!r}"
    if event.kind == "turn_start":
        return f"{head} turn {p['turn']}, speaker {p['speaker'] or '-'}"
    if event.kind == "speaker":
        return f"{head} seq {p['seq']} -> {p['speaker']}"
    if event.kind == "endpoint":
        return f"{head} {p['reason']}"
    if event.kind == "degraded":
        return f"{head} degraded={p['degraded']} lag={p['lag']:.2f}s"
    return f"{head} {dict(p)}"


if __name__ == "__main__":
    raise SystemExit(main())
