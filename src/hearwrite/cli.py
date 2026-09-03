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
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    """The whole command line surface, separate from running it.

    Split out so tests can check that every flag the code reads is a flag the
    parser defines, without starting a server or loading a model. That gap once
    shipped a `--no-turn` the parser had never heard of.
    """
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
    transcribe.add_argument(
        "--engine",
        default="sherpa",
        choices=("sherpa", "whisper"),
        help="sherpa is the streaming transducer; whisper is offline with LocalAgreement",
    )
    transcribe.add_argument(
        "--model",
        default=None,
        help="model name; defaults to nemotron-3.5-160ms for sherpa, base for whisper",
    )
    transcribe.add_argument("--speaker-model", default="titanet-small")
    transcribe.add_argument("--language", default=None, help="whisper only; auto if unset")
    transcribe.add_argument("--chunk", type=float, default=0.02)
    transcribe.add_argument("--threads", type=int, default=2)
    transcribe.add_argument("--json", action="store_true", help="emit the raw event log")
    transcribe.add_argument("--no-vad", action="store_true", help="skip the acoustic gate")
    transcribe.add_argument("--no-turn", action="store_true", help="skip the semantic gate")
    transcribe.add_argument(
        "--punctuate",
        action="store_true",
        default=None,
        help="re-render finished sentences with a second model; on by default "
        "behind a recogniser that does not punctuate, off behind one that does",
    )
    transcribe.add_argument(
        "--no-punctuate", dest="punctuate", action="store_false", help=argparse.SUPPRESS
    )

    models = sub.add_parser("models", help="list known models and their licences")
    models.add_argument(
        "--prune",
        action="store_true",
        help="delete downloaded model files no loader will ever open",
    )

    endpoints = sub.add_parser("endpoints", help="score endpointing against a mid thought corpus")
    endpoints.add_argument("path", help="directory containing index.json")
    endpoints.add_argument("--turn-model", default="smart-turn")

    bench = sub.add_parser("bench", help="score diarization against a labelled fixture")
    bench.add_argument("path", help="WAV file with a sibling .json of ground truth")
    bench.add_argument("--policy", default="conversation", choices=sorted(PRESETS))
    bench.add_argument("--speaker-model", default="titanet-small")
    bench.add_argument("--engine", default="sherpa", choices=("sherpa", "whisper"))
    bench.add_argument("--model", default=None)
    bench.add_argument("--language", default=None)
    bench.add_argument("--threads", type=int, default=2)

    serve = sub.add_parser("serve", help="run the WebSocket service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--policy",
        default="conversation",
        choices=sorted(PRESETS),
        help="conversation labels speakers; dictation is solo and cheaper",
    )
    serve.add_argument("--model", default="zipformer-en")
    serve.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="admission limit; never oversubscribe, latency is what users notice",
    )
    serve.add_argument("--engine", default="sherpa", choices=("sherpa", "whisper"))
    serve.add_argument("--speaker-model", default="titanet-small")
    serve.add_argument("--language", default=None)
    serve.add_argument("--threads", type=int, default=2)
    serve.add_argument("--no-vad", action="store_true")
    serve.add_argument("--no-turn", action="store_true")
    serve.add_argument(
        "--punctuate",
        action="store_true",
        default=None,
        help="re-render finished sentences with a second model; on by default "
        "behind a recogniser that does not punctuate, off behind one that does",
    )
    serve.add_argument(
        "--no-punctuate", dest="punctuate", action="store_false", help=argparse.SUPPRESS
    )
    serve.add_argument(
        "--open", action="store_true", help="open the browser UI once the server is up"
    )
    serve.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="save each session's audio as a WAV, for working out whether a bad "
        "transcript came from a bad recording",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        return _demo(args)
    if args.command == "policies":
        return _policies()
    if args.command == "models":
        return _models(prune=args.prune)
    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "endpoints":
        return _endpoints(args)
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
    if getattr(args, "open", False):
        import threading
        import webbrowser

        url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
        # Give the models a moment to load, or the browser lands on a refused
        # connection and the user reloads by hand for no reason.
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    try:
        asyncio.run(
            serve(
                host=args.host,
                port=args.port,
                policy=preset(args.policy),
                backends=_backends(args),
                max_sessions=args.max_sessions,
                record=Path(args.record) if args.record else None,
            )
        )
    except KeyboardInterrupt:
        return 0
    except ImportError as exc:
        print(f"hearwrite serve: {exc}", file=sys.stderr)
        return 1
    return 0


def _models(*, prune: bool = False) -> int:
    from .models import REGISTRY, cache_root, resolve, unused_files

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

    if not prune:
        return 0

    print()
    freed = 0
    for name in sorted(REGISTRY):
        try:
            directory = resolve(name, download=False)
        except Exception:
            continue
        for path in unused_files(REGISTRY[name], directory):
            size = path.stat().st_size
            path.unlink()
            freed += size
            print(f"removed {path.name} ({size / 1e6:.0f}MB)")
    if freed:
        print(f"\nfreed {freed / 1e6:.0f}MB. Re-download with `hearwrite models`.")
    else:
        print("nothing to prune.")
    return 0


def _endpoints(args) -> int:
    """Measure the semantic gate on clips cut mid thought.

    The corpus pairs each full utterance with the same audio cut just after a
    function word, which is a point no English sentence ends at. So the
    incomplete labels are trustworthy, and the false endpoint rate they produce
    is the number worth quoting.
    """
    import json
    import wave
    from pathlib import Path

    from .coordinator.policy import _ENDPOINT_PRESETS
    from .metrics import score_endpoints
    from .turn.smart_turn import SmartTurnDetector

    root = Path(args.path)
    index = root / "index.json"
    if not index.exists():
        print(
            f"hearwrite endpoints: no corpus at {index}\nBuild one with scripts/build_fixtures.py",
            file=sys.stderr,
        )
        return 2

    detector = SmartTurnDetector.from_model(args.turn_model)
    items = json.loads(index.read_text())

    def score(name: str) -> float:
        with wave.open(str(root / name)) as handle:
            pcm = handle.readframes(handle.getnframes())
        # Drop the trailing silence appended for the acoustic gate: the model
        # wants the speech leading up to the pause, not the pause itself.
        trailing = int(1.5 * 16_000) * 2
        return detector.completeness("", pcm[: max(0, len(pcm) - trailing)])

    incomplete = [score(i["incomplete"]) for i in items]
    complete = [score(i["complete"]) for i in items]

    print(f"corpus:  {len(items)} pairs from {root}")
    print()
    print(f"  {'policy':<14} {'thr':>5} {'false endpoint':>15} {'left to timeout':>17}")
    for mode, policy in _ENDPOINT_PRESETS.items():
        report = score_endpoints(incomplete, complete, policy.completeness_threshold)
        print(
            f"  {mode!s:<14} {policy.completeness_threshold:>5.2f} "
            f"{report.false_endpoint_rate:>14.1%} {report.missed_endpoint_rate:>16.1%}"
        )
    print()
    print("  false endpoint = cut the speaker off mid thought. The one that matters.")
    print("  left to timeout = finished, but the semantic gate did not say so; the")
    print("  acoustic timeout still closes the turn, so this is latency not loss.")
    return 0


def _bench(args) -> int:
    """Measure diarization on a fixture with known speakers and turns."""
    import json
    from pathlib import Path

    from . import audio
    from .metrics import evaluate, load_turns

    truth_path = Path(args.path).with_suffix(".json")
    if not truth_path.exists():
        print(
            f"hearwrite bench: no ground truth at {truth_path}\n"
            f"Expected a JSON file with a 'turns' list of "
            f"{{speaker, start, end}} beside the audio.",
            file=sys.stderr,
        )
        return 2
    try:
        pcm = audio.read_wav(args.path)
    except audio.AudioError as exc:
        print(f"hearwrite bench: {exc}", file=sys.stderr)
        return 2

    turns = load_turns(json.loads(truth_path.read_text()))
    policy = preset(args.policy)
    _, events, elapsed = _run_pipeline(args, policy, pcm)
    report = evaluate(events, turns)

    print(f"file:        {Path(args.path).name}")
    print(f"policy:      {args.policy}")
    print(f"speakers:    {report.predicted_speakers} found, {report.true_speakers} true")
    print(f"words:       {report.words} scorable")
    detail = f"{report.confused} of {report.labelled} labelled"
    print(f"confusion:   {report.confusion_rate:.1%}  ({detail})")
    print(f"null rate:   {report.null_rate:.1%}  ({report.unlabelled} abstained)")
    if report.p50_turn_latency is not None:
        print(
            f"turn label:  p50 {report.p50_turn_latency:.2f}s  p90 {report.p90_turn_latency:.2f}s"
        )
    print(f"rtf:         {elapsed / audio.duration(pcm):.3f}")
    return 0


def _backends(args):
    """Turn parsed arguments into a backend selection."""
    from .pipeline import Backends

    return Backends(
        engine=getattr(args, "engine", "sherpa"),
        model=getattr(args, "model", None),
        speaker_model=getattr(args, "speaker_model", "titanet-small"),
        language=getattr(args, "language", None),
        threads=getattr(args, "threads", 2),
        vad=not getattr(args, "no_vad", False),
        turn=not getattr(args, "no_turn", False),
        punctuate=getattr(args, "punctuate", None),
    )


def _run_pipeline(args, policy, pcm):
    """Build the real pipeline and run PCM through it. Shared by two commands."""
    from . import audio
    from .pipeline import build

    components = build(policy, _backends(args))
    coordinator = Coordinator(policy, **components.as_kwargs())

    started = time.perf_counter()
    events: list[Event] = []
    for chunk in audio.chunks(pcm, getattr(args, "chunk", 0.02)):
        events.extend(coordinator.push(chunk))
    events.extend(coordinator.finish())
    return coordinator, events, time.perf_counter() - started


def _transcribe(args) -> int:
    """Run a real file through the real pipeline."""
    from . import audio
    from .models import ModelError, UnknownModel

    try:
        pcm = audio.read_wav(args.path)
    except audio.AudioError as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 2

    policy = preset(args.policy)
    try:
        coordinator, events, elapsed = _run_pipeline(args, policy, pcm)
    except ImportError:
        print(f"hearwrite transcribe: {_MISSING_BACKEND}", file=sys.stderr)
        return 1
    except UnknownModel as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"hearwrite transcribe: {exc}", file=sys.stderr)
        return 1
    seconds = audio.duration(pcm)

    if args.json:
        for event in events:
            print(encode(event))
        return 0

    for event in events:
        print(_render(event))

    commits = [e for e in events if e.kind == "commit"]
    delays = [e.payload["delay"] for e in commits]
    print()
    from .transcript import display_text

    text = display_text(events)
    print(f"transcript: {text}")
    if not policy.is_solo:
        unlabelled = sum(1 for e in commits if e.payload["speaker"] is None)
        print(
            f"speakers:   {coordinator.speaker_count} found, "
            f"{unlabelled}/{len(commits)} words unlabelled"
        )
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
    from .transcript import display_text

    text = display_text(events)
    print(f"transcript: {text}")
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
        # A speaker event names either one word, by seq, or a whole turn that
        # only became identifiable after it started. Both are real; assuming
        # the first is a KeyError waiting for the second.
        if "seq" in p:
            return f"{head} seq {p['seq']} -> {p['speaker']}"
        return f"{head} turn {p['turn']} -> {p['speaker']}"
    if event.kind == "endpoint":
        return f"{head} {p['reason']}"
    if event.kind == "polished":
        return f"{head} seq {p['from_seq']}..{p['to_seq']}  {p['text']!r}"
    if event.kind == "degraded":
        return f"{head} degraded={p['degraded']} lag={p['lag']:.2f}s"
    return f"{head} {dict(p)}"


if __name__ == "__main__":
    raise SystemExit(main())
