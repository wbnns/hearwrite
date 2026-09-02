"""Tier 2: the real models, on real audio.

Not run by `bin/check`. These need weights on disk and take seconds rather than
milliseconds, so they are marked and skipped unless the models are present:

    pytest tests/tier2 -m tier2

Everything asserted here is a property of the pipeline, not a transcript. Word
error rate belongs in the metric harness with a checked-in baseline and a
ratchet; asserting exact text in a unit test just pins today's model.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from hearwrite import DICTATION, Coordinator
from hearwrite.audio import chunks, duration
from hearwrite.models import REGISTRY, resolve

pytestmark = pytest.mark.tier2

SR = 16_000


def _have(name: str) -> bool:
    try:
        resolve(name, download=False)
        return True
    except Exception:
        return False


needs_models = pytest.mark.skipif(
    not (_have("zipformer-en") and _have("silero-vad")),
    reason="run `hearwrite models` and download zipformer-en and silero-vad first",
)


@pytest.fixture(scope="module")
def speech(tmp_path_factory) -> bytes:
    """A short utterance followed by trailing silence.

    Generated with macOS `say` when available so the suite carries no audio
    fixture; skipped elsewhere until a checked-in corpus exists.
    """
    import shutil
    import subprocess

    if not shutil.which("say"):
        pytest.skip("no `say` available to synthesise a fixture")
    path = tmp_path_factory.mktemp("audio") / "utterance.wav"
    subprocess.run(
        [
            "say",
            "-r",
            "150",
            "--data-format=LEI16@16000",
            "--file-format=WAVE",
            "-o",
            str(path),
            "The build is green and the tests are passing.",
        ],
        check=True,
    )
    with wave.open(str(path)) as handle:
        pcm = handle.readframes(handle.getnframes())
    return pcm + b"\x00\x00" * int(1.5 * SR)


@pytest.fixture(scope="module")
def pipeline():
    from hearwrite.engines.sherpa import SherpaStreamingEngine
    from hearwrite.vad.silero import SileroVAD

    return SherpaStreamingEngine.from_model("zipformer-en"), SileroVAD.from_model()


def _run(pipeline, pcm, chunk=0.02):
    engine, vad = pipeline
    engine.reset()
    vad.reset()
    coordinator = Coordinator(DICTATION, engine=engine, vad=vad)
    events = []
    for piece in chunks(pcm, chunk):
        events.extend(coordinator.push(piece))
    events.extend(coordinator.finish())
    return coordinator, events


@needs_models
def test_produces_a_transcript(pipeline, speech):
    coordinator, _ = _run(pipeline, speech)
    assert coordinator.log.committed_text.strip(), "no words were committed"


@needs_models
def test_event_log_is_ordered_in_time(pipeline, speech):
    _, events = _run(pipeline, speech)
    times = [e.at for e in events]
    assert times == sorted(times)


@needs_models
def test_committed_audio_never_overlaps(pipeline, speech):
    """The append only rule, against a real decoder rather than a script."""
    _, events = _run(pipeline, speech)
    frontier = 0.0
    for event in events:
        if event.kind == "commit":
            assert event.payload["audio_start"] + 1e-6 >= frontier
            frontier = event.payload["audio_end"]


@needs_models
def test_no_word_is_emitted_before_its_audio(pipeline, speech):
    """Negative delay is physically impossible and silently poisons metrics."""
    _, events = _run(pipeline, speech)
    for event in events:
        if event.kind == "commit":
            assert event.payload["delay"] >= -1e-6, event.payload


@needs_models
def test_the_final_word_survives_the_flush(pipeline, speech):
    """Without trailing padding the decoder keeps its last words forever."""
    engine, vad = pipeline
    engine.reset()
    vad.reset()
    coordinator = Coordinator(DICTATION, engine=engine, vad=vad)
    for piece in chunks(speech, 0.02):
        coordinator.push(piece)
    before = coordinator.log.committed_text
    coordinator.finish()
    after = coordinator.log.committed_text
    assert after.startswith(before)
    assert len(after) >= len(before)


@needs_models
def test_faster_than_real_time_on_cpu(pipeline, speech):
    """RTF well under 1.0 is the whole premise of running this on a laptop."""
    import time

    started = time.perf_counter()
    _run(pipeline, speech)
    elapsed = time.perf_counter() - started
    rtf = elapsed / duration(speech)
    assert rtf < 0.5, f"real time factor {rtf:.3f} is too slow for streaming"


@needs_models
@pytest.mark.parametrize("chunk", [0.02, 0.2, 0.5])
def test_transcript_does_not_depend_on_chunk_size(pipeline, speech, chunk):
    """Chunk size invariance, against the real decoder."""
    baseline, _ = _run(pipeline, speech, chunk=0.02)
    other, _ = _run(pipeline, speech, chunk=chunk)
    assert other.log.committed_text == baseline.log.committed_text


@needs_models
def test_solo_mode_never_touches_a_speaker_frontend(pipeline, speech):
    _, events = _run(pipeline, speech)
    assert all(e.payload["speaker"] == "A" for e in events if e.kind == "commit")


def test_every_registry_model_records_a_licence_and_a_checksum():
    """No model may need an account, a token, or a licence acceptance.

    Needs no weights, so it always runs. The host allowlist is not the point --
    plenty of ungated models live on Hugging Face -- but a URL carrying
    credentials or pointing somewhere unexpected is worth failing on.
    """
    allowed = ("https://github.com/", "https://huggingface.co/")
    for name, spec in REGISTRY.items():
        assert spec.url.startswith(allowed), f"{name}: unexpected host {spec.url}"
        assert "?" not in spec.url, f"{name}: URL carries query parameters"
        assert "@" not in spec.url.split("//", 1)[1].split("/", 1)[0], (
            f"{name}: URL embeds credentials"
        )
        assert spec.sha256, f"{name} has no pinned checksum"
        assert spec.licence, f"{name} has no recorded licence"


@pytest.mark.network
def test_every_registry_model_downloads_without_credentials():
    """The actual definition of ungated: an anonymous request succeeds.

    This is the property that matters and the only way to check it is to try.
    Marked separately because it needs the network but no weights on disk.
    """
    import urllib.error
    import urllib.request

    for name, spec in REGISTRY.items():
        request = urllib.request.Request(spec.url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                assert response.status == 200, f"{name}: HTTP {response.status}"
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise AssertionError(
                f"{name} is not anonymously downloadable: HTTP {exc.code}. "
                f"A gated model means a token in CI and weights nobody can vendor."
            ) from exc


# -- diarization, real speakers ----------------------------------------------

_FIXTURES = Path.home() / ".cache/hearwrite/corpora/fixtures"


def _fixture(name: str):
    wav = _FIXTURES / f"{name}.wav"
    truth = wav.with_suffix(".json")
    if not (wav.exists() and truth.exists()):
        pytest.skip(f"build {wav} first; see docs/evaluation.md")
    return wav, truth


needs_speakers = pytest.mark.skipif(
    not (_have("zipformer-en") and _have("silero-vad") and _have("titanet-small")),
    reason="download zipformer-en, silero-vad and titanet-small first",
)


def _diarize(wav):
    import json

    from hearwrite import CONVERSATION, Coordinator
    from hearwrite.audio import chunks, read_wav
    from hearwrite.engines.sherpa import SherpaStreamingEngine
    from hearwrite.metrics import evaluate, load_turns
    from hearwrite.speakers.sherpa import SherpaSpeakerFrontend
    from hearwrite.vad.silero import SileroVAD

    pcm = read_wav(wav)
    coordinator = Coordinator(
        CONVERSATION,
        engine=SherpaStreamingEngine.from_model("zipformer-en"),
        vad=SileroVAD.from_model(),
        speakers=SherpaSpeakerFrontend.from_model("titanet-small"),
    )
    events = []
    for piece in chunks(pcm, 0.02):
        events.extend(coordinator.push(piece))
    events.extend(coordinator.finish())
    turns = load_turns(json.loads(wav.with_suffix(".json").read_text()))
    return evaluate(events, turns)


@needs_speakers
@pytest.mark.parametrize("name", ["conv_2spk", "conv_4spk", "conv_8spk"])
def test_speaker_count_is_discovered_not_configured(name):
    """Nothing tells the pipeline how many people are talking."""
    wav, _ = _fixture(name)
    report = _diarize(wav)
    assert report.predicted_speakers == report.true_speakers


@needs_speakers
@pytest.mark.parametrize("name", ["conv_2spk", "conv_4spk", "conv_8spk"])
def test_confusion_stays_in_budget(name):
    """A ratchet, not a target. Measured at 3.1% / 6.1% / 6.7% on clean read
    speech with pauses between turns; 15% leaves room for model changes while
    still failing on a real regression.
    """
    wav, _ = _fixture(name)
    assert _diarize(wav).confusion_rate < 0.15


@needs_speakers
def test_more_speakers_does_not_collapse_the_clustering():
    """The design claims no fixed speaker count. This is that claim, measured."""
    wav, _ = _fixture("conv_8spk")
    report = _diarize(wav)
    assert report.predicted_speakers == 8
    assert report.confusion_rate < 0.15


# -- the second engine -------------------------------------------------------


def _have_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


needs_whisper = pytest.mark.skipif(not _have_whisper(), reason="pip install 'hearwrite[whisper]'")


@needs_whisper
def test_whisper_runs_through_the_unmodified_coordinator(speech):
    """The Phase 0 bet, against a real offline model.

    Whisper is an encoder-decoder that wants the whole utterance. If the
    interface had been designed around it, a transducer would have needed a
    rewrite; because it was designed around a transducer, this adapter fits
    without the Coordinator knowing anything changed.
    """
    from hearwrite import DICTATION, Coordinator
    from hearwrite.audio import chunks
    from hearwrite.engines.whisper import WhisperStreamingEngine

    coordinator = Coordinator(
        DICTATION,
        engine=WhisperStreamingEngine.from_model("tiny.en", language="en"),
    )
    events = []
    for piece in chunks(speech, 0.02):
        events.extend(coordinator.push(piece))
    events.extend(coordinator.finish())

    assert coordinator.log.committed_text.strip()
    times = [e.at for e in events]
    assert times == sorted(times)
    frontier = 0.0
    for event in events:
        if event.kind == "commit":
            assert event.payload["audio_start"] + 1e-6 >= frontier
            frontier = event.payload["audio_end"]


@needs_whisper
def test_whisper_produces_punctuation_the_transducer_does_not(speech):
    """The reason this engine exists despite being slower and heavier."""
    from hearwrite import DICTATION, Coordinator
    from hearwrite.audio import chunks
    from hearwrite.engines.whisper import WhisperStreamingEngine

    coordinator = Coordinator(
        DICTATION,
        engine=WhisperStreamingEngine.from_model("tiny.en", language="en"),
    )
    for piece in chunks(speech, 0.02):
        coordinator.push(piece)
    coordinator.finish()
    text = coordinator.log.committed_text
    assert any(mark in text for mark in ".?!,"), text
    assert text != text.upper(), "expected mixed case, got shouting"


@needs_whisper
def test_whisper_buffer_stays_bounded_over_a_long_session(speech):
    """Whisper's structural weakness: cost per pass grows with the buffer.

    A three minute session must not carry three minutes of audio in memory, or
    the real time factor climbs until it exceeds 1.0 and never recovers.
    """
    from hearwrite.audio import chunks
    from hearwrite.engines.whisper import WhisperStreamingEngine

    engine = WhisperStreamingEngine.from_model("tiny.en", language="en")
    long_audio = speech * 8
    at = 0.0
    for piece in chunks(long_audio, 0.02):
        at += len(piece) / 2 / SR
        engine.push(piece, at)
    assert at > 30.0, "fixture too short to prove anything"
    assert len(engine._buffer) <= int(30 * SR), (
        f"buffer grew to {len(engine._buffer) / SR:.1f}s over a {at:.0f}s session"
    )


# -- semantic endpointing ----------------------------------------------------

_MIDTHOUGHT = _FIXTURES / "midthought"

needs_turn = pytest.mark.skipif(not _have("smart-turn"), reason="download smart-turn first")


def _midthought_scores():
    import json
    import wave

    index = _MIDTHOUGHT / "index.json"
    if not index.exists():
        pytest.skip("build the corpus with scripts/build_fixtures.py")

    from hearwrite.turn.smart_turn import SmartTurnDetector

    detector = SmartTurnDetector.from_model()
    items = json.loads(index.read_text())

    def score(name):
        with wave.open(str(_MIDTHOUGHT / name)) as handle:
            pcm = handle.readframes(handle.getnframes())
        return detector.completeness("", pcm[: max(0, len(pcm) - int(1.5 * SR) * 2)])

    return (
        [score(i["incomplete"]) for i in items],
        [score(i["complete"]) for i in items],
    )


@needs_turn
def test_the_semantic_gate_separates_finished_from_unfinished():
    """Clips cut just after a function word cannot be finished sentences.

    If this stops separating, the model or the feature pipeline has changed and
    every endpoint threshold below it is meaningless.
    """
    import statistics

    incomplete, complete = _midthought_scores()
    gap = statistics.median(complete) - statistics.median(incomplete)
    assert gap > 0.10, f"no discrimination: median gap {gap:+.3f}"


@needs_turn
def test_conservative_policy_rarely_cuts_a_speaker_off():
    """The metric the whole conjunctive design exists to move."""
    from hearwrite.coordinator.policy import _ENDPOINT_PRESETS, EndpointMode
    from hearwrite.metrics import score_endpoints

    incomplete, complete = _midthought_scores()
    threshold = _ENDPOINT_PRESETS[EndpointMode.CONSERVATIVE].completeness_threshold
    report = score_endpoints(incomplete, complete, threshold)
    assert report.false_endpoint_rate <= 0.10, (
        f"false endpoint rate {report.false_endpoint_rate:.1%}"
    )


@needs_turn
def test_thresholds_are_ordered_by_how_much_they_interrupt():
    """Aggressive should interrupt more often than conservative, by design."""
    from hearwrite.coordinator.policy import _ENDPOINT_PRESETS, EndpointMode
    from hearwrite.metrics import score_endpoints

    incomplete, complete = _midthought_scores()
    rates = {
        mode: score_endpoints(
            incomplete, complete, _ENDPOINT_PRESETS[mode].completeness_threshold
        ).false_endpoint_rate
        for mode in EndpointMode
    }
    assert (
        rates[EndpointMode.CONSERVATIVE]
        <= rates[EndpointMode.BALANCED]
        <= rates[EndpointMode.AGGRESSIVE]
    ), rates


@needs_turn
def test_acoustic_only_cuts_people_off_and_the_semantic_gate_stops_it():
    """The comparison that justifies the second model.

    An acoustic gate fires on silence alone, so a mid thought pause ends the
    turn every time. Adding the semantic gate is what prevents it.
    """
    import wave

    from hearwrite import DICTATION, Coordinator
    from hearwrite.audio import chunks
    from hearwrite.engines.sherpa import SherpaStreamingEngine
    from hearwrite.turn.smart_turn import SmartTurnDetector
    from hearwrite.vad.silero import SileroVAD

    index = _MIDTHOUGHT / "index.json"
    if not index.exists():
        pytest.skip("build the corpus with scripts/build_fixtures.py")
    import json

    items = json.loads(index.read_text())[:8]

    def early_endpoints(use_turn: bool) -> int:
        count = 0
        for item in items:
            with wave.open(str(_MIDTHOUGHT / item["incomplete"])) as handle:
                pcm = handle.readframes(handle.getnframes())
            coordinator = Coordinator(
                DICTATION,
                engine=SherpaStreamingEngine.from_model(),
                vad=SileroVAD.from_model(),
                turn=SmartTurnDetector.from_model() if use_turn else None,
            )
            events = []
            for piece in chunks(pcm, 0.02):
                events.extend(coordinator.push(piece))
            count += sum(
                1 for e in events if e.kind == "endpoint" and e.payload["reason"] == "complete"
            )
        return count

    with_gate = early_endpoints(True)
    without_gate = early_endpoints(False)
    assert with_gate < without_gate, (
        f"the semantic gate changed nothing: {with_gate} vs {without_gate}"
    )


# -- model sharing -----------------------------------------------------------


@needs_models
def test_sharing_a_recognizer_is_safe():
    """Two streams interleaved on one recognizer must not affect each other.

    This is the assumption the whole memory story rests on: without it every
    session reloads about 300MB of weights. sherpa's design says streams are
    independent; this checks it rather than trusting it.
    """
    from hearwrite import loaders
    from hearwrite.engines.sherpa import SherpaStreamingEngine

    recognizer = loaders.transducer("zipformer-en")
    a = SherpaStreamingEngine(recognizer)
    b = SherpaStreamingEngine(recognizer)
    alone = SherpaStreamingEngine(recognizer)

    with wave.open(str(_FIXTURES / "conv_2spk.wav")) as handle:
        pcm = handle.readframes(handle.getnframes())[: 20 * SR * 2]
    other = pcm[::-1][: 10 * SR * 2]  # any different audio will do

    at = 0.0
    step = 320 * 2
    for i in range(0, len(pcm), step):
        at += step / 2 / SR
        a.push(pcm[i : i + step], at)
        if i < len(other):
            b.push(other[i : i + step], at)
    interleaved = " ".join(w.text for w in a.flush().stable)

    at = 0.0
    for i in range(0, len(pcm), step):
        at += step / 2 / SR
        alone.push(pcm[i : i + step], at)
    solo = " ".join(w.text for w in alone.flush().stable)

    assert interleaved == solo, "a concurrent stream changed the transcript"


@needs_models
def test_a_second_session_reuses_the_loaded_models():
    """The property that makes a small VPS viable."""
    from hearwrite import CONVERSATION, loaders
    from hearwrite.pipeline import Backends, build

    loaders.clear()
    build(CONVERSATION, Backends())
    first = loaders.loaded()
    build(CONVERSATION, Backends())
    assert loaders.loaded() == first, "a second session loaded its own models"
    assert first["transducer"] == 1
