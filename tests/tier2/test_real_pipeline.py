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


def test_every_registry_model_is_ungated_and_pinned():
    """No model may need an account, a token, or a licence acceptance.

    This one needs no weights, so it always runs.
    """
    for name, spec in REGISTRY.items():
        assert spec.url.startswith("https://github.com/"), (
            f"{name} is not a plain public download: {spec.url}"
        )
        assert "huggingface.co" not in spec.url, f"{name} may be gated"
        assert spec.sha256, f"{name} has no pinned checksum"
        assert spec.licence, f"{name} has no recorded licence"


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
