"""Whisper style log mel features.

Reimplementing a feature extractor is a good way to be silently wrong: the
output looks plausible, the model still returns probabilities, and accuracy just
quietly drops. So the filterbank is checked against an independent
implementation where one is available, and the pipeline's own invariants are
checked always.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from hearwrite.features import (  # noqa: E402
    CHUNK_FRAMES,
    CHUNK_SAMPLES,
    N_MELS,
    log_mel,
    mel_filters,
    normalise,
    window_features,
)


def test_filterbank_shape_and_positivity():
    filters = mel_filters()
    assert filters.shape == (N_MELS, 201)
    assert (filters >= 0).all()
    assert (filters.sum(axis=1) > 0).all(), "a mel filter is entirely empty"


def test_filterbank_matches_an_independent_implementation():
    """faster-whisper computes the same bank from the same definition."""
    fw = pytest.importorskip("faster_whisper.feature_extractor")
    theirs = fw.FeatureExtractor(
        feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=8, n_fft=400
    ).get_mel_filters(16000, 400, 80)
    assert np.allclose(mel_filters(), theirs, atol=1e-7)


def test_window_features_have_the_shape_the_model_wants():
    audio = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    assert window_features(audio).shape == (1, N_MELS, CHUNK_FRAMES)


def test_short_audio_is_padded_and_long_audio_keeps_the_END():
    """The question is whether the speaker has finished, and the evidence for
    that is in what they just said, not what they said a minute ago."""
    short = window_features(np.ones(1000, dtype=np.float32))
    assert short.shape == (1, N_MELS, CHUNK_FRAMES)

    ramp = np.linspace(0, 1, CHUNK_SAMPLES * 3).astype(np.float32)
    kept = window_features(ramp)
    tail = window_features(ramp[-CHUNK_SAMPLES:])
    assert np.allclose(kept, tail)


def test_normalisation_happens_before_padding():
    """Whisper normalises under an attention mask, so only real samples count.

    Folding the padding zeros into the mean and variance shifts every value the
    model sees, and the model responds by returning the same probability for
    everything -- which is exactly what it did before this was fixed.
    """
    audio = np.full(4000, 0.5, dtype=np.float32) + np.random.RandomState(0).randn(4000) * 0.01
    padded_then_normalised = normalise(
        np.pad(audio.astype(np.float32), (0, CHUNK_SAMPLES - len(audio)))
    )
    correct = window_features(audio)
    wrong = log_mel(padded_then_normalised)[np.newaxis, :, :CHUNK_FRAMES]
    assert not np.allclose(correct, wrong), "the two orders should differ"


def test_normalise_gives_zero_mean_unit_variance():
    audio = np.random.RandomState(1).randn(8000).astype(np.float32) * 3 + 5
    out = normalise(audio)
    assert abs(float(out.mean())) < 1e-4
    assert abs(float(out.std()) - 1.0) < 1e-3


def test_normalise_survives_silence():
    assert np.isfinite(normalise(np.zeros(1000, dtype=np.float32))).all()


def test_log_mel_dynamic_range_is_capped_and_gain_only_shifts_it():
    """Whisper floors at eight decades below the peak, then divides by four.

    Two consequences worth pinning. The span can never exceed 2.0, because the
    floor caps it. And a gain change shifts every value by the same amount
    without altering the span, which is why the extractor normalises the
    waveform first -- otherwise a quiet recording would land somewhere the model
    never saw. Note the output is NOT bounded to [-1, 1]: loud audio legitimately
    exceeds 1.
    """
    spans, peaks = [], []
    for scale in (0.01, 1.0, 4.0):
        audio = np.random.RandomState(2).randn(CHUNK_SAMPLES).astype(np.float32) * scale
        spec = log_mel(audio)
        assert spec.shape[0] == N_MELS
        span = float(spec.max()) - float(spec.min())
        assert span <= 2.0 + 1e-6, f"floor did not cap the range: {span}"
        spans.append(span)
        peaks.append(float(spec.max()))

    assert max(spans) - min(spans) < 1e-4, "gain changed the span, not just the level"
    assert peaks[0] < peaks[1] < peaks[2], "louder audio should sit higher"
