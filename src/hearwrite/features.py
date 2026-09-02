"""Whisper style log mel spectrograms, in numpy.

smart-turn takes an 80 by 800 log mel spectrogram, the same features Whisper
uses. Its reference implementation gets them from `transformers`, which is a
very large dependency to add for an eight megabyte model, so they are computed
here instead.

Reimplementing a feature extractor is a good way to be silently wrong: the
output looks plausible, the model still returns probabilities, and accuracy just
quietly drops. So this is checked against faster-whisper's independent
implementation in `tests/tier1/test_features.py` rather than trusted.

The pipeline matches Whisper exactly: reflect padding, a 400 sample Hann window
with hop 160, magnitude squared, a Slaney scale mel filterbank, then log10 with
Whisper's particular clamping and rescaling.
"""

from __future__ import annotations

import functools
import math
from typing import Any

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
SAMPLE_RATE = 16_000

#: smart-turn was trained on eight second windows.
CHUNK_SECONDS = 8
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH  # 800


def _hz_to_mel(hz: float) -> float:
    """Slaney scale, as used by librosa and therefore by Whisper."""
    f_min, f_sp = 0.0, 200.0 / 3
    mel = (hz - f_min) / f_sp
    min_log_hz = 1000.0
    if hz >= min_log_hz:
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        mel = min_log_mel + math.log(hz / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel: Any) -> Any:
    import numpy as np

    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        freqs,
    )


@functools.lru_cache(maxsize=4)
def mel_filters(n_mels: int = N_MELS, n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE):
    """Slaney normalised triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    import numpy as np

    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_min, mel_max = _hz_to_mel(0.0), _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    diff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]

    lower = -ramps[:-2] / diff[:-1][:, None]
    upper = ramps[2:] / diff[1:][:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalisation: each filter integrates to the same area, so wide
    # high frequency filters do not dominate narrow low frequency ones.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    return weights * enorm[:, None]


def normalise(samples: Any) -> Any:
    """Zero mean, unit variance over the whole window.

    Whisper's feature extractor does this when `do_normalize` is set, and
    smart-turn's reference sets it. Skipping it shifts every value the model
    sees.
    """
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return audio
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)


def log_mel(samples: Any, *, n_mels: int = N_MELS) -> Any:
    """Log mel spectrogram of a 1D float array, shape (n_mels, frames)."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)

    frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    strides = np.lib.stride_tricks.as_strided(
        padded,
        shape=(frames, N_FFT),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
    )
    spectrum = np.fft.rfft(strides * window, axis=-1)
    # Whisper drops the final frame, so 8s of audio yields exactly 800.
    magnitudes = (np.abs(spectrum[:-1]) ** 2).T

    mel_spec = mel_filters(n_mels) @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def window_features(samples: Any) -> Any:
    """Take the LAST eight seconds, pad if short, and return (1, 80, 800).

    The end of the audio is what matters: the question is whether the speaker
    has finished, and the evidence for that is in what they just said.
    """
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if len(audio) > CHUNK_SAMPLES:
        audio = audio[-CHUNK_SAMPLES:]
    # Normalise BEFORE padding. Whisper's extractor normalises under an
    # attention mask, so only real samples count; folding the padding zeros into
    # the mean and variance shifts every value the model sees, and the model
    # responds by returning the same probability for everything.
    audio = normalise(audio)
    if len(audio) < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
    return log_mel(audio)[np.newaxis, :, :CHUNK_FRAMES]
