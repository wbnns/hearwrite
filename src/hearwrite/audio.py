"""Reading audio files, and refusing to guess.

HearWrite works in 16kHz mono signed 16 bit PCM, because that is what every
model in the stack expects. This module reads a WAV file in that format and
reports a precise, actionable error for anything else.

It deliberately does NOT resample. Naive linear interpolation aliases, and
aliasing costs word error rate in a way that is invisible until someone
benchmarks it and blames the model. Telling the caller the exact ffmpeg command
is more honest than silently degrading their audio.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1


class AudioError(ValueError):
    """The file cannot be used as is."""


def read_wav(path: str | Path) -> bytes:
    """Read a 16kHz mono 16 bit WAV file and return raw PCM."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"no such file: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise AudioError(f"{path} is not a readable WAV file: {exc}") from exc

    problems = []
    if rate != SAMPLE_RATE:
        problems.append(f"sample rate is {rate}Hz, need {SAMPLE_RATE}Hz")
    if channels != CHANNELS:
        problems.append(f"{channels} channels, need mono")
    if width != SAMPLE_WIDTH:
        problems.append(f"{width * 8} bit samples, need 16 bit")

    if problems:
        raise AudioError(
            f"{path}: " + "; ".join(problems) + ".\n"
            f"Convert it first:\n"
            f"  ffmpeg -i {path} -ar {SAMPLE_RATE} -ac 1 -c:a pcm_s16le converted.wav"
        )
    return frames


def chunks(pcm: bytes, seconds: float, sample_rate: int = SAMPLE_RATE) -> Iterator[bytes]:
    """Slice PCM into fixed size pieces. The tail is yielded whole, however short."""
    size = int(seconds * sample_rate) * SAMPLE_WIDTH
    if size <= 0:
        raise AudioError(f"chunk of {seconds}s is too small to hold a sample")
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


def duration(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / (SAMPLE_WIDTH * sample_rate)
