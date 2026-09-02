"""Silero VAD, ONNX build, via sherpa-onnx.

Deliberately NOT the `silero-vad` pip package. That one is a torch model, and
adding torch to the default install would cost roughly two gigabytes and end the
"runs on a laptop with no GPU" property that the rest of the design is built
around. sherpa-onnx already ships an ONNX Silero, so the acoustic gate runs in
the same runtime as the recognizer and costs a 629KB download.

The model insists on a fixed window (512 samples, 32ms, at 16kHz) while the
Coordinator pushes whatever the caller gave it, often 20ms. This adapter buffers
to whole windows. Buffering is what keeps the decision sequence identical
regardless of how the caller sliced its audio, which is the same chunk size
invariance the transcript has.
"""

from __future__ import annotations

import array
from typing import Any

from .base import SpeechState

_INT16_FULL_SCALE = 32768.0


class SileroVAD:
    """Frame level speech detection with an ONNX Silero model."""

    def __init__(self, model: Any, *, sample_rate: int = 16_000) -> None:
        self._model = model
        self.sample_rate = sample_rate
        window = model.window_size
        self._window = int(window() if callable(window) else window)
        self._buffer: list[float] = []
        self._speaking = False
        self._since = 0.0
        self._started = False

    @classmethod
    def from_model(
        cls,
        name_or_path: str = "silero-vad",
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_silence_duration: float = 0.1,
        min_speech_duration: float = 0.1,
        num_threads: int = 1,
    ) -> SileroVAD:
        """Build a VAD. Deliberately NOT shared: it carries state across calls,
        so two sessions sharing one would contaminate each other's speech
        boundaries. At 629KB a copy per session costs almost nothing.
        """
        from ..loaders import vad_model

        return cls(
            vad_model(
                name_or_path,
                sample_rate=sample_rate,
                threshold=threshold,
                min_silence_duration=min_silence_duration,
                min_speech_duration=min_speech_duration,
                num_threads=num_threads,
            ),
            sample_rate=sample_rate,
        )

    def push(self, pcm: bytes, at: float) -> SpeechState:
        self._buffer.extend(_to_floats(pcm))

        while len(self._buffer) >= self._window:
            window = self._buffer[: self._window]
            del self._buffer[: self._window]
            speaking = bool(self._model.is_speech(window))
            if speaking != self._speaking or not self._started:
                self._speaking = speaking
                self._since = at
                self._started = True

        return SpeechState(speaking=self._speaking, at=at, since=self._since)

    def reset(self) -> None:
        self._model.reset()
        self._buffer.clear()
        self._speaking = False
        self._since = 0.0
        self._started = False


def _to_floats(pcm: bytes) -> list[float]:
    if not pcm:
        return []
    samples = array.array("h")
    samples.frombytes(pcm)
    import sys

    if sys.byteorder != "little":  # pragma: no cover - no big endian CI
        samples.byteswap()
    return [s / _INT16_FULL_SCALE for s in samples]
