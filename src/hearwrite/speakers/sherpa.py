"""Speaker frontend: cut speech into regions, embed each one.

The design doc calls for segmentation plus embeddings from sherpa-onnx, with the
clustering left to our own code. sherpa-onnx exposes the embedding extractor
standalone but not its segmentation model, which is reachable only through the
whole offline diarization pipeline -- and that pipeline does its own clustering,
which is exactly the part we need to own to handle more than four speakers.

So segmentation here is voice activity plus a fixed window:

  * A VAD cuts the stream into speech regions, and a region survives a short
    pause. Closing on the first quiet frame fragments ordinary speech into
    pieces too short to embed, which shows up as words with no speaker.
  * A long region is emitted as consecutive windows AS THEY COMPLETE, not when
    the region ends. Waiting for the end would delay every label by the length
    of the turn, and turn label latency is a metric this project cares about.

Windows are a FIXED length because the threshold that compares them is a fixed
number. Measured on 40 LibriSpeech speakers with TitaNet embeddings, similarity
between two windows of the same speaker climbs with window length (0.48 at 1s,
0.69 at 2s, 0.76 at 3s) while different speakers stay flat near 0.06. A short
window therefore looks unlike itself, and at a fixed threshold of 0.40 a one
second window splits the same speaker in two 19.6% of the time against 0.4% for
two seconds. Anything below the minimum is not embedded at all.

The honest limitation: a speaker change inside a window is detected at window
granularity, not at the instant it happens, and the window straddling the change
is a blend of two voices. That blend is what the clustering abstains on.
"""

from __future__ import annotations

import array
from typing import Any

from .base import Segment

_INT16_FULL_SCALE = 32768.0

#: Embedding window, in seconds. See the module docstring for the measurements.
WINDOW = 2.0

#: Shortest tail worth embedding when a region ends mid window.
MIN_REGION = 1.5

#: A region survives a pause shorter than this. Ordinary speech is full of gaps
#: at clause boundaries; treating each as a speaker boundary is what produced
#: 87% unlabelled words in the first version of this file.
MIN_SILENCE = 0.35

#: Hard cap on a buffered region, so someone who talks without pausing cannot
#: grow the buffer without bound.
MAX_REGION = 30.0


class SherpaSpeakerFrontend:
    """Emits (start, end, embedding) windows as speech accumulates."""

    def __init__(
        self,
        extractor: Any,
        vad: Any,
        *,
        sample_rate: int = 16_000,
        window: float = WINDOW,
        min_region: float = MIN_REGION,
        min_silence: float = MIN_SILENCE,
        max_region: float = MAX_REGION,
    ) -> None:
        self._extractor = extractor
        self._vad = vad
        self.sample_rate = sample_rate
        self._window = int(window * sample_rate)
        self._min_region = int(min_region * sample_rate)
        self._min_silence = min_silence
        self._max_region = int(max_region * sample_rate)

        self._samples: list[float] = []
        self._region_start: float | None = None
        self._silence_run = 0.0
        self._position = 0.0

    @classmethod
    def from_model(
        cls,
        name_or_path: str = "titanet-small",
        *,
        vad_model: str = "silero-vad",
        sample_rate: int = 16_000,
        num_threads: int = 2,
        **kwargs: Any,
    ) -> SherpaSpeakerFrontend:
        """The extractor is shared; the VAD is not, because it carries state."""
        from ..loaders import speaker_embedder
        from ..vad.silero import SileroVAD

        extractor = speaker_embedder(name_or_path, num_threads=num_threads)
        # Its own VAD instance. Sharing one with the Coordinator would couple two
        # components meant to be independently swappable.
        vad = SileroVAD.from_model(vad_model, sample_rate=sample_rate)
        return cls(extractor, vad, sample_rate=sample_rate, **kwargs)

    # -- SpeakerFrontend ---------------------------------------------------

    def push(self, pcm: bytes, at: float) -> tuple[Segment, ...]:
        state = self._vad.push(pcm, at)
        chunk = _to_floats(pcm)
        self._position = at

        if state.speaking:
            if self._region_start is None:
                self._region_start = max(0.0, at - len(chunk) / self.sample_rate)
            self._silence_run = 0.0
            self._samples.extend(chunk)
        elif self._region_start is not None:
            # Keep buffering through a short pause so the region stays whole.
            self._silence_run += len(chunk) / self.sample_rate
            self._samples.extend(chunk)
            if self._silence_run >= self._min_silence:
                return self._close()
        else:
            return ()

        if len(self._samples) >= self._max_region:
            return self._close()
        return self._drain_windows()

    def flush(self) -> tuple[Segment, ...]:
        if self._region_start is None:
            return ()
        return self._close()

    def reset(self) -> None:
        self._vad.reset()
        self._samples.clear()
        self._region_start = None
        self._silence_run = 0.0
        self._position = 0.0

    # -- internals ---------------------------------------------------------

    def _drain_windows(self) -> tuple[Segment, ...]:
        """Emit every complete window, then keep the remainder."""
        out: list[Segment] = []
        while len(self._samples) >= self._window and self._region_start is not None:
            out.append(self._segment(self._region_start, self._samples[: self._window]))
            del self._samples[: self._window]
            self._region_start += self._window / self.sample_rate
        return tuple(out)

    def _close(self) -> tuple[Segment, ...]:
        out = list(self._drain_windows())
        tail = self._samples
        start = self._region_start

        self._samples = []
        self._region_start = None
        self._silence_run = 0.0

        if start is not None and len(tail) >= self._min_region:
            out.append(self._segment(start, tail))
        return tuple(out)

    def _segment(self, start: float, samples: list[float]) -> Segment:
        return Segment(
            start=start,
            end=start + len(samples) / self.sample_rate,
            embedding=self._embed(samples),
        )

    def _embed(self, samples: list[float]) -> tuple[float, ...]:
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        stream.input_finished()
        return tuple(self._extractor.compute(stream))


def _to_floats(pcm: bytes) -> list[float]:
    if not pcm:
        return []
    samples = array.array("h")
    samples.frombytes(pcm)
    import sys

    if sys.byteorder != "little":  # pragma: no cover - no big endian CI
        samples.byteswap()
    return [s / _INT16_FULL_SCALE for s in samples]
