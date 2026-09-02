"""The stream clock: audio-relative time, never wall-clock time.

Every timestamp in a HearWrite event log is a position in the audio stream,
measured in seconds from the first sample. Nothing in the pipeline reads the
system clock to produce an event.

That is the property that makes replay work. A fixture pushed through the
Coordinator at 10x speed produces a byte-identical event log to the same fixture
pushed at 1x, because neither run ever asked what time it was. Wall-clock lag is
a real and useful operational metric, but it is measured separately and is not
part of the deterministic log -- mixing the two is what produces tests that pass
on a laptop and fail in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_SAMPLE = 2  # signed 16-bit little-endian PCM


@dataclass
class StreamClock:
    """Tracks how much audio has entered the pipeline.

    The clock only moves when audio is pushed. It is the single source of truth
    for `at`, `audio_start`, `audio_end` and `emitted_at` on every event.
    """

    sample_rate: int
    _samples: int = 0

    @property
    def position(self) -> float:
        """Stream position in seconds: the end of the audio pushed so far."""
        return self._samples / self.sample_rate

    @property
    def samples(self) -> int:
        return self._samples

    def advance_bytes(self, n: int) -> float:
        """Advance by `n` bytes of PCM and return the new position.

        Raises ValueError on a partial sample, because a caller that splits a
        sample across two pushes has a framing bug that would otherwise show up
        much later as a slow timestamp drift.
        """
        if n < 0:
            raise ValueError(f"cannot advance by {n} bytes")
        if n % BYTES_PER_SAMPLE:
            raise ValueError(f"{n} bytes is not a whole number of {BYTES_PER_SAMPLE}-byte samples")
        self._samples += n // BYTES_PER_SAMPLE
        return self.position

    def advance_seconds(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError(f"cannot advance by {seconds}s")
        self._samples += round(seconds * self.sample_rate)
        return self.position

    def reset(self) -> None:
        self._samples = 0


def duration_of(pcm: bytes, sample_rate: int) -> float:
    """Seconds of audio in a PCM buffer."""
    return len(pcm) / (BYTES_PER_SAMPLE * sample_rate)
