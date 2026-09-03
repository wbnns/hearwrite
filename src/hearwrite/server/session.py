"""One session per connection.

The Coordinator is stateful and single threaded by design, so a session owns its
own engine, VAD and Coordinator and never shares them. That is the whole
concurrency model: no locks, no shared decoder state, and an explicit admission
limit instead of hoping.

Admission control is a semaphore rather than a queue on purpose. Latency
degradation under contention is the failure a user notices first, so refusing a
connection outright is kinder than accepting it and being slow for everyone.
"""

from __future__ import annotations

import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..coordinator import Coordinator, Policy
from ..events import Event


class Rejected(RuntimeError):
    """The server is at capacity."""


@dataclass
class Admission:
    """A hard cap on concurrent sessions."""

    limit: int
    _active: int = field(default=0, init=False)

    @property
    def active(self) -> int:
        return self._active

    def acquire(self) -> None:
        if self._active >= self.limit:
            raise Rejected(f"at capacity: {self._active} of {self.limit} sessions in use")
        self._active += 1

    def release(self) -> None:
        self._active = max(0, self._active - 1)


class Session:
    """Wraps a Coordinator with the wall clock bookkeeping the server needs."""

    def __init__(
        self,
        policy: Policy,
        *,
        engine,
        vad=None,
        speakers=None,
        turn=None,
        polish=None,
        record: Path | None = None,
    ) -> None:
        self.policy = policy
        self.coordinator = Coordinator(
            policy, engine=engine, vad=vad, speakers=speakers, turn=turn, polish=polish
        )
        self._started = time.monotonic()
        self._audio_seconds = 0.0
        # Recording exists for one reason: when a transcript is bad, the first
        # question is whether the audio was. Guessing at that from the far side
        # of a browser, a microphone and a resampler is how you end up tuning
        # the wrong thing.
        self._recorder = None
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            # The recorder spans the whole session and is closed in close(), so
            # a context manager cannot express its lifetime.
            self._recorder = wave.open(str(record), "wb")  # noqa: SIM115
            self._recorder.setnchannels(1)
            self._recorder.setsampwidth(2)
            self._recorder.setframerate(policy.sample_rate)

    @property
    def lag(self) -> float:
        """How far processing trails the audio handed over, in seconds.

        This is the one number derived from wall clock time, and it exists only
        to drive backpressure. It never reaches an event timestamp: those all
        come from the stream clock, which is what keeps a replayed session
        identical to a live one.
        """
        return max(0.0, (time.monotonic() - self._started) - self._audio_seconds)

    def push(self, pcm: bytes) -> Iterator[Event]:
        self._audio_seconds += len(pcm) / (2 * self.policy.sample_rate)
        if self._recorder is not None:
            self._recorder.writeframes(pcm)
        yield from self.coordinator.push(pcm, lag=self.lag)

    def finish(self) -> Iterator[Event]:
        yield from self.coordinator.finish()
        self.close()

    def close(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
