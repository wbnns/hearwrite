"""Backpressure: drop partials, never commits.

When the pipeline falls behind real time there are three options, and two of them
are bad. Silently growing a buffer is the worst and the easiest one to write by
accident -- latency climbs without bound and nothing reports it. Dropping audio
loses words permanently.

The third option is to keep every commit and shed the provisional work. Partials
are by contract revisable and withdrawable, so dropping them costs a consumer
nothing it was promised. The client is told via a `degraded` event, because
silently getting worse is its own failure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackpressureGate:
    """Tracks how far behind real time the pipeline is running."""

    max_lag_seconds: float
    _degraded: bool = False
    _lag: float = 0.0

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def lag(self) -> float:
        return self._lag

    def observe(self, lag_seconds: float) -> bool | None:
        """Report current lag. Returns the new state on a transition, else None.

        `lag_seconds` is how far the pipeline's processing trails the audio it
        has been handed. It is the one number in the Coordinator derived from
        wall time, and it deliberately never reaches an event timestamp.
        """
        self._lag = lag_seconds
        should_degrade = lag_seconds > self.max_lag_seconds
        if should_degrade != self._degraded:
            self._degraded = should_degrade
            return should_degrade
        return None

    def allows_partial(self) -> bool:
        return not self._degraded

    def reset(self) -> None:
        self._degraded = False
        self._lag = 0.0
