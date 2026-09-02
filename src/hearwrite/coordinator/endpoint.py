"""The endpoint gate: acoustic AND semantic, with a timeout fallback.

Silence alone is not an endpoint. "What's the weather in--" and "What's the
weather in Menlo Park" are indistinguishable to a VAD if the pauses match, so an
acoustic-only endpoint cuts people off mid-thought. That is the single biggest
experiential gap between this stack and a learned endpoint token.

So the gate is conjunctive: silence long enough AND the utterance reads as
finished. Both, or neither.

The fallback is not optional. A speaker who trails off mid-sentence never
satisfies the semantic gate, so without a ceiling on silence the session hangs
forever. `max_silence_seconds` is what stops that, and it is the reason this is
a state machine rather than a boolean expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..vad.base import SpeechState
from .policy import EndpointPolicy


class Reason(StrEnum):
    #: Both gates agreed. The good case.
    COMPLETE = "complete"
    #: The acoustic gate held past the ceiling. The speaker trailed off, or the
    #: turn detector cannot judge this language.
    TIMEOUT = "timeout"
    #: End of stream.
    FLUSH = "flush"


@dataclass(frozen=True)
class Endpoint:
    at: float
    reason: Reason
    completeness: float


class EndpointGate:
    """Tracks silence and decides when an utterance has actually ended."""

    def __init__(self, policy: EndpointPolicy) -> None:
        self._policy = policy
        self._silence_since: float | None = None
        self._fired_for_silence: float | None = None
        self._had_speech = False

    @property
    def in_silence(self) -> bool:
        return self._silence_since is not None

    def silence_held(self, at: float) -> float:
        """How long the current run of silence has lasted, 0.0 if speaking.

        The Coordinator uses this to avoid running the turn detector on every
        20ms frame. The semantic score only matters once the acoustic gate is
        close to satisfied, so it is only computed then.
        """
        if self._silence_since is None or not self._had_speech:
            return 0.0
        return max(0.0, at - self._silence_since)

    @property
    def silence_since(self) -> float | None:
        """Stream position where the current run of silence began, if any."""
        return self._silence_since if self._had_speech else None

    @property
    def wants_completeness(self) -> bool:
        """True when a semantic score could change the outcome right now."""
        return self._silence_since is not None and self._had_speech

    def reset(self) -> None:
        self._silence_since = None
        self._fired_for_silence = None
        self._had_speech = False

    def observe(self, state: SpeechState, completeness: float) -> Endpoint | None:
        """Feed one VAD reading plus the current semantic score.

        `completeness` is only consulted once the acoustic gate is satisfied, so
        a caller may pass a stale or default score while speech is ongoing.
        """
        if state.speaking:
            self._had_speech = True
            self._silence_since = None
            self._fired_for_silence = None
            return None

        # Silence with nothing before it is not the end of anything.
        if not self._had_speech:
            return None

        if self._silence_since is None:
            self._silence_since = state.at
            return None

        held = state.at - self._silence_since

        # Already fired for this run of silence. One endpoint per utterance.
        if self._fired_for_silence == self._silence_since:
            return None

        if held >= self._policy.max_silence_seconds:
            return self._fire(state.at, Reason.TIMEOUT, completeness)

        # The conjunction: silence long enough AND the utterance reads finished.
        if (
            held >= self._policy.silence_seconds
            and completeness >= self._policy.completeness_threshold
        ):
            return self._fire(state.at, Reason.COMPLETE, completeness)

        return None

    def flush(self, at: float, completeness: float = 1.0) -> Endpoint | None:
        """End of stream. Close an open utterance, if there is one."""
        if not self._had_speech:
            return None
        if self._fired_for_silence is not None and self._silence_since == self._fired_for_silence:
            return None
        return self._fire(at, Reason.FLUSH, completeness)

    def _fire(self, at: float, reason: Reason, completeness: float) -> Endpoint:
        self._fired_for_silence = self._silence_since
        self._had_speech = False
        return Endpoint(at=at, reason=reason, completeness=completeness)
