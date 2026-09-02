"""The semantic gate: is that a finished thought.

This is the single biggest experiential gap between an acoustic VAD and a
learned endpoint token. A turn detector looks at what was actually said and
judges whether the utterance is syntactically and pragmatically complete, so a
speaker who pauses mid-sentence is not cut off.

Implementations are small enough to run on CPU beside everything else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TurnDetector(Protocol):
    """Scores how complete an utterance is, in [0.0, 1.0].

    1.0 means "certainly finished". 0.0 means "certainly mid-thought". The
    threshold that turns this into a decision lives in the endpoint policy, not
    here -- a dictation user and a voice agent want very different cutoffs from
    the same score.
    """

    def completeness(self, text: str, pcm: bytes | None = None) -> float: ...

    def reset(self) -> None: ...
