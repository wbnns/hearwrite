"""Policy: every tunable in one frozen object, not scattered constants.

Speaker mode and endpoint aggressiveness are ORTHOGONAL AXES, because they vary
independently. A solo voice agent wants one speaker and an impatient endpoint. A
meeting wants many speakers and a patient one. Dictation wants one speaker and a
very patient one. Three opaque presets cannot express that; two axes can.

Named presets are just combinations, kept because most callers want a sensible
default rather than a tuning session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class SpeakerMode(StrEnum):
    #: One speaker. The speaker frontend is not run at all -- see below.
    SOLO = "solo"
    #: Unbounded speakers, discovered online by clustering.
    AUTO = "auto"


class EndpointMode(StrEnum):
    #: Long trailing pauses are fine. Dictation.
    CONSERVATIVE = "conservative"
    #: Meeting capture.
    BALANCED = "balanced"
    #: Latency is king. Voice agents.
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class EndpointPolicy:
    """Thresholds for the conjunctive endpoint gate."""

    #: Acoustic gate: silence this long is a candidate endpoint.
    silence_seconds: float
    #: Semantic gate: completeness at or above this counts as a finished thought.
    completeness_threshold: float
    #: Timeout fallback. If the acoustic gate has held this long, endpoint
    #: regardless of what the semantic gate says. Without it a speaker who
    #: trails off hangs the session forever.
    max_silence_seconds: float


_ENDPOINT_PRESETS = {
    EndpointMode.CONSERVATIVE: EndpointPolicy(1.0, 0.75, 4.0),
    EndpointMode.BALANCED: EndpointPolicy(0.6, 0.60, 2.0),
    EndpointMode.AGGRESSIVE: EndpointPolicy(0.25, 0.45, 1.2),
}


@dataclass(frozen=True)
class SpeakerPolicy:
    """Tuning for the online clustering. Ignored entirely when mode is SOLO."""

    mode: SpeakerMode = SpeakerMode.AUTO
    #: A segment joins a cluster only if cosine similarity clears this.
    #: Calibrated on 40 LibriSpeech speakers with TitaNet embeddings over 1.5 to
    #: 3 second windows: different speakers sit at about 0.06 cosine, the same
    #: speaker at 0.62 to 0.76. At 0.40 the measured error is under 0.6% merges
    #: and under 2.5% splits. Retune this if you change the embedding model --
    #: the right value is a property of the model, not of speech.
    threshold: float = 0.40
    #: ...AND beats the runner-up cluster by this margin. Without the margin,
    #: two similar voices ping-pong between labels; with it, an ambiguous
    #: segment abstains and commits `speaker: null` instead of guessing.
    margin: float = 0.10
    #: Segments shorter than this give embeddings too noisy to cluster on.
    #: See MIN_REGION in speakers/sherpa.py for the measurements behind it.
    min_duration: float = 1.5
    #: How far a word may reach to borrow a label from a nearby segment.
    #: Speech regions never tile the timeline perfectly -- a tail too short to
    #: embed leaves a hole -- so a word inside a homogeneous stretch takes the
    #: surrounding speaker rather than committing null for want of a window.
    #: Reaching across a stretch where the neighbours DISAGREE is the one case
    #: this must not do, and does not.
    max_gap: float = 2.0
    #: Embeddings retained per cluster. Bounds memory over long sessions and
    #: lets the centroid be recomputed robustly instead of drifting toward
    #: whoever spoke most recently.
    history: int = 24
    #: Hard cap on simultaneous clusters. Least-recently-heard is evicted; a
    #: returning speaker then gets a fresh ID. That is the honest trade for
    #: bounded memory over a 60-minute session.
    max_speakers: int = 32


@dataclass(frozen=True)
class Policy:
    sample_rate: int = 16_000
    speakers: SpeakerPolicy = SpeakerPolicy()
    endpoint: EndpointPolicy = _ENDPOINT_PRESETS[EndpointMode.BALANCED]
    #: C1 confidence gating (Phase 3). Commit a stable word immediately when the
    #: engine is this confident; otherwise hold it for `slow_commit_seconds` of
    #: further audio to see whether it changes. The cheapest approximation of
    #: the reference system's learned per-word delay, with no training.
    confidence_gate: float = 0.0
    slow_commit_seconds: float = 0.35
    #: Backpressure: fall this far behind real time and partials get dropped.
    max_lag_seconds: float = 1.5

    @property
    def is_solo(self) -> bool:
        return self.speakers.mode is SpeakerMode.SOLO

    def with_endpoint(self, mode: EndpointMode) -> Policy:
        return replace(self, endpoint=_ENDPOINT_PRESETS[mode])


def preset(name: str) -> Policy:
    """Look up a named preset. Raises KeyError with the valid names on a typo."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; expected one of {sorted(PRESETS)}") from None


CONVERSATION = Policy(
    speakers=SpeakerPolicy(mode=SpeakerMode.AUTO),
    endpoint=_ENDPOINT_PRESETS[EndpointMode.BALANCED],
)

DICTATION = Policy(
    speakers=SpeakerPolicy(mode=SpeakerMode.SOLO),
    endpoint=_ENDPOINT_PRESETS[EndpointMode.CONSERVATIVE],
)

AGENT = Policy(
    speakers=SpeakerPolicy(mode=SpeakerMode.SOLO),
    endpoint=_ENDPOINT_PRESETS[EndpointMode.AGGRESSIVE],
)

PRESETS: dict[str, Policy] = {
    "conversation": CONVERSATION,
    "dictation": DICTATION,
    "agent": AGENT,
}
