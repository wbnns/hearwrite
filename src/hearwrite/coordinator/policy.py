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


# Completeness thresholds are calibrated for smart-turn v3.1 on a 40 pair mid
# thought corpus, and they trade the two failures against each other:
#
#   thr 0.70 -> cuts someone off  5% of the time, waits for the timeout 60%
#   thr 0.60 -> cuts someone off 15% of the time, waits for the timeout 33%
#   thr 0.55 -> cuts someone off 23% of the time, waits for the timeout 23%
#
# Waiting is not a hang: the timeout fallback still closes the turn. So the
# conservative end buys "never interrupt me" with latency, which is exactly the
# trade dictation wants and a voice agent does not.
_ENDPOINT_PRESETS = {
    EndpointMode.CONSERVATIVE: EndpointPolicy(1.0, 0.70, 4.0),
    EndpointMode.BALANCED: EndpointPolicy(0.6, 0.60, 1.4),
    EndpointMode.AGGRESSIVE: EndpointPolicy(0.25, 0.55, 1.2),
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
    #: How long a resolved segment is kept for labelling words.
    #:
    #: Words are labelled at commit time, moments after their audio, and an
    #: unlabelled one is abandoned when its turn closes. So nothing needs a
    #: segment from two minutes ago, and keeping them is not free: `label_for`
    #: scans the list, so an unpruned session pays 1us per word in the first
    #: minutes and 61us per word four hours in. Linear per word is quadratic
    #: over a session.
    segment_memory: float = 120.0
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
    # C1 confidence gating: two dials that trade accuracy against latency in
    # opposite directions, which is the point. The reference system learns a
    # per word delay; this approximates it with policy and no training.
    #
    #: Hold a STABLE word this unconfident for `slow_commit_seconds` of further
    #: audio, in case the engine changes its mind. Costs latency, buys accuracy.
    #: 0.0 disables it.
    confidence_gate: float = 0.0
    slow_commit_seconds: float = 0.35
    #: Commit a TENTATIVE word this confident without waiting for the engine to
    #: call it settled. Buys latency, costs accuracy. Only meaningful for an
    #: engine that has a real tentative state to skip -- a greedy transducer
    #: settles immediately, so there is nothing to skip and this does nothing.
    #: 1.01 disables it (no probability can reach it).
    early_commit_confidence: float = 1.01
    #: Minimum gap between two turn detector runs.
    #:
    #: This is a correctness dial, not only a cost one. The completeness score
    #: is noisy frame to frame -- measured on one pause it read 0.58, 0.61,
    #: 0.67, 0.66, 0.61, 0.65, 0.61, 0.65, then spiked to 0.71 -- and the gate
    #: fires on the FIRST frame that crosses. Scoring at frame rate therefore
    #: turns the threshold into "the maximum over fifty samples a second",
    #: which is far more permissive than the number it was calibrated as, and
    #: would push the false endpoint rate above what docs/evaluation.md reports.
    #: Sampling a few times a second keeps the runtime decision close to the
    #: single window decision the calibration measured. It also happens to cost
    #: a great deal less.
    turn_interval: float = 0.2
    #: Seconds of audio handed to the turn detector. smart-turn was trained on
    #: eight second windows, so more is wasted and less is a different model.
    turn_context_seconds: float = 8.0
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
