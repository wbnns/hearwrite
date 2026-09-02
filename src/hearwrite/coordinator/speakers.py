"""Online speaker clustering and word-to-speaker alignment.

This module carries most of HearWrite's technical risk, so it is written to fail
in the least damaging direction available.

The append-only rule is what forces the design. A speaker label attached to a
committed word can never be corrected, so a wrong label is permanent in a way a
missing label is not. Everything below follows from preferring abstention to a
guess:

  * A segment joins a cluster only if it clears the similarity threshold AND
    beats the runner-up by a margin. Ambiguous segments get no label; the word
    commits with `speaker: null` and a later `speaker` event fills it in once
    the identity is unambiguous.
  * Centroids are recomputed from a bounded ring buffer rather than accumulated
    into a running mean, so one long turn cannot drag a centroid onto itself.
  * Two clusters found to be the same person merge going forward. Already-frozen
    labels stand. That is a real, bounded imperfection, and it is documented
    rather than papered over.
  * Total clusters are capped with least-recently-heard eviction, which bounds
    memory over an hour-long session at the cost of giving a long-absent speaker
    a fresh ID when they return.

There is no fixed speaker count anywhere in here. Twenty speakers is twenty
centroids. What degrades as speakers are added is accuracy, not capacity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..speakers.base import Segment
from .policy import SpeakerPolicy


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity. Returns 0.0 for a zero vector rather than raising."""
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


@dataclass
class Cluster:
    """One discovered speaker."""

    label: str
    embeddings: deque[tuple[float, ...]]
    #: Total speech duration attributed to this cluster. Used to weight the
    #: centroid toward long, well-conditioned segments.
    weights: deque[float]
    last_heard: float = 0.0
    _centroid: tuple[float, ...] | None = field(default=None, repr=False)

    @property
    def centroid(self) -> tuple[float, ...]:
        if self._centroid is None:
            dim = len(self.embeddings[0])
            total = sum(self.weights) or 1.0
            acc = [0.0] * dim
            for emb, w in zip(self.embeddings, self.weights, strict=True):
                for i, v in enumerate(emb):
                    acc[i] += v * w
            self._centroid = tuple(v / total for v in acc)
        return self._centroid

    def observe(self, embedding: tuple[float, ...], weight: float, at: float) -> None:
        self.embeddings.append(embedding)
        self.weights.append(weight)
        self.last_heard = at
        self._centroid = None


@dataclass(frozen=True)
class Assignment:
    """The outcome of offering one segment to the clustering."""

    segment: Segment
    label: str | None
    #: Why no label was assigned. Useful in metrics: abstention and error are
    #: different outcomes and must not be collapsed into one DER number.
    reason: str = ""


class SpeakerTracker:
    """Assigns speaker labels to segments, then to words by timestamp overlap."""

    def __init__(self, policy: SpeakerPolicy) -> None:
        self._policy = policy
        self._clusters: dict[str, Cluster] = {}
        self._resolved: list[tuple[float, float, str]] = []
        self._next_label = 0
        #: Labels merged away, mapped to their survivor. Applied to future
        #: lookups only -- frozen labels are never rewritten.
        self._merged: dict[str, str] = {}

    @property
    def speaker_count(self) -> int:
        return len(self._clusters)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._clusters))

    def assign(self, segment: Segment) -> Assignment:
        """Offer one segment to the clustering. May return no label."""
        p = self._policy

        if segment.overlap:
            # Overlap is detected, not separated. Guessing which of two
            # simultaneous voices "owns" the segment is a coin flip, and a coin
            # flip is exactly what the append-only rule makes unaffordable.
            return Assignment(segment, None, "overlap")

        if segment.duration < p.min_duration:
            return Assignment(segment, None, "too_short")

        if not self._clusters:
            return Assignment(segment, self._new_cluster(segment), "")

        ranked = sorted(
            ((cosine(segment.embedding, c.centroid), c) for c in self._clusters.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score, best = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0

        if best_score < p.threshold:
            # Nobody is close enough. Treat it as a voice we have not heard.
            return Assignment(segment, self._new_cluster(segment), "")

        if best_score - runner_up < p.margin:
            # Close enough to two people to be worth nobody's guess.
            return Assignment(segment, None, "ambiguous")

        best.observe(segment.embedding, segment.duration, segment.end)
        self._resolved.append((segment.start, segment.end, best.label))
        return Assignment(segment, best.label, "")

    def label_for(self, start: float, end: float) -> str | None:
        """Label a word by the segment it overlaps most.

        A word straddling a boundary belongs to whichever speaker was talking for
        more of it. A word overlapping nothing is unlabeled, not guessed.
        """
        best_label: str | None = None
        best_overlap = 0.0
        for seg_start, seg_end, label in self._resolved:
            if seg_end <= start:
                continue
            if seg_start >= end:
                break
            overlap = min(end, seg_end) - max(start, seg_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label is None:
            return None
        return self.resolve(best_label)

    def resolve(self, label: str) -> str:
        """Follow any merges this label has been through."""
        seen: set[str] = set()
        while label in self._merged and label not in seen:
            seen.add(label)
            label = self._merged[label]
        return label

    def merge_candidates(self) -> list[tuple[str, str]]:
        """Cluster pairs now close enough to be the same person.

        Returns (survivor, absorbed) pairs. The survivor is the earlier label, so
        merging is stable and does not depend on iteration order.
        """
        pairs: list[tuple[str, str]] = []
        items = sorted(self._clusters.items())
        for i, (label_a, a) in enumerate(items):
            for label_b, b in items[i + 1 :]:
                # Merging is a stronger claim than assigning, so it needs a
                # stronger threshold. A merge that is wrong splits one person's
                # transcript across two names permanently.
                if cosine(a.centroid, b.centroid) >= self._policy.threshold + self._policy.margin:
                    pairs.append((label_a, label_b))
        return pairs

    def merge(self, survivor: str, absorbed: str) -> None:
        """Fold `absorbed` into `survivor` for open and future turns only."""
        if survivor not in self._clusters or absorbed not in self._clusters:
            return
        target = self._clusters[survivor]
        source = self._clusters.pop(absorbed)
        for emb, w in zip(source.embeddings, source.weights, strict=True):
            target.observe(emb, w, max(target.last_heard, source.last_heard))
        self._merged[absorbed] = survivor

    def _new_cluster(self, segment: Segment) -> str:
        if len(self._clusters) >= self._policy.max_speakers:
            self._evict_least_recent()
        label = self._mint_label()
        cluster = Cluster(
            label=label,
            embeddings=deque([segment.embedding], maxlen=self._policy.history),
            weights=deque([segment.duration], maxlen=self._policy.history),
            last_heard=segment.end,
        )
        self._clusters[label] = cluster
        self._resolved.append((segment.start, segment.end, label))
        return label

    def _evict_least_recent(self) -> None:
        victim = min(self._clusters.values(), key=lambda c: c.last_heard)
        del self._clusters[victim.label]

    def _mint_label(self) -> str:
        """A, B, ... Z, AA, AB, ... Never reused, even after an eviction."""
        n = self._next_label
        self._next_label += 1
        label = ""
        while True:
            label = chr(ord("A") + n % 26) + label
            n = n // 26 - 1
            if n < 0:
                return label
