"""Diarization metrics, reported the way this project needs them.

Two decisions here matter more than the arithmetic.

FIRST, ABSTENTION IS NOT ERROR. A word the clustering declined to label and a
word it labelled wrongly are different outcomes with different costs, and the
whole design prefers the former. Collapsing both into one diarization error rate
makes a system that abstains look identical to one that guesses, which would
punish exactly the behaviour the append only rule requires. So confusion and
null rate are always reported side by side, never summed.

SECOND, SPEAKER LABELS ARE ARBITRARY. The pipeline invents A, B, C in the order
it hears voices; ground truth uses its own names. Any comparison has to find the
best mapping between the two first, or it measures nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

from .events import Event, EventKind


@dataclass(frozen=True)
class Turn:
    """A ground truth turn."""

    speaker: str
    start: float
    end: float

    def covers(self, at: float) -> bool:
        return self.start <= at <= self.end


@dataclass(frozen=True)
class SpeakerReport:
    words: int
    labelled: int
    unlabelled: int
    correct: int
    confused: int
    predicted_speakers: int
    true_speakers: int
    turn_label_latency: tuple[float, ...]

    @property
    def null_rate(self) -> float:
        """Share of words the clustering declined to label. Not an error."""
        return self.unlabelled / self.words if self.words else 0.0

    @property
    def confusion_rate(self) -> float:
        """Share of LABELLED words given to the wrong speaker. The real error."""
        return self.confused / self.labelled if self.labelled else 0.0

    @property
    def p50_turn_latency(self) -> float | None:
        return _percentile(self.turn_label_latency, 0.5)

    @property
    def p90_turn_latency(self) -> float | None:
        return _percentile(self.turn_label_latency, 0.9)


def load_turns(raw: Mapping[str, object]) -> tuple[Turn, ...]:
    turns = raw["turns"]
    assert isinstance(turns, list)
    return tuple(
        Turn(speaker=str(t["speaker"]), start=float(t["start"]), end=float(t["end"])) for t in turns
    )


def evaluate(events: Iterable[Event], turns: Sequence[Turn]) -> SpeakerReport:
    """Score committed words against ground truth turns."""
    commits = [e for e in events if e.kind is EventKind.COMMIT]
    filled = _apply_speaker_events(events)

    pairs: list[tuple[str | None, str | None]] = []
    for event in commits:
        middle = (event.payload["audio_start"] + event.payload["audio_end"]) / 2
        truth = next((t.speaker for t in turns if t.covers(middle)), None)
        predicted = filled.get(event.seq, event.payload["speaker"])
        pairs.append((predicted, truth))

    scorable = [(p, t) for p, t in pairs if t is not None]
    labelled = [(p, t) for p, t in scorable if p is not None]
    mapping = _best_mapping(labelled)
    correct = sum(1 for p, t in labelled if mapping.get(p) == t)

    return SpeakerReport(
        words=len(scorable),
        labelled=len(labelled),
        unlabelled=len(scorable) - len(labelled),
        correct=correct,
        confused=len(labelled) - correct,
        predicted_speakers=len({p for p, _ in labelled}),
        true_speakers=len({t.speaker for t in turns}),
        turn_label_latency=_turn_latencies(commits, filled, turns),
    )


def _apply_speaker_events(events: Iterable[Event]) -> dict[int, str]:
    """Later `speaker` events fill labels a commit left null."""
    filled: dict[int, str] = {}
    for event in events:
        if event.kind is EventKind.SPEAKER:
            filled[int(event.payload["seq"])] = str(event.payload["speaker"])
    return filled


def _best_mapping(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Map predicted labels onto true speakers so that agreement is maximised.

    Exhaustive while the label count is small, greedy beyond that. A wrong
    mapping would report a perfect system as broken, so this is not a place to
    approximate unless the arithmetic forces it.
    """
    predicted = sorted({p for p, _ in pairs})
    truth = sorted({t for _, t in pairs})
    if not predicted or not truth:
        return {}

    counts: dict[tuple[str, str], int] = {}
    for p, t in pairs:
        counts[(p, t)] = counts.get((p, t), 0) + 1

    if len(predicted) <= 7 and len(truth) <= 7:
        best: dict[str, str] = {}
        best_score = -1
        longer, shorter = (truth, predicted) if len(truth) >= len(predicted) else (predicted, truth)
        for arrangement in permutations(longer, len(shorter)):
            if longer is truth:
                candidate = dict(zip(shorter, arrangement, strict=True))
            else:
                candidate = {a: s for s, a in zip(shorter, arrangement, strict=True)}
            score = sum(counts.get((p, t), 0) for p, t in candidate.items())
            if score > best_score:
                best_score, best = score, candidate
        return best

    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for (p, t), _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        if p in mapping or t in taken:
            continue
        mapping[p] = t
        taken.add(t)
    return mapping


def _turn_latencies(
    commits: Sequence[Event], filled: Mapping[int, str], turns: Sequence[Turn]
) -> tuple[float, ...]:
    """How long after a turn begins before any word in it carries a speaker.

    Tracked separately from accuracy because a diarizer that is eventually right
    but two seconds late feels broken while scoring well.
    """
    out: list[float] = []
    for turn in turns:
        for event in commits:
            middle = (event.payload["audio_start"] + event.payload["audio_end"]) / 2
            if not turn.covers(middle):
                continue
            if filled.get(event.seq, event.payload["speaker"]) is None:
                continue
            out.append(max(0.0, float(event.at) - turn.start))
            break
    return tuple(out)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]
