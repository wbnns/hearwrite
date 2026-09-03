"""A serialized chain of text stages over committed words.

Each stage re-renders a finished utterance: punctuation and casing from a small
model, spoken numbers into figures from rules. They run one after another in a
fixed order, and the design exists because THEY INTERFERE WITH EACH OTHER, which
is not a theoretical worry:

  * The punctuation model, given text that is ALREADY punctuated, returns
    "stairs.." and lowercases proper nouns. So a stage must be skipped when what
    it produces is already there.
  * Run inverse text normalisation first and the punctuation model uppercases
    the digits it does not understand: "January 3RD 2009". So the order is
    fixed, models first and deterministic rewrites last.

Three rules keep them out of each other's way:

  1. ORDER IS FIXED AND DECLARED, lowest first. Not the order someone happened
     to build the list in.
  2. A STAGE DECLARES WHAT IT PRODUCES and is skipped when that is already
     present, whether because the recogniser did it or an earlier stage did.
  3. A STAGE THAT FAILS ITS OWN CHECK IS DISCARDED, and the chain carries on
     from the previous text. One bad stage degrades the output by exactly its
     own contribution and takes nothing else with it.

Every stage is also cheap: the whole chain costs single digit milliseconds per
utterance, against tens or hundreds for the recogniser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Everything that is not a letter, digit or apostrophe is rendering.
_WORDS = re.compile(r"[^0-9a-z']+")


def words_of(text: str) -> list[str]:
    """The word sequence, ignoring case and punctuation entirely."""
    return [w for w in _WORDS.split(text.lower()) if w]


def preserves_words(original: str, polished: str) -> bool:
    """True when a change touched only rendering: same words, same order."""
    return words_of(original) == words_of(polished)


@runtime_checkable
class Stage(Protocol):
    """One re-rendering step."""

    #: Lower runs first. Model stages before deterministic rewrites, because a
    #: model mishandles the forms a rewrite produces.
    order: int
    #: What this stage adds, so it can be skipped when already present.
    produces: frozenset[str]
    name: str

    def apply(self, text: str, context: str = "") -> str:
        """Re-render `text`. `context` is the tail of the previous utterance."""
        ...

    def verify(self, before: str, after: str) -> bool:
        """Whether the change was legitimate for THIS stage.

        Each stage answers for itself, because the stages promise different
        things: punctuation must not change a word, while inverse text
        normalisation exists precisely to turn four words into one figure.
        """
        ...


@dataclass
class Chain:
    """Runs stages in order, skipping and discarding as the rules require."""

    stages: tuple[Stage, ...] = ()
    #: What the recogniser already provides, so those stages never run.
    already_present: frozenset[str] = frozenset()
    #: Stage names whose output failed verification, for reporting.
    rejected: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.stages = tuple(sorted(self.stages, key=lambda s: s.order))

    def run(self, text: str, context: str = "") -> str:
        if not text.strip():
            return text
        present = set(self.already_present)
        current = text
        for stage in self.stages:
            if stage.produces & present:
                continue
            try:
                candidate = stage.apply(current, context)
            except Exception:
                # A stage that raises must not take the utterance with it.
                self.rejected.append(stage.name)
                continue
            if candidate == current:
                continue
            if not stage.verify(current, candidate):
                self.rejected.append(stage.name)
                continue
            current = candidate
            present |= stage.produces
        return current
