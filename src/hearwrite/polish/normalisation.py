"""Inverse text normalisation as a chain stage.

Runs LAST, after every model stage, because a text model mishandles the forms
this produces: given "January 3rd 2009" the punctuation model returns
"January 3RD 2009".
"""

from __future__ import annotations

from .base import words_of
from .itn import NUMBER_WORDS, ORDINALS, UNIT_FOLLOWERS, normalise


class NormalisationStage:
    order = 90
    produces = frozenset({"digits"})
    name = "normalisation"

    def apply(self, text: str, context: str = "") -> str:
        return normalise(text)

    def verify(self, before: str, after: str) -> bool:
        """Only number words may have changed.

        This stage exists to turn four words into one figure, so the punctuation
        stage's "same words" check would reject everything it does. The promise
        here is narrower and still checkable: every word that is NOT part of a
        number must survive, in order. A rule that rewrote "banks" would be
        caught even though rewriting "two thousand nine" is the point.
        """
        return _non_numeric(before) == _non_numeric(after)

    def reset(self) -> None:
        """Stateless."""


def _non_numeric(text: str) -> list[str]:
    skip = NUMBER_WORDS | set(ORDINALS) | UNIT_FOLLOWERS
    return [w for w in words_of(text) if w not in skip and not any(c.isdigit() for c in w)]
