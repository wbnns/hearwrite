"""The polish pass: a second model that punctuates what the first one heard.

A streaming recogniser optimised for latency emits bare words. That reads as
wrong even when every word is right, which is most of why raw transducer output
feels broken. A small text model can put the punctuation and casing back a beat
later, for a fraction of the cost of a recogniser that does it natively.

THE HARD CONSTRAINT: a polish must never change what was said.

HearWrite's one rule is that a commit is never contradicted, and rewriting
"green" as "Green." is a change to a committed word. The way out is to be strict
about what a polish is allowed to be: a change of RENDERING, never of content.
So `polished` is a separate event that supersedes for display only, and the
Coordinator verifies that the polished text contains exactly the same words in
the same order before emitting it. A model that drops, adds or alters a word has
its output thrown away rather than shown.

That check is not paranoia. The punctuation model available here rewrites
already-punctuated input into nonsense ("stairs.." and "TEST one, two, three"),
so it runs only behind a recogniser that does not punctuate, and the invariant
is what makes that safe rather than merely intended.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

#: Everything that is not a letter, a digit or an apostrophe is punctuation as
#: far as this comparison is concerned.
_WORDS = re.compile(r"[^0-9a-z']+")


def words_of(text: str) -> list[str]:
    """The word sequence, ignoring case and punctuation entirely."""
    return [w for w in _WORDS.split(text.lower()) if w]


def preserves_words(original: str, polished: str) -> bool:
    """True when a polish changed only the rendering.

    This is the whole safety argument for running a second model over committed
    text, so it is deliberately strict: same words, same order, no additions and
    no removals.
    """
    return words_of(original) == words_of(polished)


@runtime_checkable
class Punctuator(Protocol):
    """Adds punctuation and casing to a finished utterance."""

    def polish(self, text: str, context: str = "") -> str:
        """Return `text` with punctuation and casing, and the same words.

        `context` is the tail of the PREVIOUS utterance, and it exists because
        an utterance is a fragment, not a sentence. Without it every fragment is
        punctuated as though it began a sentence, so a clause continuing across
        an endpoint comes back as "the Stairs" with a capital in the middle.
        With it, the model sees that the words continue and leaves them alone.

        Implementations must strip the context back off. It is context, not
        content, and the word preserving check is applied to what is returned.
        """
        ...

    def reset(self) -> None: ...
