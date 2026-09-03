"""Endpointing from the recogniser's own punctuation.

A recogniser that writes "shells." has already decided a sentence ended, using
the whole acoustic and linguistic context it was trained on. That judgement is
free, and on some material it is markedly better than a separate turn detector.

Measured on a 39 second reading of verse: smart-turn scored SEVEN pauses between
0.60 and 0.73, and the highest scores went to "upon a", "the muttering" and
"what is" -- mid clause every time. It could not separate them from the one real
sentence boundary, because verse read aloud falls in pitch at every line end and
an audio native model reads that as finality.

Nemotron, meanwhile, put a full stop at exactly one place in those 39 seconds:
"and sawdust restaurants with oyster shells." Which was correct.

So this detector asks the transcript instead of the audio. It is only meaningful
behind a recogniser that punctuates -- ask it about block capitals with no full
stops and it will veto every endpoint forever -- so `pipeline` builds it only
there, and smart-turn keeps the job everywhere else.
"""

from __future__ import annotations

#: Marks that end a sentence. A comma or a dash means the speaker is mid flow.
TERMINAL = ".?!"

#: Returned when the transcript cannot answer, so the acoustic side decides
#: alone rather than being vetoed forever.
UNKNOWN = 0.5


class PunctuationTurnDetector:
    """Completeness from terminal punctuation in the committed text."""

    def __init__(self, *, confident: float = 1.0, incomplete: float = 0.0) -> None:
        self._confident = confident
        self._incomplete = incomplete

    def completeness(self, text: str, pcm: bytes | None = None) -> float:
        stripped = text.rstrip().rstrip("\"')]")
        if not stripped:
            return UNKNOWN
        return self._confident if stripped[-1] in TERMINAL else self._incomplete

    def reset(self) -> None:
        """Stateless: the transcript carries everything it needs."""
