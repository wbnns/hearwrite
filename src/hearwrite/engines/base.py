"""The ASR engine interface, defined to the TRANSDUCER's shape.

This is the single most important interface in HearWrite, and the shape of it is
a deliberate bet.

The obvious move is to define the interface around Whisper, because Whisper is
familiar. That guarantees a rewrite: Whisper is an offline encoder-decoder that
emulates streaming by re-running inference over a growing buffer, so an interface
built around it encodes "re-transcribe everything and diff" as the model of the
world. A streaming transducer does not work that way.

The defining property of a transducer is that THE BLANK SYMBOL IS A WAIT
DECISION. Per chunk, the model either emits tokens or declines to. So `push`
returns `Hypothesis | None`, and returning `None` *is* the blank -- a
first-class "I processed this audio and chose to emit nothing", not an error and
not an empty result.

sherpa-onnx maps onto this natively. The Whisper adapter is the one that has to
work to fit, returning None between LocalAgreement passes and promoting words
from `tentative` to `stable` once two consecutive passes agree. Designing to the
easier case first is what would have cost us the rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Word:
    """One recognized word with its position in the audio stream."""

    text: str
    audio_start: float
    audio_end: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.audio_end < self.audio_start:
            raise ValueError(
                f"word {self.text!r} ends ({self.audio_end}) before it starts ({self.audio_start})"
            )


@dataclass(frozen=True)
class Hypothesis:
    """What the engine currently believes, split by how much it will stand behind.

    `stable` words are ones the engine is willing to be held to; the Coordinator
    may commit them. `tentative` words may change on the next push and may only
    be emitted as `partial`. `consumed_to` is how far into the stream the engine
    has actually processed, which is what lets the Coordinator distinguish
    "still thinking" from "nothing was said".
    """

    stable: tuple[Word, ...] = ()
    tentative: tuple[Word, ...] = ()
    consumed_to: float = 0.0
    #: True when the LAST tentative word may still be a fragment rather than a
    #: whole word.
    #:
    #: The two engine shapes mean different things by "tentative" and the
    #: difference is not cosmetic. A transducer assembles a word from sub word
    #: pieces, so its trailing entry can be "Ja" on the way to "January".
    #: LocalAgreement over an offline model yields whole words that may later be
    #: replaced by other whole words.
    #:
    #: Committing the first kind early truncates it. Measured: with early commit
    #: on, "The Times January 3rd 2009 Chancellor" came back as "The Time Ja
    #: third 2009 Ch". So a fragment is shown as a partial and never committed
    #: ahead of the engine.
    tentative_is_fragment: bool = False

    @property
    def all_words(self) -> tuple[Word, ...]:
        return self.stable + self.tentative


@runtime_checkable
class ASREngine(Protocol):
    """A streaming recognizer. Stateless from the Coordinator's point of view.

    Implementations may hold internal decoder state, but they hold no policy:
    what gets committed and when is the Coordinator's decision, never the
    engine's.
    """

    sample_rate: int

    def push(self, pcm: bytes, at: float) -> Hypothesis | None:
        """Feed audio. Return a hypothesis, or None for the blank/wait decision.

        `at` is the stream position of the END of this chunk.
        """
        ...

    def flush(self) -> Hypothesis:
        """End of stream. Return everything left, all of it stable."""
        ...

    def reset(self) -> None:
        """Drop all internal state. Used between sessions and in tests."""
        ...
