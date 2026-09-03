"""Punctuation and casing from a small ONNX model, as a chain stage.

31MB, about 6ms an utterance. It expects LOWERCASE, UNPUNCTUATED input: given
punctuated text it produces "stairs..", and given uppercase it returns the text
unchanged, which would make the stage a silent no-op behind exactly the
recogniser it exists to help. Both are handled by declaring what it produces so
the chain can skip it, and by lowercasing before the call.
"""

from __future__ import annotations

from typing import Any

from ..models import find, resolve
from .base import preserves_words

#: Words of the previous utterance handed over as context. An utterance is a
#: fragment, and without this every fragment is punctuated as though it began a
#: sentence, capitalising the middle of a clause: "the Stairs".
CONTEXT_WORDS = 6


class PunctuationStage:
    order = 10
    produces = frozenset({"punctuation", "casing"})
    name = "punctuation"

    def __init__(self, punctuation: Any) -> None:
        self._punctuation = punctuation

    @classmethod
    def from_model(cls, name_or_path: str = "punct-en", *, num_threads: int = 1):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise ImportError(
                "the ONNX backend is not installed.\n  pip install 'hearwrite[onnx]'"
            ) from exc

        directory = resolve(name_or_path)
        config = sherpa_onnx.OnlinePunctuationConfig()
        config.model_config.cnn_bilstm = str(
            find(directory, ("model.int8.onnx", "model.onnx"), what="punctuation model")
        )
        config.model_config.bpe_vocab = str(find(directory, ("bpe.vocab",), what="bpe vocab"))
        config.model_config.num_threads = num_threads
        config.model_config.provider = "cpu"
        return cls(sherpa_onnx.OnlinePunctuation(config))

    def apply(self, text: str, context: str = "") -> str:
        lead = context.split()[-CONTEXT_WORDS:] if context else []
        joined = " ".join(lead + text.split()).lower()
        polished = self._punctuation.add_punctuation_with_case(joined)
        # Strip the context back off. It was there to tell the model the clause
        # continues, not to be shown again.
        return " ".join(polished.split()[len(lead) :])

    def verify(self, before: str, after: str) -> bool:
        """Punctuation may change rendering and nothing else."""
        return preserves_words(before, after)

    def reset(self) -> None:
        """Stateless between utterances."""
