"""Punctuation and casing via sherpa-onnx, on CPU.

31MB, about 6ms per utterance, so it costs roughly nothing next to a recogniser.

It has one sharp edge that decides where it can be used: it expects LOWERCASE,
UNPUNCTUATED input. Given text that is already punctuated it produces "stairs.."
and "TEST one, two, three"; given uppercase text it returns it unchanged. So the
input is lowercased first, and the pipeline only builds this at all behind a
recogniser whose own output is bare.
"""

from __future__ import annotations

from typing import Any

from ..models import find, resolve

#: Words of the previous utterance handed to the model as context. Enough to
#: establish that a clause continues; few enough to cost nothing.
CONTEXT_WORDS = 6


class SherpaPunctuator:
    """Wraps sherpa-onnx OnlinePunctuation."""

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

    def polish(self, text: str, context: str = "") -> str:
        if not text.strip():
            return text

        # Lowercase first. The model leaves uppercase input untouched, which
        # would silently make this whole stage a no-op behind a recogniser that
        # shouts -- which is exactly the recogniser it exists to help.
        lead = context.split()[-CONTEXT_WORDS:] if context else []
        joined = " ".join(lead + text.split()).lower()
        polished = self._punctuation.add_punctuation_with_case(joined)

        # Strip the context back off by word count. It was there to tell the
        # model this clause is a continuation, not to be shown again.
        return " ".join(polished.split()[len(lead) :])

    def reset(self) -> None:
        """Stateless between utterances."""
