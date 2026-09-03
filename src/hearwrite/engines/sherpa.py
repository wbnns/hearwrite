"""sherpa-onnx streaming transducer: the default engine.

This is the engine the ASR interface was designed around, so the adapter is
thin. A transducer decides, per chunk, either to emit tokens or to emit nothing,
and `ASREngine.push` returning `None` is exactly that blank decision. Nothing
here has to fake streaming.

Three things are genuinely the adapter's job, and each one is a real behaviour
the Coordinator would otherwise get wrong:

  * TOKENS ARE NOT WORDS. The model emits BPE pieces, and a piece beginning with
    a space starts a new word. Words are assembled here, incrementally, so a
    long session does not re-scan its whole token history on every 20ms push.

  * THE LAST WORD IS NOT FINISHED. Greedy transducer output is monotonic, so an
    emitted token never changes -- but the final word may still gain pieces
    ("AFTER" becoming "AFTERNOON"). Every word except the last is `stable`; the
    last is `tentative`. That is the honest split, and it is what lets the
    Coordinator commit safely.

  * A STREAM MUST BE FLUSHED WITH SILENCE. The decoder needs trailing audio
    before it will emit its final words. Without the padding in `flush`, every
    session silently loses the end of its last sentence -- measured, not
    assumed: on a test clip, no padding lost "THIS AFTERNOON" entirely, 0.3s
    recovered "THIS AFTER", and 0.5s recovered all of it.
"""

from __future__ import annotations

import array
import math
from pathlib import Path
from typing import Any

from ..models import resolve
from .base import Hypothesis, Word

#: Assumed duration of a token that has no successor yet. Observed spacing on a
#: streaming zipformer is 40 to 80ms; this is the generous end of that.
TOKEN_TAIL = 0.08

#: Trailing silence fed on flush so the decoder releases its final words.
FLUSH_PADDING = 0.5

_INT16_FULL_SCALE = 32768.0


class SherpaStreamingEngine:
    """Wraps a sherpa-onnx OnlineRecognizer as an ASREngine."""

    def __init__(
        self,
        recognizer: Any,
        *,
        sample_rate: int = 16_000,
        flush_padding: float = FLUSH_PADDING,
    ) -> None:
        self._recognizer = recognizer
        self.sample_rate = sample_rate
        self._flush_padding = flush_padding
        self._stream = recognizer.create_stream()
        self._tokens_seen = 0
        self._words: list[Word] = []
        self._pending: list[tuple[str, float, float]] = []
        self._consumed_to = 0.0

    @classmethod
    def from_model(
        cls,
        name_or_path: str = "nemotron-3.5-160ms",
        *,
        num_threads: int = 2,
        provider: str = "cpu",
        decoding_method: str = "greedy_search",
        sample_rate: int = 16_000,
    ) -> SherpaStreamingEngine:
        """Build an engine from a registry name or a directory of ONNX files.

        The recognizer comes from the shared cache, so a second session costs a
        stream rather than another 300MB of weights.
        """
        from ..loaders import transducer

        recognizer = transducer(
            name_or_path,
            num_threads=num_threads,
            provider=provider,
            decoding_method=decoding_method,
            sample_rate=sample_rate,
        )
        return cls(recognizer, sample_rate=sample_rate)

    # -- ASREngine ---------------------------------------------------------

    def push(self, pcm: bytes, at: float) -> Hypothesis | None:
        self._consumed_to = at
        self._feed(_to_floats(pcm))
        if not self._drain():
            # No new tokens. This is the blank: audio was processed, nothing to
            # say about it yet.
            return None
        return self._hypothesis(final=False)

    def flush(self) -> Hypothesis:
        padding = [0.0] * int(self._flush_padding * self.sample_rate)
        self._feed(padding)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        self._drain()
        return self._hypothesis(final=True)

    def reset(self) -> None:
        self._stream = self._recognizer.create_stream()
        self._tokens_seen = 0
        self._words.clear()
        self._pending.clear()
        self._consumed_to = 0.0

    # -- internals ---------------------------------------------------------

    def _feed(self, samples: list[float]) -> None:
        if not samples:
            return
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _drain(self) -> bool:
        """Turn any newly decoded tokens into words. True if anything arrived."""
        tokens = self._recognizer.tokens(self._stream)
        if len(tokens) <= self._tokens_seen:
            return False

        timestamps = self._recognizer.timestamps(self._stream)
        probs = self._recognizer.ys_probs(self._stream)

        for i in range(self._tokens_seen, len(tokens)):
            token = tokens[i]
            start = float(timestamps[i]) if i < len(timestamps) else self._consumed_to
            logp = float(probs[i]) if i < len(probs) else 0.0
            # A leading space marks the start of a new word, so the pending one
            # is now complete and its end is bounded by this token's start.
            if token.startswith(" ") and self._pending:
                self._words.append(_assemble(self._pending, cap=start))
                self._pending = []
            self._pending.append((token, start, logp))

        self._tokens_seen = len(tokens)
        return True

    def _hypothesis(self, *, final: bool) -> Hypothesis:
        tail: tuple[Word, ...] = ()
        if self._pending:
            tail = (_assemble(self._pending, cap=self._consumed_to),)
        if final:
            # End of stream: nothing can change any more, so nothing is tentative.
            return Hypothesis(stable=tuple(self._words) + tail, consumed_to=self._consumed_to)
        return Hypothesis(
            stable=tuple(self._words),
            tentative=tail,
            consumed_to=self._consumed_to,
            # The trailing word is assembled from sub word pieces and may be
            # half built: "Ja" on the way to "January". Committing it early
            # would deliver half a word permanently, not a word sooner.
            tentative_is_fragment=True,
        )


def _assemble(tokens: list[tuple[str, float, float]], *, cap: float) -> Word:
    """Join BPE pieces into one word.

    `cap` bounds the end timestamp. This is an estimate being kept honest, not a
    bad timestamp being hidden: a token's duration is not reported, so the end is
    inferred, and it must not be allowed to claim audio that has not arrived.
    """
    text = "".join(t for t, _, _ in tokens).strip()
    start = tokens[0][1]
    end = min(max(tokens[-1][1] + TOKEN_TAIL, start), max(cap, start))
    mean_logp = sum(p for _, _, p in tokens) / len(tokens)
    confidence = min(1.0, math.exp(mean_logp))
    return Word(text=text, audio_start=start, audio_end=end, confidence=confidence)


def _to_floats(pcm: bytes) -> list[float]:
    """Signed 16 bit little endian PCM to the floats sherpa-onnx expects."""
    if not pcm:
        return []
    samples = array.array("h")
    samples.frombytes(pcm)
    import sys

    if sys.byteorder != "little":  # pragma: no cover - no big endian CI
        samples.byteswap()
    return [s / _INT16_FULL_SCALE for s in samples]


def available(name_or_path: str = "zipformer-en") -> Path | None:
    """Local path if the model is already downloaded, else None."""
    try:
        return resolve(name_or_path, download=False)
    except Exception:
        return None
