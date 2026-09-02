"""faster-whisper behind the streaming engine interface.

Whisper is offline by construction: an encoder-decoder trained on 30 second
padded windows, with a bidirectional encoder that wants the whole utterance. It
does not stream. This adapter makes it behave as if it does, and the cost of
that pretence is worth stating plainly, because it is the reason the transducer
is the default engine and this one is the alternative.

The technique is LocalAgreement. Transcribe a growing buffer repeatedly; where
two consecutive passes agree on a prefix, treat that prefix as settled. Anything
past the agreement point is a guess that the next pass may overturn, so it is
`tentative`. This costs a full inference per update over an ever growing buffer,
which is why the buffer is trimmed at the agreement point rather than allowed to
grow with the session.

The interface fits without modification, which was the point of defining it
against a transducer first. `push` returns None between passes -- the same blank
decision a transducer makes per chunk, here meaning "no new inference has run
yet". Neither the Coordinator nor the wire protocol can tell the two engines
apart.

What you get for the cost: punctuation, casing, and about a hundred languages.
What you pay: latency measured in seconds rather than milliseconds, and twenty
transitive dependencies against sherpa-onnx's one.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Hypothesis, Word

#: Seconds of new audio between inference passes. Whisper's cost is roughly
#: constant per pass regardless of how much audio arrived, so this is the main
#: latency and CPU dial.
INTERVAL = 1.0

#: Left context retained past the agreement point when the buffer is trimmed.
#: Whisper transcribes a cut-off phrase badly, so the seam gets some history.
CONTEXT = 2.0

#: Hard cap on the buffer. Whisper's window is 30s; beyond that it truncates
#: internally and the extra audio is pure cost.
MAX_BUFFER = 28.0

_NORMALISE = re.compile(r"[^\w']+")


class WhisperStreamingEngine:
    """LocalAgreement over faster-whisper."""

    def __init__(
        self,
        model: Any,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        interval: float = INTERVAL,
        context: float = CONTEXT,
        max_buffer: float = MAX_BUFFER,
    ) -> None:
        self._model = model
        self.sample_rate = sample_rate
        self._language = language
        self._interval = int(interval * sample_rate)
        self._context = context
        self._max_buffer = int(max_buffer * sample_rate)

        self._buffer: list[float] = []
        self._offset = 0.0
        self._pending = 0
        self._previous: tuple[Word, ...] = ()
        self._stable: list[Word] = []
        self._settled_to = 0.0
        self._position = 0.0

    @classmethod
    def from_model(
        cls,
        name: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        num_threads: int = 2,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> WhisperStreamingEngine:
        """Build from any faster-whisper model name or local directory.

        `base` is multilingual and about 140MB. `small`, `medium`, `large-v3`
        and `distil-large-v3.5` all work and are all better and slower.

        Note that unlike every model in `hearwrite.models`, this downloads from
        the Hugging Face hub rather than the checksummed registry, so the pinned
        integrity guarantee does not extend to it.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise ImportError(
                "the whisper backend is not installed.\n  pip install 'hearwrite[whisper]'"
            ) from exc

        model = WhisperModel(
            name, device=device, compute_type=compute_type, cpu_threads=num_threads
        )
        return cls(model, sample_rate=sample_rate, language=language, **kwargs)

    # -- ASREngine ---------------------------------------------------------

    def push(self, pcm: bytes, at: float) -> Hypothesis | None:
        self._position = at
        samples = _to_floats(pcm)
        self._buffer.extend(samples)
        self._pending += len(samples)

        if self._pending < self._interval:
            # No pass has run, so there is nothing new to say. Same blank
            # decision a transducer makes, for a completely different reason.
            return None

        self._pending = 0
        self._absorb(self._transcribe())
        self._trim()
        return self._hypothesis(final=False)

    def flush(self) -> Hypothesis:
        current = self._transcribe()
        # End of stream: there will be no second pass to agree with, so the
        # remaining hypothesis is as settled as it will ever be.
        for word in current:
            if word.audio_start + 1e-6 >= self._settled_to:
                self._stable.append(word)
        self._stable.sort(key=lambda w: w.audio_start)
        self._previous = ()
        return Hypothesis(stable=tuple(self._stable), consumed_to=self._position)

    def reset(self) -> None:
        self._buffer.clear()
        self._offset = 0.0
        self._pending = 0
        self._previous = ()
        self._stable.clear()
        self._settled_to = 0.0
        self._position = 0.0

    # -- internals ---------------------------------------------------------

    def _transcribe(self) -> tuple[Word, ...]:
        if not self._buffer:
            return ()

        segments, _ = self._model.transcribe(
            _as_audio(self._buffer),
            word_timestamps=True,
            language=self._language,
            condition_on_previous_text=False,
        )
        out: list[Word] = []
        for segment in segments:
            for word in segment.words or ():
                text = word.word.strip()
                if not text:
                    continue
                start = self._offset + float(word.start)
                end = self._offset + float(word.end)
                # A word cannot end after the audio that has been pushed; the
                # Coordinator rejects that as an adapter bug, correctly.
                end = min(end, self._position)
                if end < start:
                    continue
                out.append(
                    Word(
                        text=text,
                        audio_start=start,
                        audio_end=end,
                        confidence=float(word.probability),
                    )
                )
        return tuple(out)

    def _absorb(self, current: tuple[Word, ...]) -> None:
        """Promote the prefix two consecutive passes agree on."""
        agreed = _common_prefix(self._previous, current)
        for word in agreed:
            if word.audio_start + 1e-6 >= self._settled_to:
                self._stable.append(word)
                self._settled_to = max(self._settled_to, word.audio_end)
        self._previous = current

    def _trim(self) -> None:
        """Drop audio behind the agreement point, keeping a little context.

        Without this the cost of every pass grows with the session, which is the
        structural reason Whisper streaming degrades over minutes in a way it
        never does in a ten second demo.
        """
        keep_from = max(0.0, self._settled_to - self._context)
        drop = int((keep_from - self._offset) * self.sample_rate)
        if len(self._buffer) > self._max_buffer:
            drop = max(drop, len(self._buffer) - self._max_buffer)
        if drop <= 0:
            return
        drop = min(drop, len(self._buffer))
        del self._buffer[:drop]
        self._offset += drop / self.sample_rate

    def _hypothesis(self, *, final: bool) -> Hypothesis:
        tentative = tuple(w for w in self._previous if w.audio_start + 1e-6 >= self._settled_to)
        if final:
            return Hypothesis(stable=tuple(self._stable) + tentative, consumed_to=self._position)
        return Hypothesis(
            stable=tuple(self._stable), tentative=tentative, consumed_to=self._position
        )


def _as_audio(buffer: list[float]):
    """faster-whisper wants a float32 array; a plain list is enough for a stub.

    The fallback is not a nicety. It keeps the LocalAgreement logic in this file
    testable in Tier 1, which must run with no model runtime installed at all.
    """
    try:
        import numpy as np
    except ImportError:
        return buffer
    return np.array(buffer, dtype=np.float32)


def _key(word: Word) -> str:
    return _NORMALISE.sub("", word.text).lower()


def _common_prefix(previous: tuple[Word, ...], current: tuple[Word, ...]) -> tuple[Word, ...]:
    """Words two passes agree on, compared by normalised text.

    Punctuation and casing are ignored, because Whisper revises both freely
    while the underlying word stays put, and treating "today" and "today?" as a
    disagreement would stall the commit frontier forever.
    """
    out: list[Word] = []
    for a, b in zip(previous, current, strict=False):
        if _key(a) != _key(b):
            break
        # Prefer the newer pass: same word, possibly better punctuation.
        out.append(b)
    return tuple(out)


def _to_floats(pcm: bytes) -> list[float]:
    if not pcm:
        return []
    import array
    import sys

    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":  # pragma: no cover - no big endian CI
        samples.byteswap()
    return [s / 32768.0 for s in samples]
