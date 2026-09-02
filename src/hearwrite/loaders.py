"""Loading heavy models once and sharing them across sessions.

Without this, every WebSocket connection loads its own copy of everything: about
230MB and a second of startup per session, so the default admission limit of
four sessions wanted roughly a gigabyte of resident memory before anyone had
said a word. On a small VPS that is the whole machine.

Most of these models are shareable, and the split is not a guess:

  SHAREABLE -- the recognizer, the embedding extractor and the turn detector's
  ONNX session hold no per stream state. sherpa's design says so (you call
  `create_stream()` per stream) and it is verified rather than assumed: two
  streams interleaved chunk by chunk on one recognizer produce byte identical
  output to running each alone. See `test_sharing_a_recognizer_is_safe`.

  NOT SHAREABLE -- the VAD. `VadModel` has a `reset()`, which is the tell: it
  carries state across calls, so two sessions sharing one would contaminate each
  other's speech boundaries. It is also only 629KB, so a copy per session costs
  almost nothing.

Caches are bounded and never evicted under load; a model held here is one that
stays resident on purpose.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .models import resolve


def _sherpa() -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the ONNX backend is not installed.\n  pip install 'hearwrite[onnx]'"
        ) from exc
    return sherpa_onnx


@lru_cache(maxsize=4)
def transducer(
    name: str,
    *,
    num_threads: int = 2,
    provider: str = "cpu",
    decoding_method: str = "greedy_search",
    sample_rate: int = 16_000,
) -> Any:
    """A streaming recognizer, shared across every stream that wants it."""
    from .models import REGISTRY, find

    sherpa_onnx = _sherpa()
    directory = resolve(name)
    spec = REGISTRY.get(name) or REGISTRY["zipformer-en"]
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(find(directory, spec.tokens, what="tokens file")),
        encoder=str(find(directory, spec.encoder, what="encoder")),
        decoder=str(find(directory, spec.decoder, what="decoder")),
        joiner=str(find(directory, spec.joiner, what="joiner")),
        num_threads=num_threads,
        provider=provider,
        decoding_method=decoding_method,
        sample_rate=sample_rate,
        # HearWrite owns endpointing. Letting the recognizer also decide would
        # give two components an opinion about the same question.
        enable_endpoint_detection=False,
    )


@lru_cache(maxsize=4)
def speaker_embedder(name: str, *, num_threads: int = 2) -> Any:
    """A speaker embedding extractor, shared. One stream per segment."""
    sherpa_onnx = _sherpa()
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = str(resolve(name))
    config.num_threads = num_threads
    config.provider = "cpu"
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


@lru_cache(maxsize=4)
def turn_session(name: str, *, num_threads: int = 1) -> Any:
    """An ONNX session for the turn detector, shared. `run()` is thread safe."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the turn detector needs onnxruntime.\n  pip install 'hearwrite[turn]'"
        ) from exc
    options = ort.SessionOptions()
    options.inter_op_num_threads = num_threads
    options.intra_op_num_threads = num_threads
    return ort.InferenceSession(
        str(resolve(name)), sess_options=options, providers=["CPUExecutionProvider"]
    )


def vad_model(
    name: str,
    *,
    sample_rate: int = 16_000,
    threshold: float = 0.5,
    min_silence_duration: float = 0.1,
    min_speech_duration: float = 0.1,
    num_threads: int = 1,
) -> Any:
    """A VAD. NOT cached: it carries state, so every session needs its own."""
    sherpa_onnx = _sherpa()
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(resolve(name))
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence_duration
    config.silero_vad.min_speech_duration = min_speech_duration
    config.sample_rate = sample_rate
    config.num_threads = num_threads
    config.provider = "cpu"
    return sherpa_onnx.VadModel.create(config)


def clear() -> None:
    """Drop every shared model. For tests and for freeing memory deliberately."""
    transducer.cache_clear()
    speaker_embedder.cache_clear()
    turn_session.cache_clear()


def loaded() -> dict[str, int]:
    """How many of each shared model are resident."""
    return {
        "transducer": transducer.cache_info().currsize,
        "speaker_embedder": speaker_embedder.cache_info().currsize,
        "turn_session": turn_session.cache_info().currsize,
    }
