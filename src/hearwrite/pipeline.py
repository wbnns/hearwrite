"""One place that assembles the four models from a policy.

This exists because the CLI and the WebSocket service each used to build their
own, and they drifted: the server was written before diarization and semantic
endpointing existed and was never updated, so `serve --policy conversation`
quietly delivered no speaker labels and no semantic gate while `transcribe`
delivered both. Nothing failed. The output was just worse over one entrance than
the other.

So there is one builder, both entrances call it, and a test asserts the server
uses it.

Everything here is lazily imported. The base install has no model runtime at
all, and building a pipeline is the moment that stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .coordinator import Policy

#: Nemotron rather than the zipformer, on measurement. On one test recording it
#: committed sooner (p50 0.36s against 0.52s) AND produced punctuation and
#: capitalisation, which is most of why a raw transducer transcript reads as
#: wrong even when every word is right. It costs about five times the CPU:
#: real time factor 0.26 against 0.05 on Apple silicon.
#:
#: That trade is right for a laptop and wrong for a very small VPS. Use
#: `--model zipformer-en` where cycles are scarce; see docs/deployment.md.
DEFAULT_SHERPA_MODEL = "nemotron-3.5-160ms"


@dataclass(frozen=True)
class Backends:
    """Which implementation to use for each of the four interfaces."""

    engine: str = "sherpa"
    #: None means the engine's own default: nemotron for sherpa, base for
    #: whisper.
    model: str | None = None
    speaker_model: str = "titanet-small"
    turn_model: str = "smart-turn"
    language: str | None = None
    threads: int = 2
    #: Disable the acoustic gate. Endpointing then relies on the flush path.
    vad: bool = True
    #: Disable the semantic gate. Endpointing then reduces to a silence timer.
    turn: bool = True
    #: Punctuate committed text with a second model. None means "only if the
    #: recogniser does not do it already", which is the sensible default in both
    #: directions: redundant behind nemotron, and actively harmful, because the
    #: punctuation model mangles input that is already punctuated.
    punctuate: bool | None = None
    punctuate_model: str = "punct-en"
    #: Rewrite spoken numbers as figures: "two thousand nine" to 2009. Rules,
    #: not a model, so it is deterministic and costs microseconds. It applies
    #: whatever the recogniser is, because none of them write figures.
    normalise: bool = True
    #: ONNX Runtime execution provider. "cpu" everywhere by default, because it
    #: is the only one measured here and the alternatives are not always faster:
    #: CoreML came in SLOWER than CPU on Apple silicon for the int8 models this
    #: ships (0.33 against 0.24), which is common when a quantised graph gets
    #: converted for an accelerator that would rather have floats.
    provider: str = "cpu"


@dataclass
class Components:
    engine: Any
    vad: Any = None
    speakers: Any = None
    turn: Any = None
    polish: Any = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "vad": self.vad,
            "speakers": self.speakers,
            "turn": self.turn,
            "polish": self.polish,
        }


def build_engine(backends: Backends) -> Any:
    """The ASR engine. The Coordinator cannot tell the two shapes apart."""
    if backends.engine == "whisper":
        from .engines.whisper import WhisperStreamingEngine

        return WhisperStreamingEngine.from_model(
            backends.model or "base",
            num_threads=backends.threads,
            language=backends.language,
        )
    from .engines.sherpa import SherpaStreamingEngine

    return SherpaStreamingEngine.from_model(
        backends.model or DEFAULT_SHERPA_MODEL,
        num_threads=backends.threads,
        provider=backends.provider,
    )


def build(policy: Policy, backends: Backends | None = None) -> Components:
    """Assemble every component the policy actually needs.

    Solo mode does not merely ignore the speaker frontend, it never builds it:
    diarizing one voice occasionally splits that person in two, and the models
    cost real time on the hot path for the privilege.
    """
    backends = backends or Backends()
    components = Components(engine=build_engine(backends))

    if backends.vad:
        from .vad.silero import SileroVAD

        components.vad = SileroVAD.from_model()

    if not policy.is_solo:
        from .speakers.sherpa import SherpaSpeakerFrontend

        components.speakers = SherpaSpeakerFrontend.from_model(
            backends.speaker_model, num_threads=backends.threads
        )

    if backends.turn:
        from .turn.smart_turn import SmartTurnDetector

        components.turn = SmartTurnDetector.from_model(backends.turn_model)

    components.polish = build_polish(backends)
    return components


def build_polish(backends: Backends) -> Any:
    """Assemble the re-rendering chain, skipping what the recogniser already does.

    Inverse text normalisation always applies: no recogniser here writes figures,
    and turning spoken numbers into them is orthogonal to punctuation. The
    punctuation stage is the one that has to be asked about, because running it
    behind a model that already punctuates makes good text worse.
    """
    from .polish.base import Chain

    stages: list[Any] = []
    if _wants_punctuation(backends):
        from .polish.punctuation import PunctuationStage

        stages.append(PunctuationStage.from_model(backends.punctuate_model))
    if backends.normalise:
        from .polish.normalisation import NormalisationStage

        stages.append(NormalisationStage())
    return Chain(stages=tuple(stages)) if stages else None


def _wants_punctuation(backends: Backends) -> bool:
    """Whether a second model should re-render the text.

    Asked of the RECOGNISER, because that is where the answer lives: a model
    that punctuates natively must not be polished again, and one that emits bare
    words benefits enormously. An explicit setting still wins.
    """
    if backends.punctuate is not None:
        return backends.punctuate
    if backends.engine == "whisper":
        return False  # Whisper punctuates.
    from .models import REGISTRY

    spec = REGISTRY.get(backends.model or DEFAULT_SHERPA_MODEL)
    return not (spec.punctuates if spec else False)
