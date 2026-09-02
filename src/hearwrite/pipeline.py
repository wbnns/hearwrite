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


@dataclass(frozen=True)
class Backends:
    """Which implementation to use for each of the four interfaces."""

    engine: str = "sherpa"
    model: str | None = None
    speaker_model: str = "titanet-small"
    turn_model: str = "smart-turn"
    language: str | None = None
    threads: int = 2
    #: Disable the acoustic gate. Endpointing then relies on the flush path.
    vad: bool = True
    #: Disable the semantic gate. Endpointing then reduces to a silence timer.
    turn: bool = True


@dataclass
class Components:
    engine: Any
    vad: Any = None
    speakers: Any = None
    turn: Any = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "vad": self.vad,
            "speakers": self.speakers,
            "turn": self.turn,
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
        backends.model or "zipformer-en", num_threads=backends.threads
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

    return components
