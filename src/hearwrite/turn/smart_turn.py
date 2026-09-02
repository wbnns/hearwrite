"""smart-turn: the semantic half of the endpoint gate.

An acoustic VAD times silence. It cannot tell "what's the weather in..." from
"what's the weather in Menlo Park" when the pauses match, so an acoustic only
endpointer cuts people off mid thought. That is the single biggest experiential
gap between this stack and a learned endpoint token, and this model is what
closes it.

It is audio native rather than text based: it reads prosody as well as grammar,
so it works on the raw waveform and needs no transcript. Eight megabytes,
quantised to int8, about ten milliseconds on a CPU, twenty three languages, and
BSD-2 for the code AND the weights. The weights are vendored under
`src/hearwrite/vendor/smart_turn/` with their licence.

It only runs during silence. At twenty millisecond frames, scoring every push
would dominate the CPU budget for no benefit, since the answer cannot change
while somebody is still talking.
"""

from __future__ import annotations

import math
from typing import Any

#: Nothing shorter than this carries enough evidence to judge completeness.
MIN_AUDIO = 0.5


class SmartTurnDetector:
    """Scores whether the speaker has finished, from the audio alone."""

    def __init__(self, session: Any, *, sample_rate: int = 16_000) -> None:
        self._session = session
        self.sample_rate = sample_rate

    @classmethod
    def from_model(
        cls,
        name_or_path: str = "smart-turn",
        *,
        sample_rate: int = 16_000,
        num_threads: int = 1,
    ) -> SmartTurnDetector:
        """The ONNX session is shared: it holds no state and `run()` is safe."""
        from ..loaders import turn_session

        return cls(
            turn_session(name_or_path, num_threads=num_threads),
            sample_rate=sample_rate,
        )

    def completeness(self, text: str = "", pcm: bytes | None = None) -> float:
        """Probability in [0, 1] that the utterance is finished.

        Stateless: it scores whatever audio it is handed, which is the audio
        leading up to the pause. The Coordinator owns that rolling buffer,
        because the Coordinator is the one component allowed to hold state.

        `text` is accepted and ignored -- this model is audio native. The
        parameter exists because the protocol has to serve text based detectors
        too, and the Coordinator should not know which kind it has.
        """
        if not pcm:
            return 0.5
        samples = _to_floats(pcm)
        if len(samples) < MIN_AUDIO * self.sample_rate:
            # Too little evidence to judge. Returning 0.0 would veto every
            # endpoint and leave only the timeout; 1.0 would defeat the gate.
            # Say nothing useful and let the acoustic side decide.
            return 0.5
        from ..features import window_features

        logits = self._session.run(["logits"], {"input_features": window_features(samples)})[0]
        return 1.0 / (1.0 + math.exp(-float(logits[0][0])))

    def reset(self) -> None:
        """Nothing to reset. Kept so the protocol is satisfied uniformly."""


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
