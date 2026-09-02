"""The sherpa-onnx adapter's logic, with no model and no sherpa-onnx.

The adapter does three things the Coordinator would otherwise get wrong, and all
three are pure logic over a token stream. A stub recognizer makes them testable
in milliseconds, which is the same argument as every other fake in this suite:
the model's behaviour becomes an INPUT to the test rather than a dependency.
"""

from __future__ import annotations

import math

import pytest

from hearwrite.engines.sherpa import FLUSH_PADDING, SherpaStreamingEngine


class StubRecognizer:
    """Replays scripted token output the way sherpa-onnx would.

    `script` maps a cumulative sample count to the token list visible once the
    stream has been fed that much audio.
    """

    def __init__(self, script):
        self.script = script
        self.fed = 0

    def create_stream(self):
        self.fed = 0
        return self

    def is_ready(self, _stream):
        return False

    def decode_stream(self, _stream):  # pragma: no cover - never ready
        raise AssertionError("decode_stream called when not ready")

    def accept_waveform(self, _rate, samples):
        self.fed += len(samples)

    def input_finished(self):
        pass

    def _visible(self):
        best = []
        for threshold, entry in sorted(self.script.items()):
            if self.fed >= threshold:
                best = entry
        return best

    def tokens(self, _s):
        return [t for t, _, _ in self._visible()]

    def timestamps(self, _s):
        return [ts for _, ts, _ in self._visible()]

    def ys_probs(self, _s):
        return [p for _, _, p in self._visible()]


SR = 16_000


def pcm(seconds):
    return b"\x00\x00" * int(seconds * SR)


def engine_for(script):
    return SherpaStreamingEngine(StubRecognizer(script), sample_rate=SR)


def test_tokens_are_assembled_into_words_on_the_leading_space():
    """The model emits BPE pieces. A leading space starts a new word."""
    engine = engine_for(
        {
            SR: [(" ME", 0.1, 0.0), ("N", 0.2, 0.0), ("LO", 0.3, 0.0), (" PARK", 0.5, 0.0)],
        }
    )
    hypothesis = engine.push(pcm(1.0), 1.0)
    assert hypothesis is not None
    assert [w.text for w in hypothesis.stable] == ["MENLO"]
    assert [w.text for w in hypothesis.tentative] == ["PARK"]


def test_only_the_last_word_is_tentative():
    """Greedy transducer output is monotonic, so an emitted token is final.
    The last word is not, because it may still gain pieces.
    """
    engine = engine_for({SR: [(" A", 0.1, 0.0), (" B", 0.2, 0.0), (" C", 0.3, 0.0)]})
    hypothesis = engine.push(pcm(1.0), 1.0)
    assert [w.text for w in hypothesis.stable] == ["A", "B"]
    assert [w.text for w in hypothesis.tentative] == ["C"]


def test_a_growing_last_word_is_never_committed_early():
    """ "AFTER" becoming "AFTERNOON" must not surface as two committed words."""
    engine = engine_for(
        {
            SR: [(" AFTER", 1.0, 0.0)],
            2 * SR: [(" AFTER", 1.0, 0.0), ("NOON", 1.1, 0.0)],
        }
    )
    first = engine.push(pcm(1.0), 1.0)
    assert first.stable == ()
    assert [w.text for w in first.tentative] == ["AFTER"]

    second = engine.push(pcm(1.0), 2.0)
    assert second.stable == ()
    assert [w.text for w in second.tentative] == ["AFTERNOON"]


def test_no_new_tokens_returns_the_blank():
    """push() returning None IS the transducer's wait decision."""
    engine = engine_for({SR: [(" HELLO", 0.5, 0.0)]})
    assert engine.push(pcm(0.5), 0.5) is None  # nothing decoded yet
    assert engine.push(pcm(0.5), 1.0) is not None  # token appears
    assert engine.push(pcm(0.5), 1.5) is None  # nothing new


def test_flush_feeds_trailing_silence():
    """Without padding the decoder never releases its final words.

    Measured on a real clip: no padding lost "THIS AFTERNOON" entirely, 0.3s
    recovered "THIS AFTER", 0.5s recovered all of it.
    """
    stub = StubRecognizer({SR: [(" DONE", 0.5, 0.0)]})
    engine = SherpaStreamingEngine(stub, sample_rate=SR)
    engine.push(pcm(1.0), 1.0)
    before = stub.fed
    engine.flush()
    assert stub.fed - before == int(FLUSH_PADDING * SR)


def test_flush_makes_everything_stable():
    engine = engine_for({SR: [(" ONE", 0.2, 0.0), (" TWO", 0.6, 0.0)]})
    engine.push(pcm(1.0), 1.0)
    final = engine.flush()
    assert [w.text for w in final.stable] == ["ONE", "TWO"]
    assert final.tentative == ()


def test_word_end_never_claims_audio_that_has_not_arrived():
    """A token's duration is not reported, so the end is inferred. The estimate
    is bounded by the audio actually pushed, which is what keeps the Coordinator
    from rejecting it as an adapter bug.
    """
    engine = engine_for({SR: [(" EDGE", 0.98, 0.0)]})
    hypothesis = engine.push(pcm(1.0), 1.0)
    word = hypothesis.tentative[0]
    assert word.audio_end <= 1.0 + 1e-9
    assert word.audio_end >= word.audio_start


def test_confidence_comes_from_the_log_probabilities():
    engine = engine_for({SR: [(" SURE", 0.2, math.log(0.9))]})
    word = engine.push(pcm(1.0), 1.0).tentative[0]
    assert word.confidence == pytest.approx(0.9, abs=1e-6)


def test_confidence_is_clamped_to_one():
    engine = engine_for({SR: [(" CERTAIN", 0.2, 0.5)]})
    word = engine.push(pcm(1.0), 1.0).tentative[0]
    assert word.confidence <= 1.0


def test_decoding_is_incremental_not_quadratic():
    """A long session must not re-scan its whole token history every push."""
    script = {}
    tokens = []
    for i in range(200):
        tokens = [*tokens, (f" W{i}", i * 0.1, 0.0)]
        script[(i + 1) * 1600] = list(tokens)
    engine = engine_for(script)
    for i in range(200):
        engine.push(pcm(0.1), (i + 1) * 0.1)
    # Every token was consumed exactly once.
    assert engine._tokens_seen == 200
    assert len(engine._words) == 199


def test_reset_clears_all_state():
    engine = engine_for({SR: [(" WORD", 0.2, 0.0)]})
    engine.push(pcm(1.0), 1.0)
    engine.reset()
    assert engine._tokens_seen == 0
    assert engine._words == []
    assert engine._pending == []
