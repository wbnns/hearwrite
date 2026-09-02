"""LocalAgreement over an offline model, with no model.

What is under test is the pretence: making an encoder-decoder that wants the
whole utterance behave like something that streams. The model is stubbed, so
each pass returns exactly the hypothesis a test wants it to, including the
hypothesis that changes its mind.
"""

from __future__ import annotations

from hearwrite.engines.whisper import WhisperStreamingEngine

SR = 16_000


class StubWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class StubSegment:
    def __init__(self, words):
        self.words = words


class StubModel:
    """Returns a scripted hypothesis per pass, in order."""

    def __init__(self, passes):
        self.passes = list(passes)
        self.calls = 0
        self.buffer_sizes: list[int] = []

    def transcribe(self, audio, **_kw):
        self.buffer_sizes.append(len(audio))
        result = self.passes[min(self.calls, len(self.passes) - 1)]
        self.calls += 1
        return ([StubSegment([StubWord(*w) for w in result])], None)


def pcm(seconds):
    return b"\x01\x00" * int(seconds * SR)


def engine_for(passes, **kw):
    return WhisperStreamingEngine(StubModel(passes), sample_rate=SR, **kw)


def test_no_pass_between_intervals_is_the_blank():
    """push() returning None means the same thing it means for a transducer:
    audio was taken and there is nothing to say yet."""
    engine = engine_for([[("hello", 0.0, 0.4)]], interval=1.0)
    assert engine.push(pcm(0.5), 0.5) is None
    assert engine.push(pcm(0.5), 1.0) is not None


def test_a_word_needs_two_passes_to_agree_before_it_is_stable():
    """The whole point of LocalAgreement. One pass is a guess."""
    engine = engine_for(
        [
            [("the", 0.0, 0.3)],
            [("the", 0.0, 0.3), ("cat", 0.3, 0.6)],
        ],
        interval=1.0,
    )
    first = engine.push(pcm(1.0), 1.0)
    assert first.stable == ()
    assert [w.text for w in first.tentative] == ["the"]

    second = engine.push(pcm(1.0), 2.0)
    assert [w.text for w in second.stable] == ["the"]
    assert [w.text for w in second.tentative] == ["cat"]


def test_a_word_the_model_changes_its_mind_about_is_never_stable():
    """The failure this design exists to prevent: committing a guess."""
    engine = engine_for(
        [
            [("their", 0.0, 0.4)],
            [("there", 0.0, 0.4)],
            [("there", 0.0, 0.4), ("we", 0.4, 0.7)],
        ],
        interval=1.0,
    )
    engine.push(pcm(1.0), 1.0)
    second = engine.push(pcm(1.0), 2.0)
    assert second.stable == (), "a retracted word was promoted to stable"
    third = engine.push(pcm(1.0), 3.0)
    assert [w.text for w in third.stable] == ["there"]


def test_punctuation_and_casing_do_not_block_agreement():
    """Whisper revises both freely while the word stays put. Treating that as
    disagreement would stall the commit frontier forever."""
    engine = engine_for(
        [
            [("today", 0.0, 0.5)],
            [("Today,", 0.0, 0.5), ("we", 0.5, 0.8)],
        ],
        interval=1.0,
    )
    engine.push(pcm(1.0), 1.0)
    second = engine.push(pcm(1.0), 2.0)
    assert [w.text for w in second.stable] == ["Today,"], "the newer punctuation is kept"


def test_the_buffer_is_trimmed_at_the_agreement_point():
    """Without trimming, the cost of every pass grows with the session. That is
    the structural reason Whisper streaming degrades over minutes but looks fine
    in a ten second demo.
    """
    model = StubModel(
        [
            [("one", 0.0, 0.5)],
            [("one", 0.0, 0.5), ("two", 1.0, 1.5)],
            [("one", 0.0, 0.5), ("two", 1.0, 1.5), ("three", 2.0, 2.5)],
            [("one", 0.0, 0.5), ("two", 1.0, 1.5), ("three", 2.0, 2.5)],
        ]
    )
    engine = WhisperStreamingEngine(model, sample_rate=SR, interval=1.0, context=0.5)
    for i in range(1, 6):
        engine.push(pcm(1.0), float(i))
    assert model.buffer_sizes[-1] < model.buffer_sizes[1] + 3 * SR, (
        f"buffer grew unbounded: {model.buffer_sizes}"
    )


def test_the_buffer_is_capped_even_with_no_agreement():
    """A model that never agrees with itself must not grow the buffer forever."""
    model = StubModel([[("x", 0.0, 0.1)], [("y", 0.0, 0.1)]] * 20)
    engine = WhisperStreamingEngine(model, sample_rate=SR, interval=1.0, max_buffer=4.0)
    for i in range(1, 15):
        engine.push(pcm(1.0), float(i))
    assert max(model.buffer_sizes) <= 5 * SR


def test_flush_settles_whatever_is_left():
    """At end of stream there is no second pass to agree with."""
    engine = engine_for([[("all", 0.0, 0.4), ("done", 0.4, 0.8)]], interval=1.0)
    engine.push(pcm(1.0), 1.0)
    final = engine.flush()
    assert [w.text for w in final.stable] == ["all", "done"]
    assert final.tentative == ()


def test_a_word_never_claims_audio_that_has_not_arrived():
    """The Coordinator rejects that as an adapter bug, correctly."""
    engine = engine_for([[("late", 0.0, 5.0)]], interval=1.0)
    hypothesis = engine.push(pcm(1.0), 1.0)
    for word in hypothesis.all_words:
        assert word.audio_end <= 1.0 + 1e-6


def test_reset_clears_everything():
    engine = engine_for([[("a", 0.0, 0.2)], [("a", 0.0, 0.2)]], interval=1.0)
    engine.push(pcm(1.0), 1.0)
    engine.push(pcm(1.0), 2.0)
    engine.reset()
    assert engine._stable == []
    assert engine._buffer == []
    assert engine._settled_to == 0.0
