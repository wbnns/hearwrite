"""The speaker frontend's segmentation logic, with no models.

The embedding model is stubbed, so what is under test is the part that was
actually wrong the first time: which spans of audio get embedded, and when.
"""

from __future__ import annotations

from itertools import pairwise

from hearwrite.speakers.sherpa import SherpaSpeakerFrontend
from hearwrite.vad.fake import ScriptedVAD

SR = 16_000


class StubExtractor:
    """Returns a vector derived from the audio, so identical audio matches."""

    def __init__(self):
        self.calls: list[int] = []

    def create_stream(self):
        return self

    def accept_waveform(self, _rate, samples):
        self._samples = samples

    def input_finished(self):
        pass

    def compute(self, _stream):
        self.calls.append(len(self._samples))
        total = sum(self._samples)
        return [total, len(self._samples) / SR, 1.0]


def pcm(seconds):
    return b"\x01\x00" * int(seconds * SR)


def frontend(speech, **kw):
    return SherpaSpeakerFrontend(StubExtractor(), ScriptedVAD(speech=speech), sample_rate=SR, **kw)


def drive(fe, seconds, chunk=0.02):
    at = 0.0
    out = []
    for _ in range(round(seconds / chunk)):
        at = round(at + chunk, 6)
        out.extend(fe.push(pcm(chunk), at))
    out.extend(fe.flush())
    return out


def test_windows_are_emitted_during_speech_not_after_it():
    """Waiting for a region to end would delay every label by the turn length.

    A ten second turn must produce its first segment about two seconds in, not
    ten. Turn label latency is a metric this project reports.
    """
    fe = frontend(((0.0, 10.0),))
    at = 0.0
    first_at = None
    for _ in range(500):
        at = round(at + 0.02, 6)
        if fe.push(pcm(0.02), at) and first_at is None:
            first_at = at
    assert first_at is not None
    assert first_at < 3.0, f"first segment took {first_at}s of a 10s turn"


def test_a_long_region_is_tiled_by_windows():
    fe = frontend(((0.0, 8.0),))
    segments = drive(fe, 9.0)
    assert len(segments) >= 3
    for earlier, later in pairwise(segments):
        assert later.start >= earlier.end - 1e-6, "windows overlap or go backwards"


def test_windows_are_a_fixed_length():
    """The threshold that compares embeddings is a fixed number, so the windows
    it compares have to be a fixed length. See the module docstring."""
    fe = frontend(((0.0, 8.0),))
    full = [s for s in drive(fe, 9.0) if s.duration > 1.9]
    assert full, "no full windows were produced"
    for segment in full:
        assert abs(segment.duration - 2.0) < 1e-6


def test_a_short_pause_does_not_split_a_region():
    """Ordinary speech pauses at clause boundaries. Treating each as a speaker
    boundary fragments the audio into pieces too short to embed, which is what
    produced 87% unlabelled words before this existed.
    """
    together = frontend(((0.0, 3.0), (3.2, 6.0)), min_silence=0.35)
    spanning = [s for s in drive(together, 7.0) if s.start < 3.2 < s.end]
    assert spanning, "a 0.2s pause split the region"


def test_a_long_pause_does_split_a_region():
    fe = frontend(((0.0, 3.0), (5.0, 8.0)), min_silence=0.35)
    segments = drive(fe, 9.0)
    assert not [s for s in segments if s.start < 4.0 < s.end]


def test_a_tail_below_the_minimum_is_not_embedded():
    """A short window looks unlike itself, so embedding it invites a split."""
    extractor = StubExtractor()
    fe = SherpaSpeakerFrontend(
        extractor, ScriptedVAD(speech=((0.0, 2.5),)), sample_rate=SR, min_region=1.5
    )
    drive(fe, 3.5)
    # One full 2s window; the 0.5s remainder is dropped.
    assert extractor.calls == [2 * SR]


def test_a_tail_above_the_minimum_is_embedded():
    extractor = StubExtractor()
    fe = SherpaSpeakerFrontend(
        extractor, ScriptedVAD(speech=((0.0, 3.8),)), sample_rate=SR, min_region=1.5
    )
    drive(fe, 4.5)
    assert len(extractor.calls) == 2
    assert extractor.calls[0] == 2 * SR
    assert extractor.calls[1] >= int(1.5 * SR)


def test_silence_alone_produces_nothing():
    fe = frontend(())
    assert drive(fe, 3.0) == []


def test_a_region_is_capped_so_memory_stays_bounded():
    """Someone who talks for a minute without pausing must not grow a buffer."""
    fe = frontend(((0.0, 60.0),), max_region=6.0)
    drive(fe, 20.0)
    assert len(fe._samples) <= int(6.0 * SR)


def test_segment_timestamps_are_stream_relative():
    fe = frontend(((4.0, 9.0),))
    segments = drive(fe, 10.0)
    assert segments
    assert segments[0].start >= 3.9, "a segment claimed audio before the speech began"
