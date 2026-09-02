"""The stream clock. Everything downstream inherits its correctness."""

from __future__ import annotations

import pytest

from hearwrite.clock import StreamClock, duration_of


def test_position_tracks_pushed_audio():
    clock = StreamClock(16_000)
    assert clock.position == 0.0
    clock.advance_bytes(32_000)  # 1 second of 16-bit mono
    assert clock.position == 1.0


def test_partial_sample_is_rejected():
    """A caller splitting a sample across pushes has a framing bug.

    Accepting it would show up much later as a slow timestamp drift, which is far
    harder to trace than a loud failure here.
    """
    clock = StreamClock(16_000)
    with pytest.raises(ValueError, match="whole number"):
        clock.advance_bytes(1)


def test_negative_advance_is_rejected():
    clock = StreamClock(16_000)
    with pytest.raises(ValueError):
        clock.advance_bytes(-2)


def test_many_small_pushes_do_not_drift():
    """Sample counting, not float accumulation. 20ms frames for an hour."""
    clock = StreamClock(16_000)
    frame = 320 * 2
    for _ in range(180_000):
        clock.advance_bytes(frame)
    assert clock.position == 3600.0
    assert clock.samples == 16_000 * 3600


def test_duration_of_matches_the_clock():
    pcm = b"\x00\x00" * 8_000
    assert duration_of(pcm, 16_000) == 0.5
