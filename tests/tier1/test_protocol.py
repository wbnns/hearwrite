"""The wire protocol. Frozen at the end of Phase 0."""

from __future__ import annotations

import json

from hearwrite.events import Event, EventKind
from hearwrite.protocol import PROTOCOL_VERSION, decode, encode, hello_frame


def test_round_trip():
    event = Event(
        seq=7,
        kind=EventKind.COMMIT,
        at=1.25,
        payload={"text": "hello", "audio_start": 0.5, "audio_end": 1.0, "speaker": "A"},
    )
    back = decode(encode(event))
    assert back.seq == event.seq
    assert back.kind is EventKind.COMMIT
    assert back.at == event.at
    assert back.payload["text"] == "hello"


def test_null_speaker_survives_serialization():
    """`speaker: null` is a supported value, not an omission."""
    event = Event(seq=1, kind=EventKind.COMMIT, at=1.0, payload={"speaker": None})
    assert json.loads(encode(event))["payload"]["speaker"] is None
    assert decode(encode(event)).payload["speaker"] is None


def test_float_precision_is_pinned():
    """Two logically identical logs must serialize byte-identically.

    Without rounding at the serializer, a float differing in its last bit makes a
    byte comparison of two event logs fail for no meaningful reason -- and byte
    comparison is exactly how replay determinism is asserted.
    """
    a = Event(seq=1, kind=EventKind.COMMIT, at=0.1 + 0.2, payload={"audio_end": 0.1 + 0.2})
    b = Event(seq=1, kind=EventKind.COMMIT, at=0.3, payload={"audio_end": 0.3})
    assert encode(a) == encode(b)


def test_encoding_is_stable_across_key_order():
    a = Event(seq=1, kind=EventKind.COMMIT, at=1.0, payload={"x": 1, "y": 2})
    b = Event(seq=1, kind=EventKind.COMMIT, at=1.0, payload={"y": 2, "x": 1})
    assert encode(a) == encode(b)


def test_encoded_events_are_single_lines():
    """The transport is line-delimited JSON; an embedded newline would break it."""
    event = Event(seq=1, kind=EventKind.PARTIAL, at=1.0, payload={"text": "a b"})
    assert "\n" not in encode(event)


def test_hello_frame_announces_the_version():
    frame = json.loads(hello_frame(sample_rate=16000, policy="conversation"))
    assert frame["protocol"] == PROTOCOL_VERSION
    assert frame["sample_rate"] == 16000
    assert frame["policy"] == "conversation"
