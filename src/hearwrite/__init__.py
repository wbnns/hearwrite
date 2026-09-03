"""HearWrite -- real-time transcription, speaker labels and endpointing.

The public surface is deliberately small: build a Policy, build a Coordinator
with whichever model wrappers you have installed, push PCM, read events.
"""

from __future__ import annotations

from .clock import StreamClock
from .coordinator import (
    AGENT,
    CONVERSATION,
    DICTATION,
    Coordinator,
    EndpointMode,
    Policy,
    SpeakerMode,
    SpeakerPolicy,
    preset,
)
from .events import Event, EventKind, EventLog
from .protocol import PROTOCOL_VERSION, decode, encode
from .transcript import committed_text, display_text

__version__ = "0.1.0"

__all__ = [
    "AGENT",
    "CONVERSATION",
    "DICTATION",
    "PROTOCOL_VERSION",
    "Coordinator",
    "EndpointMode",
    "Event",
    "EventKind",
    "EventLog",
    "Policy",
    "SpeakerMode",
    "SpeakerPolicy",
    "StreamClock",
    "__version__",
    "committed_text",
    "decode",
    "display_text",
    "encode",
    "preset",
]
