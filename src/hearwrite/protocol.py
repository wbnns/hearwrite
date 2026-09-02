"""The wire protocol: JSON frames down, binary PCM up.

This module owns serialization and the version constant. It is deliberately
small and deliberately boring -- the schema freezes at the end of Phase 0, and
every later change to the pipeline has to fit through it unchanged.

Client contract:
  * `partial` may be revised or withdrawn at any time.
  * `commit` is final and is never contradicted by a later event.
  * A consumer that ignores `partial` entirely still gets a correct transcript.

That last property is what lets a simple integration stay simple.
"""

from __future__ import annotations

import json
from typing import Any

from .events import Event, EventKind

#: Bumped only for a breaking change to the frame shape. Consumers should refuse
#: a major version they do not recognise rather than guess.
PROTOCOL_VERSION = "1.0"


def encode(event: Event) -> str:
    """Serialize one event to a single JSON line."""
    return json.dumps(to_dict(event), separators=(",", ":"), sort_keys=True)


def to_dict(event: Event) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "kind": str(event.kind),
        "at": round(event.at, 6),
        "payload": _round_floats(event.payload),
    }


def decode(line: str) -> Event:
    """Parse one JSON line back into an Event."""
    raw = json.loads(line)
    return Event(
        seq=int(raw["seq"]),
        kind=EventKind(raw["kind"]),
        at=float(raw["at"]),
        payload=raw.get("payload", {}),
    )


def hello_frame(*, sample_rate: int, policy: str) -> str:
    """The first frame a server sends, so a client can check compatibility."""
    return json.dumps(
        {
            "kind": "hello",
            "protocol": PROTOCOL_VERSION,
            "sample_rate": sample_rate,
            "policy": policy,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _round_floats(payload: Any) -> Any:
    """Round floats to microseconds before serializing.

    Without this, a float that differs in its last bit between two runs makes a
    byte-comparison of two event logs fail for no meaningful reason. Replay
    determinism is asserted by diffing serialized logs, so the serializer has to
    be the place that pins precision.
    """
    if isinstance(payload, dict):
        return {k: _round_floats(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_round_floats(v) for v in payload]
    if isinstance(payload, float):
        return round(payload, 6)
    return payload
