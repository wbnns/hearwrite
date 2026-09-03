"""Turning an event log into text a person reads.

There are two correct answers and they are different documents.

The COMMITTED transcript is the words, exactly as they were finalised, and it is
what the append only rule guarantees. Anything downstream that must not be
surprised should use it.

The DISPLAY transcript additionally applies `polished` events, which re-render a
finished utterance with punctuation and casing. A polish is verified to contain
the same words in the same order before it is ever emitted, so the two differ in
presentation and never in content.

A consumer that ignores `polished` entirely still gets a correct transcript.
That is the same promise `partial` carries, and it is the reason a second model
is allowed to touch committed text at all.
"""

from __future__ import annotations

from collections.abc import Iterable

from .events import Event, EventKind


def committed_text(events: Iterable[Event]) -> str:
    """The words as committed. Never revised, never re-rendered."""
    return " ".join(str(e.payload["text"]) for e in events if e.kind is EventKind.COMMIT)


def display_text(events: Iterable[Event]) -> str:
    """The words with any polish applied. Same content, better rendering."""
    events = list(events)
    polished: dict[int, tuple[int, str]] = {}
    for event in events:
        if event.kind is EventKind.POLISHED:
            polished[int(event.payload["from_seq"])] = (
                int(event.payload["to_seq"]),
                str(event.payload["text"]),
            )

    out: list[str] = []
    skip_until: int | None = None
    for event in events:
        if event.kind is not EventKind.COMMIT:
            continue
        if skip_until is not None and event.seq <= skip_until:
            continue
        skip_until = None
        if event.seq in polished:
            to_seq, text = polished[event.seq]
            out.append(text)
            skip_until = to_seq
            continue
        out.append(str(event.payload["text"]))
    return " ".join(out)
