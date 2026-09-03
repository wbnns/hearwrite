"""A scripted punctuator for Tier 1."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScriptedPunctuator:
    """Returns canned polish, or a naive one that capitalises and adds a stop."""

    by_text: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list, init=False)
    contexts: list[str] = field(default_factory=list, init=False)

    def polish(self, text: str, context: str = "") -> str:
        self.calls.append(text)
        self.contexts.append(context)
        if text in self.by_text:
            return self.by_text[text]
        lowered = text.lower()
        return lowered[:1].upper() + lowered[1:] + "." if lowered else lowered

    def reset(self) -> None:
        self.calls.clear()
        self.contexts.clear()
