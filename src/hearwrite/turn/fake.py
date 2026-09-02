"""A scripted turn detector for Tier 1.

The default is the honest one for a fake: a sentence that ends in terminal
punctuation reads as complete, anything else does not. That is enough to test the
conjunctive gate without pretending to model language.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScriptedTurnDetector:
    """Returns a fixed score, or one looked up by the committed text."""

    fixed: float | None = None
    by_text: dict[str, float] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list, init=False)
    audio_lengths: list[int] = field(default_factory=list, init=False)

    def completeness(self, text: str, pcm: bytes | None = None) -> float:
        self.calls.append(text)
        self.audio_lengths.append(len(pcm) if pcm else 0)
        if self.fixed is not None:
            return self.fixed
        if text in self.by_text:
            return self.by_text[text]
        return 1.0 if text.rstrip().endswith((".", "?", "!")) else 0.0

    def reset(self) -> None:
        self.calls.clear()
        self.audio_lengths.clear()
