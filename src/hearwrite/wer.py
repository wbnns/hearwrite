"""Word error rate, and the normalisation that makes it mean anything.

WER is a ratio of edit distance to reference length, which is the easy half. The
hard half is deciding what counts as a word, and getting that wrong moves the
number by more than most model changes do.

Two normalisations matter here and neither is optional:

  * CASE AND PUNCTUATION. LibriSpeech references are bare uppercase. A model
    that writes "The Times, January" is not wrong for doing so, and scoring it
    against "THE TIMES JANUARY" without normalising would charge it three
    errors for being better.

  * NUMBERS. HearWrite's inverse text normalisation turns "two thousand nine"
    into "2009" ON PURPOSE. A reference that spells it out would score that as
    one substitution and two deletions. So digits are spoken back out before
    comparison, and the feature that improves the transcript stops being
    punished by the metric.

Both directions are applied to hypothesis AND reference, so the comparison is
between what was said, not between two spelling conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w'\s]")
_SPACE = re.compile(r"\s+")

UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

#: Contractions LibriSpeech writes out and a modern model does not, or the
#: reverse. Comparing them as different words measures the era of the corpus.
EQUIVALENTS = {
    "mr": "mister",
    "mrs": "missus",
    "dr": "doctor",
    "st": "saint",
    "ok": "okay",
    "'em": "them",
}


def _say_number(n: int) -> str:
    if n < 20:
        return UNITS[n]
    if n < 100:
        return (TENS[n // 10] + (" " + UNITS[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        rest = f" {_say_number(n % 100)}" if n % 100 else ""
        return f"{UNITS[n // 100]} hundred{rest}"
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if n >= scale:
            rest = f" {_say_number(n % scale)}" if n % scale else ""
            return f"{_say_number(n // scale)} {name}{rest}"
    return str(n)


#: Ordinals that are not simply the cardinal plus a suffix.
ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    5: "fifth",
    8: "eighth",
    9: "ninth",
    12: "twelfth",
}


def _say_ordinal(n: int) -> str:
    if n in ORDINAL_WORDS:
        return ORDINAL_WORDS[n]
    cardinal = _say_number(n)
    head, _, tail = cardinal.rpartition(" ")
    last = ORDINAL_WORDS.get(n % 10) if n % 100 not in (11, 12, 13) else None
    if last is None:
        last = tail[:-1] + "ieth" if tail.endswith("y") else tail + "th"
    return f"{head} {last}".strip()


def _spell_numbers(text: str) -> str:
    """Say digits back out, so inverse text normalisation is not penalised."""

    def sub(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", "")
        ordinal = raw.endswith(("st", "nd", "rd", "th"))
        if ordinal:
            raw = raw[:-2]
        try:
            value = int(raw)
        except ValueError:
            return match.group(0)
        try:
            return _say_ordinal(value) if ordinal else _say_number(value)
        except (ValueError, IndexError):
            return match.group(0)

    text = text.replace("%", " percent").replace("$", " dollars ")
    return re.sub(r"\d[\d,]*(?:st|nd|rd|th)?", sub, text)


def normalise(text: str) -> list[str]:
    """The word sequence a WER comparison should actually run on."""
    text = _spell_numbers(text.lower())
    text = _PUNCT.sub(" ", text)
    words = [EQUIVALENTS.get(w, w) for w in _SPACE.sub(" ", text).strip().split()]
    return _drop_number_and(words)


#: Every word that can appear inside a spoken number, so an "and" between two of
#: them can be recognised as a convention rather than a word.
_NUMBER_WORDS = frozenset(
    UNITS
    + [t for t in TENS if t]
    + ["hundred", "thousand", "million", "billion"]
    + list(ORDINAL_WORDS.values())
)


def _drop_number_and(words: list[str]) -> list[str]:
    """Remove the "and" in "three hundred and forty two".

    LibriSpeech writes it and a modern model does not. Keeping it measures which
    side of the Atlantic the corpus came from rather than what was heard.
    """
    out: list[str] = []
    for index, word in enumerate(words):
        if (
            word == "and"
            and index
            and index + 1 < len(words)
            and words[index - 1] in _NUMBER_WORDS
            and words[index + 1] in _NUMBER_WORDS
        ):
            continue
        out.append(word)
    return out


@dataclass(frozen=True)
class WordErrors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def compare(reference: str, hypothesis: str) -> WordErrors:
    """Levenshtein alignment over normalised words."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1], distance[i - 1][j], distance[i][j - 1]
                )

    # Walk the alignment back to separate the three error kinds, because they
    # say different things: deletions mean audio was missed, insertions mean
    # words were invented.
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return WordErrors(subs, dels, ins, len(ref))


def aggregate(pairs: list[tuple[str, str]]) -> WordErrors:
    """Corpus WER: errors over the whole corpus, not the mean of per file rates.

    A short utterance with one error would otherwise weigh as much as a long one
    transcribed perfectly.
    """
    total = WordErrors(0, 0, 0, 0)
    for reference, hypothesis in pairs:
        e = compare(reference, hypothesis)
        total = WordErrors(
            total.substitutions + e.substitutions,
            total.deletions + e.deletions,
            total.insertions + e.insertions,
            total.reference_words + e.reference_words,
        )
    return total
