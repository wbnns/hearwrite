"""Inverse text normalisation: spoken numbers to written ones.

A recogniser transcribes what was said, so "two thousand nine" arrives as four
words. Readers want 2009. That conversion is inverse text normalisation, and it
is done here with rules rather than a model, for three reasons: it is
deterministic, it takes microseconds, and every decision it makes can be
explained and tested. A neural ITN would be none of those.

THE DANGER IS OVER EAGERNESS, and it is not hypothetical. A microphone test
recorded during development began "test one two three". Converting that to
"test 123" would be worse than leaving it alone, and a keen implementation does
exactly that. So the rules are deliberately narrow:

  * A run of number words is converted only when it reads as ONE value.
    "twenty five" is 25. "one two three" is three separate values, and separate
    values are left alone.
  * An ordinal becomes 3rd only next to a month, where it is a date.
  * Currency and percent attach to a number that is already being converted.

Anything not covered is passed through untouched. Under conversion is invisible;
over conversion corrupts the transcript.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "thirty first": 31,
}
MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
NUMBER_WORDS = set(UNITS) | set(TENS) | set(SCALES) | {"and"}


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _value(tokens: list[str]) -> int | None:
    """The single value a run of number words spells, or None if it spells several.

    "twenty five" is one value. "one two three" is three, and returning None for
    it is the whole point: a count-off must survive as words.
    """
    if not tokens or tokens[0] == "and":
        return None

    total = current = 0
    seen_scale = False
    previous: str | None = None

    for token in tokens:
        if token == "and":
            previous = token
            continue
        if token in UNITS:
            # Two bare units in a row are two numbers, not one: "one two".
            if previous in UNITS or (previous in TENS and UNITS[token] >= 10):
                return None
            if previous in UNITS:
                return None
            current += UNITS[token]
        elif token in TENS:
            if previous in UNITS or previous in TENS:
                return None
            current += TENS[token]
        elif token in SCALES:
            scale = SCALES[token]
            if current == 0 and not seen_scale:
                return None  # "hundred" with nothing before it
            if scale == 100:
                current *= 100
            else:
                total += (current or 1) * scale
                current = 0
            seen_scale = True
        else:
            return None
        previous = token

    return total + current


def _convert_numbers(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        bare = tokens[i].lower().strip(".,!?;:")
        if bare not in NUMBER_WORDS or bare == "and":
            out.append(tokens[i])
            i += 1
            continue

        j = i
        run: list[str] = []
        while j < len(tokens):
            candidate = tokens[j].lower().strip(".,!?;:")
            if candidate not in NUMBER_WORDS:
                break
            run.append(candidate)
            j += 1
        while run and run[-1] == "and":
            run.pop()
            j -= 1

        value = _value(run)
        if value is None or len(run) < 2:
            # One word, or several values. Leave every word exactly as it was.
            out.extend(tokens[i:j] if j > i else [tokens[i]])
            i = max(j, i + 1)
            continue

        trailing = tokens[j - 1][len(tokens[j - 1].rstrip(".,!?;:")) :]
        out.append(f"{value}{trailing}")
        i = j
    return " ".join(out)


def _convert_dates(text: str) -> str:
    """An ordinal beside a month is a date. Elsewhere it is a word."""
    tokens = text.split()
    out: list[str] = []
    for index, token in enumerate(tokens):
        bare = token.lower().strip(".,!?;:")
        previous = tokens[index - 1].lower().strip(".,!?;:") if index else ""
        following = tokens[index + 1].lower().strip(".,!?;:") if index + 1 < len(tokens) else ""
        if bare in ORDINALS and (previous in MONTHS or following in MONTHS):
            n = ORDINALS[bare]
            trailing = token[len(token.rstrip(".,!?;:")) :]
            out.append(f"{n}{_suffix(n)}{trailing}")
        else:
            out.append(token)
    return " ".join(out)


#: A unit that follows a number turns even a single spoken word into a figure.
#: "ten percent" is 10% and nobody writes it otherwise, whereas a bare "ten" in
#: running prose is usually better left as a word.
UNIT_FOLLOWERS = {"percent", "dollar", "dollars", "pound", "pounds", "euro", "euros"}


def _digits_before_units(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    for index, token in enumerate(tokens):
        bare = token.lower().strip(".,!?;:")
        following = tokens[index + 1].lower().strip(".,!?;:") if index + 1 < len(tokens) else ""
        if bare in UNITS and following in UNIT_FOLLOWERS:
            trailing = token[len(token.rstrip(".,!?;:")) :]
            out.append(f"{UNITS[bare]}{trailing}")
        else:
            out.append(token)
    return " ".join(out)


def _convert_units(text: str) -> str:
    """Attach currency and percent to a number already written as digits."""
    text = re.sub(r"\b(\d[\d,]*) percent\b", r"\1%", text)
    text = re.sub(r"\b(\d[\d,]*) dollars?\b", r"$\1", text)
    text = re.sub(r"\b(\d[\d,]*) pounds?\b", r"£\1", text)
    text = re.sub(r"\b(\d[\d,]*) euros?\b", r"€\1", text)
    return text


def normalise(text: str) -> str:
    """Rewrite spoken numbers as written ones, conservatively."""
    if not text.strip():
        return text
    return _convert_units(_digits_before_units(_convert_dates(_convert_numbers(text))))
