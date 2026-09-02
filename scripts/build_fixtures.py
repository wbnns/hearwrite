#!/usr/bin/env python3
"""Build multi speaker diarization fixtures from LibriSpeech dev-clean.

The corpus is not vendored: it is about 340MB and CC BY 4.0, so it is downloaded
on demand into the HearWrite cache. Conversations are assembled by concatenating
one utterance per speaker in rotation with a real pause between turns, and the
ground truth is written beside the audio as JSON.

Exits 0 on success, 1 if ffmpeg is missing or the corpus cannot be fetched.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
CACHE = Path.home() / ".cache" / "hearwrite" / "corpora"
ROOT = CACHE / "LibriSpeech" / "dev-clean"
OUT = CACHE / "fixtures"
SAMPLE_RATE = 16_000
GAP = 0.7
MAX_TURN = 10.0


def ensure_corpus() -> None:
    if ROOT.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    print(
        f"downloading LibriSpeech dev-clean (~340MB, CC BY 4.0) into {CACHE}",
        file=sys.stderr,
    )
    with urllib.request.urlopen(URL) as response:
        payload = io.BytesIO(response.read())
    with tarfile.open(fileobj=payload, mode="r:gz") as tar:
        tar.extractall(CACHE, filter="data")


def to_pcm(flac: Path) -> bytes:
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-i",
            str(flac),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "-",
        ],
        capture_output=True,
    ).stdout
    with wave.open(io.BytesIO(out)) as handle:
        return handle.readframes(handle.getnframes())


def build(count: int, turns_each: int = 3) -> Path:
    speakers = sorted(p.name for p in ROOT.iterdir() if p.is_dir())[:count]
    files = {s: sorted(ROOT.glob(f"{s}/*/*.flac")) for s in speakers}
    gap = b"\x00\x00" * int(GAP * SAMPLE_RATE)

    pcm, turns, at = b"", [], 0.0
    for index in range(turns_each):
        for speaker in speakers:
            data = to_pcm(files[speaker][index])
            seconds = len(data) / 2 / SAMPLE_RATE
            if seconds > MAX_TURN:
                data = data[: int(MAX_TURN * SAMPLE_RATE) * 2]
                seconds = MAX_TURN
            turns.append({"speaker": speaker, "start": round(at, 3), "end": round(at + seconds, 3)})
            pcm += data + gap
            at += seconds + GAP

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"conv_{count}spk.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm + b"\x00\x00" * int(1.5 * SAMPLE_RATE))
    path.with_suffix(".json").write_text(
        json.dumps({"speakers": speakers, "turns": turns}, indent=2)
    )
    return path


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg is required to decode LibriSpeech FLAC", file=sys.stderr)
        return 1
    ensure_corpus()
    for count in (2, 4, 8, 16, 24):
        print(f"{build(count)}  ({count} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
