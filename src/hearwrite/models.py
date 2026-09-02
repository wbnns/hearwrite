"""Model resolution: where weights come from, and what they cost you.

Two properties this module exists to protect, both of which are easy to lose by
accident:

  * NOTHING IS GATED. Every URL here is a plain public download. No Hugging Face
    acceptance, no token, no account. A gated weight would mean a secret in CI
    and a model nobody can vendor, which quietly breaks "clone it and run it".
  * EVERY MODEL HAS A LICENCE RECORDED NEXT TO IT. `hearwrite models` prints it.
    "Which licence did that checkpoint have" is exactly the question nobody
    wants to answer after shipping.

Downloads are verified against a pinned SHA-256 where one is known. A registry
entry without a hash still works, but says so out loud.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Override with HEARWRITE_CACHE. Defaults under the user cache directory.
DEFAULT_CACHE = Path.home() / ".cache" / "hearwrite" / "models"


class ModelError(RuntimeError):
    """A model could not be downloaded or verified. A runtime failure."""


class UnknownModel(ModelError):
    """The caller named a model that does not exist. A usage error.

    Kept separate so the CLI can honour its own exit code contract: 2 for a
    typo the user can fix by reading the message, 1 for a download or checksum
    that failed through no fault of theirs.
    """


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable bundle."""

    name: str
    url: str
    licence: str
    languages: str
    approx_mb: int
    summary: str
    #: SHA-256 of the downloaded artifact. None means unpinned, and the download
    #: says so rather than pretending it was verified.
    sha256: str | None = None
    #: True for a single file, False for a tarball that needs extracting.
    single_file: bool = False
    #: Globs, relative to the extracted directory, in preference order.
    encoder: tuple[str, ...] = ()
    decoder: tuple[str, ...] = ()
    joiner: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ("tokens.txt",)


REGISTRY: dict[str, ModelSpec] = {
    "zipformer-en": ModelSpec(
        name="zipformer-en",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
        ),
        licence="Apache-2.0",
        languages="en",
        approx_mb=296,
        summary="Streaming zipformer transducer, English. The default.",
        sha256="639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282",
        # int8 encoder and joiner keep it fast on CPU; the decoder is small
        # enough that the float build costs nothing and decodes a little better.
        encoder=("encoder-*chunk-16-left-128.int8.onnx", "encoder-*.int8.onnx"),
        decoder=("decoder-*chunk-16-left-128.onnx", "decoder-*.onnx"),
        joiner=("joiner-*chunk-16-left-128.int8.onnx", "joiner-*.int8.onnx"),
    ),
    "zipformer-en-small": ModelSpec(
        name="zipformer-en-small",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
        ),
        licence="Apache-2.0",
        languages="en",
        approx_mb=122,
        summary="20M parameter zipformer. Faster and much less accurate.",
        sha256="9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9",
        encoder=("encoder-*.int8.onnx",),
        decoder=("decoder-*.int8.onnx",),
        joiner=("joiner-*.int8.onnx",),
    ),
    "silero-vad": ModelSpec(
        name="silero-vad",
        url=("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"),
        licence="MIT",
        languages="any",
        approx_mb=1,
        summary="Silero VAD, ONNX build. The acoustic gate.",
        sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        single_file=True,
    ),
}


def cache_root() -> Path:
    import os

    override = os.environ.get("HEARWRITE_CACHE")
    return Path(override).expanduser() if override else DEFAULT_CACHE


def resolve(name_or_path: str, *, download: bool = True) -> Path:
    """Return a local path for a registry name, or pass a filesystem path through.

    A path that exists is used as-is, which is how you point HearWrite at a model
    it has never heard of without touching this file.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        return candidate

    spec = REGISTRY.get(name_or_path)
    if spec is None:
        raise UnknownModel(
            f"unknown model {name_or_path!r}, and no such path exists.\n"
            f"Known models: {', '.join(sorted(REGISTRY))}\n"
            f"Or pass a directory containing the ONNX files."
        )
    return ensure(spec, download=download)


def ensure(spec: ModelSpec, *, download: bool = True) -> Path:
    """Return the local path for a spec, downloading it if needed."""
    root = cache_root()
    target = root / (Path(spec.url).name if spec.single_file else spec.name)
    if target.exists():
        return target

    if not download:
        raise ModelError(f"{spec.name} is not downloaded and downloading is disabled")

    root.mkdir(parents=True, exist_ok=True)
    note = "" if spec.sha256 else "  (unpinned: no SHA-256 on record)"
    print(
        f"hearwrite: downloading {spec.name}, about {spec.approx_mb}MB, {spec.licence}{note}",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(dir=root) as tmp:
        blob = Path(tmp) / "download"
        _fetch(spec.url, blob)
        _verify(blob, spec)
        if spec.single_file:
            shutil.move(str(blob), target)
        else:
            _extract(blob, Path(tmp), target)
    return target


def _fetch(url: str, into: Path) -> None:
    try:
        with urllib.request.urlopen(url) as response, into.open("wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        raise ModelError(f"could not download {url}: {exc}") from exc


def _verify(blob: Path, spec: ModelSpec) -> None:
    if spec.sha256 is None:
        return
    digest = hashlib.sha256()
    with blob.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    got = digest.hexdigest()
    if got != spec.sha256:
        raise ModelError(
            f"{spec.name} failed checksum verification.\n"
            f"  expected {spec.sha256}\n  got      {got}\n"
            f"Refusing to use it."
        )


def _extract(blob: Path, workdir: Path, target: Path) -> None:
    staging = workdir / "staged"
    staging.mkdir()
    with tarfile.open(blob) as tar:
        # filter="data" refuses absolute paths, parent traversal and device
        # nodes. Extracting a downloaded archive without it is how a tarball
        # gets to write outside its directory.
        tar.extractall(staging, filter="data")
    roots = [p for p in staging.iterdir() if p.is_dir()]
    source = roots[0] if len(roots) == 1 else staging
    shutil.move(str(source), str(target))


def find(directory: Path, patterns: tuple[str, ...], *, what: str) -> Path:
    """First file matching any glob, in preference order."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise ModelError(f"no {what} found in {directory} (looked for {', '.join(patterns)})")
