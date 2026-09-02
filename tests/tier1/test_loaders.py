"""Which models are shared, and which must not be.

The split is a correctness question, not only a memory one. Sharing a stateful
model between sessions would let one session's audio move another's decisions,
and that is the kind of bug that shows up as "diarization is flaky under load".
"""

from __future__ import annotations

from hearwrite import loaders


def test_the_shareable_models_are_cached():
    """Otherwise every session reloads about 300MB of weights."""
    for name in ("transducer", "speaker_embedder", "turn_session"):
        loader = getattr(loaders, name)
        assert hasattr(loader, "cache_info"), f"{name} is not cached"


def test_the_vad_is_deliberately_not_cached():
    """`VadModel` has a reset(), which is the tell that it carries state.

    Two sessions sharing one would contaminate each other's speech boundaries.
    It is 629KB, so a copy per session costs almost nothing.
    """
    assert not hasattr(loaders.vad_model, "cache_info"), "the VAD is being shared between sessions"


def test_loaded_reports_what_is_resident():
    loaders.clear()
    assert loaders.loaded() == {
        "transducer": 0,
        "speaker_embedder": 0,
        "turn_session": 0,
    }


def test_clear_releases_every_shared_model():
    """Deliberate release, for tests and for freeing memory on purpose."""
    loaders.clear()
    assert sum(loaders.loaded().values()) == 0


def test_caches_are_bounded():
    """A cache that grows with the number of distinct model names asked for
    would be a slow leak in a server that accepts a model per request."""
    for name in ("transducer", "speaker_embedder", "turn_session"):
        info = getattr(loaders, name).cache_info()
        assert info.maxsize is not None and info.maxsize <= 8, name


def test_the_default_recogniser_is_the_one_that_punctuates():
    """A transcript in block capitals with no punctuation reads as broken even
    when every word is right, and that is what people judge on. The default was
    changed on measurement: nemotron committed sooner AND produced readable
    text, at about five times the CPU.
    """
    from hearwrite.pipeline import DEFAULT_SHERPA_MODEL, Backends

    assert DEFAULT_SHERPA_MODEL == "nemotron-3.5-160ms"
    assert Backends().model is None, "the default must come from one place"


def test_the_lightweight_recogniser_is_still_available():
    """The default costs five times the CPU. A small VPS needs the other one."""
    from hearwrite.models import REGISTRY

    assert "zipformer-en" in REGISTRY
    assert "nemotron-3.5-160ms" in REGISTRY
