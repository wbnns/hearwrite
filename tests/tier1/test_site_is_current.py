"""The published page must match the page the server serves.

`docs/index.html` is generated from `src/hearwrite/server/ui.html` by
`scripts/build_site.py`. Nothing about editing the UI forces a rebuild, so
without this test the published page drifts silently, and the copy that drifts is
the public one. Every number on that page is a claim about this software, so a
stale copy is worse than an ugly one.

If this fails, run:

    python3 scripts/build_site.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "build_site.py"
SOURCE = ROOT / "src" / "hearwrite" / "server" / "ui.html"
PUBLISHED = ROOT / "docs" / "index.html"
ICON = ROOT / "src" / "hearwrite" / "server" / "favicon.svg"
ICON_PUBLISHED = ROOT / "docs" / "favicon.svg"


def _builder():
    """Load the build script by path; scripts/ is deliberately not a package."""
    spec = importlib.util.spec_from_file_location("build_site", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A source checkout has all of these. A wheel has none, and there is nothing to
# check there because the wheel does not carry docs/.
pytestmark = pytest.mark.skipif(
    not (BUILDER.exists() and PUBLISHED.exists()),
    reason="not a source checkout",
)


def test_the_published_page_matches_the_source_page():
    expected = _builder().build(SOURCE.read_text())
    assert PUBLISHED.read_text() == expected, (
        "docs/index.html is stale. Run: python3 scripts/build_site.py"
    )


def test_the_published_icon_matches_the_served_icon():
    """One icon, two places it has to be. The server serves the source copy."""
    assert ICON_PUBLISHED.read_bytes() == ICON.read_bytes(), (
        "docs/favicon.svg is stale. Run: python3 scripts/build_site.py"
    )


def test_the_published_page_carries_no_capture_code():
    """The static page has no service behind it, so it must not try to reach one."""
    page = PUBLISHED.read_text()
    for leftover in ("WebSocket", "getUserMedia", 'id="mic"'):
        assert leftover not in page, f"{leftover} survived into the published page"
