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
import re
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


def test_the_link_preview_shows_the_same_figures_as_the_page():
    """The shared image is the copy most people see, and it drifted once already.

    `4560b90` corrected the four headline figures because, read as a row, they
    described a configuration that does not exist: three for the default
    recogniser and one for the light one. It fixed the page and moved the image's
    memory figure to a different wrong value, so the link preview went on
    promising 442MB beside a 6.8% word error rate that the light recogniser does
    not have. Nothing caught that, because nothing compared the two.

    If this fails, edit `scripts/og.html` to match the page, then run:

        python3 scripts/build_og.py
    """
    og = ROOT / "scripts" / "og.html"
    if not og.exists():
        pytest.skip("scripts/og.html is absent")
    figures = re.compile(r'<div class="n">(.*?)</div><div class="l">(.*?)</div>')
    on_the_page = set(figures.findall(SOURCE.read_text()))
    on_the_image = set(figures.findall(og.read_text()))
    assert on_the_page, "no headline tiles found on the page"
    assert on_the_image == on_the_page, (
        "scripts/og.html and the page disagree. Only on the image: "
        f"{sorted(on_the_image - on_the_page)}. Only on the page: "
        f"{sorted(on_the_page - on_the_image)}."
    )
