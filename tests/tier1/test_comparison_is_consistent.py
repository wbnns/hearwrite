"""The README and the page must name the same systems at the same prices.

`docs/index.html` is generated from `server/ui.html`, so `test_site_is_current`
already stops those two from disagreeing. Nothing does that for `README.md`,
which now repeats four per-hour prices and four system names by hand.

The failure this guards is quiet rather than loud. A vendor changes a price, one
copy is updated, and the repository goes on stating two different numbers for the
same thing in two places a reader is equally likely to land on. That is the same
class of mistake the site builder exists to prevent, so it is enforced here
rather than remembered.

MAI-Transcribe-2's $0.10 is a launch offer with no published standard rate, so
this test is expected to fail one day. When it does, update both files; do not
delete the assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
README = ROOT / "README.md"
PAGE = ROOT / "src" / "hearwrite" / "server" / "ui.html"

# A wheel install ships neither file, and skipping beats failing on a checkout
# that never had them.
pytestmark = pytest.mark.skipif(
    not README.exists() or not PAGE.exists(),
    reason="README or the source page is absent (installed, not a checkout)",
)

#: The hosted systems the two documents compare against, and what each charges
#: for an hour of audio. HearWrite is deliberately absent: it has no rate.
QUOTED = {
    "Muse Voice Transcribe": "$0.18",
    "GPT-Transcribe": "$0.27",
    "MAI-Transcribe-2": "$0.10",
}


@pytest.mark.parametrize(("system", "price"), sorted(QUOTED.items()))
def test_both_documents_name_the_system_and_its_price(system: str, price: str) -> None:
    for path in (README, PAGE):
        text = path.read_text()
        assert system in text, f"{path.name} does not name {system}"
        assert price in text, f"{path.name} does not quote {price} for {system}"


def test_the_page_prices_every_system_it_tabulates() -> None:
    """A row without a price is the one a reader would most want."""
    html = PAGE.read_text()
    table = html[html.index('<div class="cmp">') : html.index("</table></div>")]
    for system, price in QUOTED.items():
        row = table[table.index(system) :]
        assert price in row[: row.index("</tr>")], f"{system}'s row omits {price}"
