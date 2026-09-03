"""The browser UI is served from the same port as the socket.

Both bugs guarded here were found by actually loading the page, not by reading
the code, and both returned a plausible looking HTTP response while being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# The route builds a websockets Response, so these tests need the library even
# though they never open a socket. It is in the dev extra for that reason.
pytest.importorskip("websockets")

from hearwrite.server import app


@dataclass
class FakeRequest:
    path: str
    headers: dict


class FakeConnection:
    """Stands in for the websockets connection, which only respond() is used on."""

    def __init__(self):
        self.responded = None

    def respond(self, status, text):
        self.responded = (status, text)
        return ("respond", status, text)


def get(path, headers=None):
    return app._serve_ui(FakeConnection(), FakeRequest(path, headers or {}))


def test_the_page_is_served_with_a_body():
    """`connection.respond()` sets the body from its text argument and gives no
    way to replace it. Assigning `.body` to what it returns sends 200 and zero
    bytes, which looks fine in a log and renders a blank page.
    """
    response = get("/")
    assert response.status_code == 200
    assert response.body, "the page was served with an empty body"
    assert b"<!doctype html>" in response.body[:64].lower()
    assert response.headers["Content-Type"].startswith("text/html")
    assert int(response.headers["Content-Length"]) == len(response.body)


def test_a_query_string_still_reaches_the_page():
    """`request.path` carries the query, so "/?autostart=1" is not "/".
    Comparing them directly 404s every URL with a parameter.
    """
    response = get("/?autostart=1")
    assert response.status_code == 200
    assert response.body


@pytest.mark.parametrize("path", ["/", "/index.html", "/?x=1", "/index.html?y=2"])
def test_the_page_is_reachable_at_its_usual_paths(path):
    assert get(path).status_code == 200


def test_anything_else_is_a_404():
    assert get("/secrets")[1] == 404


def test_a_websocket_upgrade_is_passed_through():
    """Returning a response here would break the socket the page connects to."""
    assert get("/", {"Upgrade": "websocket"}) is None
    assert get("/", {"Upgrade": "WebSocket"}) is None


def test_the_page_ships_inside_the_package():
    """`hearwrite serve` must need nothing but the wheel."""
    assert app.UI.exists()
    assert app.UI.parent.name == "server"


def test_the_page_talks_to_its_own_origin():
    """A hardcoded host would break the moment anyone deploys it anywhere."""
    html = app.UI.read_text()
    assert "location.host" in html
    assert "localhost:8080" not in html


def test_the_page_asks_for_the_sample_rate_the_models_need():
    """Resampling in JavaScript is a good way to be quietly wrong, so the page
    asks the browser for 16kHz and refuses anything else."""
    html = app.UI.read_text()
    assert "sampleRate: 16000" in html
    assert "need 16000Hz" in html


def test_serve_defaults_to_the_policy_that_labels_speakers():
    """`conversation` is the project default; `serve` used to say `dictation`.

    A demo that silently runs solo shows no speaker labels at all, which is most
    of what there is to see.
    """
    from hearwrite.cli import build_parser

    assert build_parser().parse_args(["serve"]).policy == "conversation"


def test_the_current_word_is_marked_so_a_reader_can_follow_it():
    """A caption that does not say where you are is a wall of text."""
    html = app.UI.read_text()
    assert 'className = "live"' in html or '"live"' in html
    assert ".live {" in html


def test_the_waveform_is_drawn_from_the_audio_that_is_transcribed():
    """Not from a second analyser node. What you see should be what the
    recogniser is hearing, including the silences."""
    html = app.UI.read_text()
    assert "pushLevel(pcm)" in html
    assert "ws.send(pcm.buffer)" in html


def test_the_page_groups_words_by_the_turn_the_server_assigns():
    """Tracking a "current turn" on the client looks equivalent and is not: an
    event can arrive for a turn other than the newest, and a turn identified
    after the fact has to update the header it already rendered."""
    html = app.UI.read_text()
    assert "p.turn || lastTurn" in html
    assert "p.seq === undefined" in html, "no handler for a turn level speaker event"


def test_the_page_treats_an_endpoint_as_a_break_not_a_new_block():
    """An endpoint ends an utterance, not a turn. Rendering it as a new block
    chops one person talking continuously into pieces."""
    html = app.UI.read_text()
    assert 'classList.contains("brk")' in html, "consecutive endpoints not collapsed"
    assert "said.lastElementChild" in html


# -- the page around the demo ------------------------------------------------


def test_the_page_publishes_its_failure_case_next_to_its_numbers():
    """A benchmark without its failure case is marketing.

    The README once led with "exact at 2, 4, 8, 16 and 24 speakers" and buried
    the caveat, and a reader pointed it at three people in a room and got two.
    The page must not repeat that.
    """
    html = app.UI.read_text()
    assert "What does not work" in html
    assert "one microphone" in html
    assert "0.54" in html and "0.55" in html, "the measurement behind the claim is missing"


def test_the_page_does_not_claim_a_place_on_leaderboards_it_cannot_run():
    """The AA-WER index corpus is not reproducible and AMI has not been run."""
    html = app.UI.read_text()
    assert "cannot be placed on the published leaderboards" in html
    assert "AMI and VoxConverse" in html


def test_every_chart_has_a_table_view():
    """Colour-only encoding is not an accessible way to read a value."""
    html = app.UI.read_text()
    assert html.count("<details>") >= 2
    assert html.count("Table view") >= 2


def test_chart_colours_come_from_the_validated_palette():
    """Slots 1 and 2 of the reference categorical palette, stepped per mode.
    Validated on both surfaces: worst all-pairs CVD delta E 26.8 dark, 24.7
    light, against a floor of 8.
    """
    html = app.UI.read_text()
    for hex_value in ("#3987e5", "#d95926", "#2a78d6", "#eb6834"):
        assert hex_value in html, f"{hex_value} missing; palette was changed unvalidated"


def test_marks_carry_a_hit_target_bigger_than_themselves():
    """A 10px dot you must land on dead centre is not a hover affordance."""
    html = app.UI.read_text()
    assert 'r="12" fill="transparent"' in html


def test_dark_mode_is_selected_not_flipped():
    """Its own steps from the same ramps, chosen for the dark surface."""
    html = app.UI.read_text()
    assert "prefers-color-scheme: light" in html
    assert "--series-1:#3987e5" in html and "--series-1:#2a78d6" in html
