"""The browser UI is served from the same port as the socket.

Both bugs guarded here were found by actually loading the page, not by reading
the code, and both returned a plausible looking HTTP response while being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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
