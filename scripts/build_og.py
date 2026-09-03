"""Render scripts/og.html to docs/og.png, the link preview image.

The preview is a screenshot of a real page rather than a hand drawn asset, so it
inherits the site's palette and font stack and the numbers on it stay the same
numbers the page publishes. Rendering at 2x and downsampling gives clean text at
the 1200x630 that the OpenGraph consumers actually want.

Needs Chrome and the `websockets` package, both of which are already required to
verify the UI, and neither of which the package depends on at runtime. This is
not part of `bin/check`: it needs a browser, and CI has no reason to redraw an
image that only changes when the copy does.

    python3 scripts/build_og.py

Exit code 0 on success, non-zero if Chrome could not render the page.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts" / "og.html"
OUT = ROOT / "docs" / "og.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH, HEIGHT, SCALE = 1200, 630, 2
PORT = 9336


async def shoot(ws_url: str, page_url: str) -> bytes:
    import websockets

    async with websockets.connect(ws_url, max_size=None) as ws:
        counter = 0

        async def cmd(method: str, params: dict | None = None) -> dict:
            nonlocal counter
            counter += 1
            mine = counter
            await ws.send(json.dumps({"id": mine, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mine:
                    if "error" in msg:
                        raise SystemExit(f"chrome: {msg['error']}")
                    return msg.get("result", {})

        await cmd("Page.enable")
        # An explicit viewport, because the OpenGraph size is the whole point and
        # a window that does not match would letterbox or crop the layout.
        await cmd(
            "Emulation.setDeviceMetricsOverride",
            {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": SCALE, "mobile": False},
        )
        await cmd("Page.navigate", {"url": page_url})
        await asyncio.sleep(2.5)
        shot = await cmd(
            "Page.captureScreenshot",
            {"format": "png", "clip": {"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT, "scale": SCALE}},
        )
        return base64.b64decode(shot["data"])


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")
    proc = subprocess.Popen(
        [
            CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
            "--user-data-dir=/tmp/hearwrite-og", "--no-first-run",
            "--hide-scrollbars", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        tabs = None
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:
                time.sleep(0.5)
        if not tabs:
            raise SystemExit("chrome did not start")
        page = next(t for t in tabs if t["type"] == "page")
        png = asyncio.run(shoot(page["webSocketDebuggerUrl"], SOURCE.as_uri()))
    finally:
        proc.terminate()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(png)
    # Downsample the 2x render to the size consumers expect.
    subprocess.run(
        ["sips", "-z", str(HEIGHT), str(WIDTH), str(OUT)],
        check=True, capture_output=True,
    )
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes) at {WIDTH}x{HEIGHT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
