"""Render the page's raster images with a headless browser.

Two targets: docs/og.png, the link preview, and docs/apple-touch-icon.png. Both
are screenshots of real pages rather than hand drawn assets, so the preview
inherits the site's palette and font stack and the figures on it stay the same
figures the page publishes. Rendering at 2x and downsampling gives clean text at
the sizes the consumers actually want.

The favicon is docs/favicon.svg and is not built here. It is vector, so there is
nothing to rasterise; only Apple needs a PNG.

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
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCALE = 2
PORT = 9336

#: (source page, output file, width, height). Rendered in order.
TARGETS = (
    (ROOT / "scripts" / "og.html", ROOT / "docs" / "og.png", 1200, 630),
    (ROOT / "scripts" / "icon.html", ROOT / "docs" / "apple-touch-icon.png", 180, 180),
)


async def shoot(ws_url: str, page_url: str, width: int, height: int) -> bytes:
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
            {"width": width, "height": height, "deviceScaleFactor": SCALE, "mobile": False},
        )
        await cmd("Page.navigate", {"url": page_url})
        await asyncio.sleep(2.5)
        shot = await cmd(
            "Page.captureScreenshot",
            {
                "format": "png",
                "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": SCALE},
            },
        )
        return base64.b64decode(shot["data"])


def main() -> int:
    for source, _, _, _ in TARGETS:
        if not source.exists():
            raise SystemExit(f"missing {source}")
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            "--user-data-dir=/tmp/hearwrite-og",
            "--no-first-run",
            "--hide-scrollbars",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        ws_url = page["webSocketDebuggerUrl"]
        for source, out, width, height in TARGETS:
            png = asyncio.run(shoot(ws_url, source.as_uri(), width, height))
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(png)
            # Downsample the 2x render to the size consumers expect.
            subprocess.run(
                ["sips", "-z", str(height), str(width), str(out)],
                check=True,
                capture_output=True,
            )
            print(
                f"wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes) at {width}x{height}"
            )
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
