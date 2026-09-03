"""Build the published page in docs/ from the page the server actually serves.

The site and the live UI say the same things about the same numbers, so keeping
two copies by hand guarantees they drift, and the copy that drifts is the public
one. There is one source, `server/ui.html`, and this script derives the static
page from it: the live demo card becomes the recording, and the capture script
is removed rather than left to fail on a page with no service behind it.

Run it after any change to the UI or the measurements:

    python3 scripts/build_site.py

Exit code 0 on success. Non-zero if the source page no longer has the shape this
script expects, which means the transform needs updating rather than skipping.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "hearwrite" / "server" / "ui.html"
OUT = ROOT / "docs" / "index.html"
#: One icon, kept with the server that serves it, copied here for the CDN.
ICON = ROOT / "src" / "hearwrite" / "server" / "favicon.svg"
ICON_OUT = ROOT / "docs" / "favicon.svg"

# The recording replaces the interactive card. `controls` and no autoplay are
# deliberate: the point of the clip is the relationship between what was said
# and what was transcribed, and a viewer cannot judge that with the sound
# muted, which is the only way a browser will start a video by itself.
VIDEO = """  <figure class="demo">
    <div class="player" id="player">
      <video id="clip" controls playsinline preload="metadata" poster="demo.jpg"
             width="1846" height="796" aria-describedby="democap">
        <source src="demo.mp4" type="video/mp4">
        <p>Your browser cannot play the recording.
          <a href="demo.mp4">Download it instead.</a></p>
      </video>
      <button class="play" id="play" type="button" aria-controls="clip"
              aria-label="Play the recording. 8 seconds, with sound.">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7z"/></svg>
      </button>
    </div>
    <figcaption id="democap"><b>A recording, not a live service.</b> Nine words
      through the default pipeline on an Apple M4, CPU only, with diarization and
      the semantic gate on. Dim words are provisional and can still change; solid
      words are committed and never will. The speaker label arrives a beat late,
      once there is enough voice to identify one. Delay p50 settles at 0.43s.
      <a href="https://github.com/wbnns/hearwrite">Run it yourself</a> for a live
      one. It is one command, and it needs no GPU and no API key.</figcaption>
  </figure>
"""


# No border and no radius on the video: the recording already contains the
# card's own rounded border, and adding a second one draws a box inside a box.
STYLE = """  .demo { margin:0 0 34px; }
  .player { position:relative; }
  .demo video { width:100%; height:auto; display:block; background:var(--bg); }
  .demo figcaption { margin-top:14px; color:var(--text-secondary); font-size:13px;
                     line-height:1.65; max-width:72ch; }
  /* Hidden until the script claims it, so a page with no JavaScript shows the
     browser's own controls rather than a button that does nothing. */
  .play { display:none; }
  .player[data-ready] .play {
    display:grid; place-items:center; position:absolute;
    /* Sits in the empty band below the transcript rather than centred on the
       frame, because the transcript is the thing worth seeing on the poster. */
    left:50%; top:63%; transform:translate(-50%,-50%);
    width:74px; height:74px; padding:0; border:1px solid rgba(242,242,240,0.22);
    border-radius:999px; background:rgba(11,11,13,0.66); color:#f2f2f0;
    cursor:pointer; transition:transform 140ms ease, background 140ms ease;
  }
  .player[data-ready] .play svg { width:32px; height:32px; fill:currentColor;
                                  margin-left:3px; }
  .player[data-ready] .play:hover { transform:translate(-50%,-50%) scale(1.07);
                                    background:rgba(11,11,13,0.82); }
  .play:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
  .player[data-playing] .play { opacity:0; pointer-events:none; }
  @media (prefers-reduced-motion:reduce) {
    .player[data-ready] .play { transition:none; }
    .player[data-ready] .play:hover { transform:translate(-50%,-50%); }
  }
"""


SCRIPT = """<script>
(() => {
  const box = document.getElementById("player");
  const clip = document.getElementById("clip");
  const play = document.getElementById("play");
  if (!box || !clip || !play) return;
  box.dataset.ready = "1";
  play.addEventListener("click", () => clip.play());
  const sync = () => {
    if (clip.paused) delete box.dataset.playing;
    else box.dataset.playing = "1";
  };
  ["play", "pause", "ended"].forEach(e => clip.addEventListener(e, sync));
  sync();
})();
</script>
"""


# Analytics, and this one matters: the published page and the page a user gets
# from `pip install hearwrite && hearwrite serve` are built from the same source.
# A tag in that source would send every self hosted run to this property without
# the operator ever agreeing to it. It goes in at publish time only.
ANALYTICS = """<script async src="https://www.googletagmanager.com/gtag/js?id=G-4GCPDZ8NX8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-4GCPDZ8NX8');
</script>
"""


# Link preview tags, on the published copy only. They carry absolute URLs, and
# the server serving this page on localhost has no business advertising them.
# og:image is scripts/og.html rendered by scripts/build_og.py, so the numbers on
# the image are the numbers on the page.
SITE = "https://hearwrite.wbnns.com/"
OG = """<meta property="og:type" content="website">
<meta property="og:site_name" content="HearWrite">
<meta property="og:url" content="{site}">
<meta property="og:title" content="HearWrite: streaming ASR, diarization and endpointing on CPU">
<meta property="og:description" content="Real time speech to text with speaker labels and semantic endpointing, assembled from open weight models. Runs on a laptop CPU with no GPU and no API key.">
<meta property="og:image" content="{site}og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="HearWrite. Streaming ASR, diarization and endpointing on CPU. A transcript line labelled Speaker A, with measured figures: 0.28s median emission delay, 4.4% word error rate, 0.05x real time factor, 340MB memory.">
<meta name="twitter:card" content="summary_large_image">
<link rel="apple-touch-icon" href="{site}apple-touch-icon.png">
<link rel="canonical" href="{site}">
""".format(site=SITE)


# The live page can claim its own numbers because the machine serving it is the
# machine that produced them. A static page on someone else's CDN cannot, so the
# claim has to name the machine instead of pointing at itself.
OLD_SUB = """    <p class="sub">Everything below is measured on the machine serving this page,
      and every number is reproducible with a command in the repository.</p>"""
NEW_SUB = """    <p class="sub">The demo above is a recording. Every number below was measured
      on an Apple M4 with no accelerator, and each one is reproducible with a
      command in the repository.</p>"""


def build(source: str) -> str:
    """Derive the static page, or raise if the source no longer matches."""
    steps = 0

    # 1. The capture script is dead weight on a page with no service behind it,
    #    and worse than dead: it would try to open a WebSocket to the CDN.
    page, n = re.subn(r"\n<script>.*?</script>\n", "\n", source, flags=re.DOTALL)
    if n != 1:
        raise SystemExit(f"expected exactly one script block, found {n}")
    steps += 1

    # 2. The interactive card becomes the recording.
    page, n = re.subn(
        r'  <div class="card">\n.*?\n  </div>\n\n(?=  <section>)',
        VIDEO + "\n",
        page,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise SystemExit("could not find the demo card to replace")
    steps += 1

    # 3. Restate the provenance claim, which is no longer true of this page.
    if OLD_SUB not in page:
        raise SystemExit("the hero provenance line has changed; update NEW_SUB")
    page = page.replace(OLD_SUB, NEW_SUB)
    steps += 1

    # 4. Link preview tags, immediately after the description they extend.
    desc_end = 'no API key.">\n'
    if desc_end not in page:
        raise SystemExit("could not find the description tag to anchor OG tags to")
    page = page.replace(desc_end, desc_end + OG, 1)
    steps += 1

    # 5. Analytics, immediately after the preview tags.
    page = page.replace(OG, OG + ANALYTICS, 1)
    if "googletagmanager" not in page:
        raise SystemExit("analytics tag did not land")
    steps += 1

    # 6. Styles for the figure, kept beside the source's own rules.
    marker = "  .tiles {"
    if marker not in page:
        raise SystemExit("could not find an anchor for the demo styles")
    page = page.replace(marker, STYLE + marker, 1)
    steps += 1

    # 7. The overlay play button's script, at the end of the document.
    if not page.rstrip().endswith("</div>"):
        raise SystemExit("unexpected end of document; cannot append the script")
    page = page.rstrip() + "\n" + SCRIPT
    steps += 1

    banner = (
        "<!-- Generated by scripts/build_site.py from src/hearwrite/server/ui.html.\n"
        "     Do not edit this file. Edit the source page and rebuild. -->\n"
    )
    page = page.replace("<!doctype html>\n", "<!doctype html>\n" + banner, 1)

    if 'id="mic"' in page or "WebSocket" in page or "getUserMedia" in page:
        raise SystemExit("interactive leftovers survived the transform")
    if steps != 7:
        raise SystemExit(f"expected 7 transforms, ran {steps}")
    return page


def main() -> int:
    page = build(SOURCE.read_text())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    ICON_OUT.write_bytes(ICON.read_bytes())
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes) from {SOURCE.name}")
    print(f"copied {ICON_OUT.relative_to(ROOT)} from {ICON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
