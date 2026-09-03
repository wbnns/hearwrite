# Running it without a GPU

HearWrite is built to run on a laptop or a small VPS. There is no CUDA path, no
torch in the default install, and nothing to accept before the models download.
This page is what it actually costs.

## What it needs

Measured, not estimated. All four models loaded: streaming ASR, speaker
embeddings, VAD and the turn detector.

| | |
|---|---|
| Disk, Python packages | 172MB (9 packages) |
| Disk, model weights | **653MB** for the default recogniser, plus 87MB shared |
| Memory, default recogniser | **2010MB** for one session, **2015MB** for five |
| Memory, `--model zipformer-en` | **442MB** for one session, **571MB** for five |
| CPU, one stream | real time factor **0.25** (default), **0.055** (`--model zipformer-en`) |

Read the two memory rows before choosing a box, because they are eight times
apart and the default is the expensive one. The default recogniser is a 653MB
int8 encoder, and it dominates everything else: adding four more sessions costs
5MB, because the model is loaded once and shared. The light recogniser starts at
442MB and grows about 32MB per session, so it is the sessions that cost rather
than the model.

Sharing is what makes either number possible. Every session used to load its own
copy of the recognizer, so four sessions wanted four copies. The recognizer, the
speaker embedder and the turn detector hold no per-stream state, so they load
once; only the VAD is per-session, because it carries state, and it is 629KB.

## Sizing

**Which recogniser you pick decides the box.**

| Recogniser | Memory | Smallest sane box |
|---|---|---|
| `nemotron-3.5-160ms` (default) | 2010MB, flat in sessions | **4GB** |
| `zipformer-en` | 442MB + ~32MB per session | **1GB** |

A 1GB VPS runs this only with `--model zipformer-en`. The default recogniser
will be killed on that box, and it is the default because it punctuates and
commits sooner, not because it is small.

Memory is the binding constraint, not CPU. At the measured rates a machine runs
out of RAM long before it runs out of cycles, so size from the table, then set
the admission limit from what you measured.

**The CPU numbers above are from an Apple M4.** A typical shared-vCPU VPS is
several times slower per core. Even at five times slower, one stream sits near
0.25 real time and eight near 0.7, so the box keeps up. Measure it yourself
rather than trusting that arithmetic:

```sh
hearwrite transcribe your-audio.wav --policy conversation   # prints rtf
```

If the real time factor approaches 1.0 you are at capacity for that many
concurrent streams, and the fix is fewer sessions per box rather than a bigger
model.

## Admission control

The service refuses connections past a limit rather than accepting them and
being slow for everyone, because latency degradation under contention is the
failure people notice first. The default is the core count, capped at 16, which
is conservative against the measurements above. Raise it deliberately:

```sh
hearwrite serve --max-sessions 24
```

## The browser UI

`hearwrite serve` also serves a single page at `/` on the same port: microphone
capture, a live transcript, speaker colours and endpoint markers. `--open`
launches it.

It is one HTML file inside the package, with no build step and no CDN. The page
captures at 16kHz through an AudioWorklet and sends raw PCM over the same
WebSocket any other client would use, so the demo and a real integration are the
same code path rather than two things that can drift.

Two notes for deploying it beyond localhost. Browsers only grant microphone
access on a **secure context**, which means `localhost` works but a bare IP over
plain HTTP does not, so put TLS in front. And the page connects back to
`location.host`, so it follows wherever you serve it without configuration.

If you do not want it exposed, it is a static page with no privileged access;
blocking `GET /` at your proxy removes it and leaves the socket working.

## Running it

### Docker

```sh
docker build -t hearwrite .
docker run -p 8080:8080 hearwrite
```

Weights are baked in at build time and pruned, so a container starts in about a
second instead of fetching 600MB on its first request. Each is checked against
its pinned SHA-256 during the build, so a corrupted or substituted file fails
the build rather than the first request.

**Every command in the Dockerfile has been run and verified individually; a full
`docker build` has not.** It was written on a machine without Docker. If the
build fails, that is why, and a patch is welcome. To trade image size back
for a smaller build, drop the download layer and mount a volume at
`/root/.cache/hearwrite`.

### systemd

```ini
[Unit]
Description=HearWrite
After=network-online.target

[Service]
ExecStart=/opt/hearwrite/.venv/bin/hearwrite serve --host 127.0.0.1 --port 8080
Restart=on-failure
User=hearwrite
Environment=HEARWRITE_CACHE=/var/lib/hearwrite/models
# Sized from the table above: shared models plus per session, plus headroom.
MemoryMax=1G

[Install]
WantedBy=multi-user.target
```

Run `hearwrite models` once as that user before starting the service, so the
first connection does not wait for a download.

## Where to put it

**Not behind your application server.** HearWrite speaks WebSocket and expects
the client to connect to it directly, with a short lived token your application
issues. Proxying audio frames through a request/response app server is a job it
is bad at, and it means the inference service cannot be restarted independently.

**Behind TLS.** HearWrite speaks plain WebSocket and does not manage
certificates. Terminate TLS in front of it.

**With authentication in front.** There is deliberately none built in. See
[SECURITY.md](../SECURITY.md).

## Choosing a recogniser

This is the one decision that changes the resource picture, so it is worth
making deliberately. Measured on the same recording, on Apple silicon:

| Model | Real time factor | Download | Output |
|---|---|---|---|
| `nemotron-3.5-160ms` (default) | 0.26 | 453MB | `Test one two three Charlie's running up the stairs.` |
| `zipformer-en` | 0.047 | 310MB | `TEST ONE TWO THREE CHARLEY'S RUNNING UP THE STAIRS` |

Nemotron also commits SOONER, at p50 0.36s against 0.52s, so this is not the
usual accuracy for latency trade. It is accuracy and latency for CPU.

The default is right for a laptop and for any box with cycles to spare. It is
wrong for a single shared vCPU: five times 0.05 is 0.26 here, and a VPS core
several times slower than this one puts a single stream near real time, with
nothing left for a second. **Measure before you assume**, then pick:

```sh
hearwrite transcribe sample.wav                        # the default
hearwrite transcribe sample.wav --model zipformer-en   # the cheap one
```

## Making it smaller

* `hearwrite models --prune` deletes the float builds of every model, which are
  downloaded but never loaded. How much that reclaims depends on which models you
  have: it is 30MB against the default set, where only the punctuation model
  ships a float build, and several hundred against a cache that has collected the
  zipformers as well. It prints what it freed.
* **`--model zipformer-en` is the big lever**, not pruning. It takes the process
  from 2010MB to 442MB, which is the difference between a 4GB box and a 1GB one.
* `--policy dictation` skips the speaker frontend entirely, which drops two
  models from the hot path. If you do not need speaker labels, do not pay for
  them.
* `--no-turn` drops semantic endpointing. Endpointing then reduces to a silence
  timer, which is faster and interrupts people more.
* `--model zipformer-en` drops the default recogniser for one that costs about
  a fifth of the CPU and 143MB less disk, at the price of block capitals and no
  punctuation. `--model zipformer-en-small` is cheaper still and noticeably less
  accurate; see [evaluation.md](./evaluation.md).
