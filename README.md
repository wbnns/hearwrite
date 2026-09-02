# HearWrite

**Real-time transcription, speaker labels and endpointing, from open-weight models you can actually redistribute.**

Feed it a live audio stream and it emits three things: transcript words as they
are recognized rather than after you stop talking, speaker labels attached at
turn boundaries, and endpoint signals that mark when someone actually finished a
thought instead of merely stopped making noise.

Meta's Muse Voice Transcribe does all three from one closed model, emitting
speaker and endpoint markers as special tokens in a shared vocabulary. HearWrite
reproduces the capability set from separate open components. The models are the
easy part and they are swappable; the hard part is coordinating three signals
into one ordered stream that a client can trust. That coordination is what this
project actually is.

Everything in the default install runs on ONNX Runtime, on CPU, on a laptop. No
torch, no gated downloads, no licence to accept before you can try it.

## Status

Early. The event model, protocol, Coordinator and the deterministic test suite
are built and green. Real model backends are landing next -- see
[CHANGELOG.md](./CHANGELOG.md) for what is actually done rather than what is
planned.

## The one rule

**Committed output is append-only.** Once HearWrite emits a `commit`, nothing
later contradicts it -- not a correction, not a re-segmentation, not a speaker
relabel. `partial` events may be revised or withdrawn at any time.

The useful consequence: a consumer that ignores every `partial` still receives a
correct, complete transcript. That is what lets a simple integration stay simple.

## What it is not

- **Not a Whisper wrapper.** Whisper is offline by construction and streams only
  by re-running inference over a growing buffer. It is supported as a second
  engine, deliberately not as the default.
- **Not a speech separation system.** Overlapping speech is detected and
  labelled `null`. It is not untangled.
- **Not a batch transcription tool.** A simpler path exists for offline files;
  do not conflate the two.
- **Not a claim to match the reference.** HearWrite targets comparable, not
  better. Where it is worse, the numbers say so.
- **Not tied to any one model.** Every model sits behind a `Protocol` with a
  fake for testing. Swapping one is a config change.

## How it works

1. **Ingest.** Audio arrives as 16kHz mono PCM. A stream clock counts samples,
   never wall-clock seconds.
2. **Recognize.** A streaming transducer emits words, or emits nothing -- the
   blank symbol is a first-class "keep listening" decision, not an error.
3. **Cluster.** Speaker segments get embeddings; incremental clustering turns
   them into stable labels with no fixed speaker count.
4. **Gate.** An endpoint fires only when silence is long enough *and* the
   utterance reads as a finished thought, with a timeout so a speaker who trails
   off cannot hang the session.
5. **Emit.** One ordered, append-only event log over WebSocket.

## Speakers

`speakers=SOLO` skips the speaker frontend entirely. Not "clustering with one
cluster" -- the models never run. Diarizing a single voice occasionally splits
that person into two labels, which is worse than making no distinction, and it
costs two models on the hot path for the privilege.

`speakers=AUTO` has no fixed speaker count. Twenty speakers is twenty centroids.
What degrades as speakers are added is accuracy, not capacity, and because a
committed speaker label can never be corrected, the clustering abstains rather
than guesses: an ambiguous segment produces `speaker: null` and a later `speaker`
event fills it in once the identity is unambiguous.

We do not publish a supported speaker count. We publish the measured curve.

## Layout

```
src/hearwrite/
  events.py            the append-only event log
  clock.py             audio-relative time; nothing reads the system clock
  protocol.py          wire schema, frozen
  coordinator/         all state and all policy lives here
    commit.py            what text is final, and when
    speakers.py          online clustering and word-to-speaker alignment
    endpoint.py          the conjunctive acoustic + semantic gate
    policy.py            speaker mode and endpoint mode, orthogonal
    backpressure.py      drop partials, never commits
  engines/             ASR wrappers   (interface defined to the transducer)
  speakers/            speaker frontend wrappers
  vad/                 acoustic gate wrappers
  turn/                semantic gate wrappers
tests/tier1/           fast, deterministic, no models, no network
tests/tier2/           real models and real metrics, opt-in
```

## Local development

Python 3.11+ required. Nothing else.

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
bin/check                      # lint, typecheck, Tier 1 -- the definition of done
```

`bin/check` needs no GPU, no network and no model downloads, and it is the same
thing CI runs. If it is green locally the pipeline will be green too.

## Contributing

Issues and PRs welcome. Please run `bin/check` before opening one.

## License

MIT — see [LICENSE](./LICENSE). Third-party model weights carry their own terms;
see [NOTICE](./NOTICE).
