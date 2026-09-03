# HearWrite

**Real time transcription, speaker labels and endpointing. Open weights, CPU only, no GPU anywhere.**

Feed it a live audio stream and it emits three things: transcript words as they
are recognized rather than after you stop talking, speaker labels attached at
turn boundaries, and endpoint signals that mark when someone actually finished a
thought instead of merely stopped making noise.

Meta's Muse Voice Transcribe does all three from one closed model, emitting
speaker and endpoint markers as special tokens in a shared vocabulary. HearWrite
reproduces the capability set from separate open components. The models are the
easy part and they are swappable; the hard part is coordinating three signals
into one ordered stream a client can trust. That coordination is what this
project actually is.

It runs on a laptop. A 1GB VPS is enough.

## Try it on your Mac

```sh
pip install 'hearwrite[onnx,turn,server]'
hearwrite serve --open
```

That downloads the models on first run, starts the service, and opens a browser.
Click **Start listening**, allow the microphone, and talk. Words appear as they
are recognised, speaker labels attach at turn boundaries, and a `⏎` marks where
the endpoint fired.

The page is served by the same process on the same port, so there is nothing
else to run. It uses your browser's microphone, captures at 16kHz, and streams
raw PCM over the WebSocket the service already speaks — which means the demo and
a real integration are the same code path.

```
  0    0.74s  speech_onset
  1    1.12s  partial       'MISTER'
  2    1.44s  turn_start    turn 1, speaker -
  3    1.44s  commit        [-] 'MISTER'    delay=+0.56s
  6    2.08s  commit        [-] 'QUILTER'   delay=+0.64s
 11    2.72s  turn_start    turn 2, speaker A
 12    2.72s  commit        [A] 'APOSTLE'   delay=+0.36s
 14    2.72s  speaker       seq 3 -> A
 15    2.72s  speaker       seq 6 -> A
```

That is real output, not a mock up, and the `-` is the point. A speaker label
cannot exist before enough audio has been heard to identify the voice, so the
first words commit with `speaker: null` and the `speaker` events at the end fill
them in retroactively. Committing a *guess* there would be unfixable; leaving a
gap is not. The browser UI shows those words with a dotted underline until they
resolve.

## Status

**Streaming transcription is ready to use. Speaker diarization is experimental
and does not work on a shared microphone.** Those are two different maturities in
one repository and it is worth being blunt about which is which.

| | |
|---|---|
| Streaming ASR, endpointing, punctuation, numbers | **works** |
| Solo and dictation capture | **works** |
| WebSocket service, browser UI, Docker | **works** |
| Diarization, per speaker capture | works, see the numbers below |
| Diarization, several people on ONE microphone | **does not work reliably** |
| Multi stream rooms | not built, see the roadmap |

## What it does, measured

Every number below is measured on an Apple M4, CPU only, and reproducible with
the commands in [docs/evaluation.md](./docs/evaluation.md). Nothing here is an
estimate.

| | |
|---|---|
| Emission delay | p50 **0.28s** to **0.44s**, p90 **0.52s** to **1.00s** |
| False endpoint | **5.0%** of mid thought pauses |
| Real time factor | **0.26** default recogniser, **0.045** with the light one |
| Memory | ~340MB shared across sessions, ~3MB per session |
| Install | one package, zero dependencies, before any extra |

Diarization is reported separately and with its failure case attached, because a
number in a table gets read and a caveat under it does not:

| Capture | Result |
|---|---|
| Per speaker recordings, 2 to 24 voices | **3.1% to 7.6%** word confusion, speaker count exact every time |
| Three people, one laptop microphone, ~1s turns | **two speakers found for three**, turns split mid sentence |

The first row is clean audio with one speaker per recording. The second is a real
conversation someone recorded with this software. Both are true, and the second
is the one that predicts your meeting.

## The one rule

**Committed output is append only.** Once HearWrite emits a `commit`, nothing
later contradicts it. Not a correction, not a resegmentation, not a speaker
relabel. `partial` events may be revised or withdrawn at any time.

The useful consequence: a consumer that ignores every `partial` still receives a
correct, complete transcript. That is what lets a simple integration stay simple.

It has a second consequence that shapes half the codebase. A wrong value is
permanent where a missing one is not, so **HearWrite abstains rather than
guesses**: an ambiguous speaker produces `speaker: null`, and a later `speaker`
event fills it in once the identity is unambiguous.

## What it is not

* **Not a Whisper wrapper.** Whisper is offline by construction and streams only
  by re-running inference over a growing buffer. It ships as a second engine,
  deliberately not the default: on the same clip it commits three times slower
  and costs eight times more, in exchange for punctuation, casing and a hundred
  languages. Choose per job with `--engine`.
* **Not a speech separation system.** Overlapping speech is detected and
  labelled `null`. It is not untangled.
* **Not a batch transcription tool.** A simpler path exists for offline files.
* **Not a claim to beat the reference.** HearWrite targets comparable, not
  better. Where it is worse, the numbers say so.
* **Not tied to any one model.** Every model sits behind a `Protocol` with a
  fake for testing. Swapping one is a config change.

## How it works

1. **Ingest.** 16kHz mono PCM. A stream clock counts samples, never wall clock
   seconds, so a fixture replayed at 10x produces a byte identical event log.
2. **Recognize.** A streaming transducer emits words, or emits nothing. The
   blank symbol is a first class "keep listening", not an error.
3. **Cluster.** Speech windows get speaker embeddings; incremental clustering
   turns them into stable labels with no fixed speaker count.
4. **Gate.** An endpoint fires only when silence is long enough *and* an audio
   native turn detector reads the utterance as finished, with a timeout so a
   speaker who trails off cannot hang the session.
5. **Polish.** A serialized chain re-renders each finished sentence: a 31MB
   model adds punctuation and casing, then rules turn spoken numbers into
   figures. Each stage declares its order, what it produces, and its own check,
   so one cannot quietly undo another. The whole chain is single digit
   milliseconds.
6. **Emit.** One ordered, append only event log over WebSocket.

Every model wrapper is stateless. The Coordinator holds all state and all policy,
in plain synchronous Python with no I/O. That boundary is the design: it makes
models replaceable and makes the interesting behaviour testable in milliseconds
on a laptop with no GPU, no network and no downloads.

## Speakers

**Read this before pointing it at a meeting.**

Nothing tells HearWrite how many people are talking; the count is discovered. On
per speaker recordings that works well and keeps working as voices are added: it
found exactly 2, 4, 8, 16 and 24 speakers, with 3.1% to 7.6% word confusion.

**On one shared microphone it does not.** A real three person conversation,
recorded through a laptop, came back as two speakers with turns split mid
sentence. The cause is measurable rather than mysterious: two of the three voices
sat at 0.54 cross similarity against 0.55 within, so no threshold separates them.
Every voice arrives through the same room, the same distance and the same
microphone, and that shared channel signature swamps the individual one. Shorter
windows do not rescue it; four window and threshold combinations all returned two
speakers.

There is a second, compounding problem: conversational turns are short. In that
recording the median turn was 1.08s while a voice needs about 1.5s of speech
before it can be identified at all, so most turns produce no embedding.

**The honest fix is not a better threshold, it is per speaker capture.** If each
person has their own microphone and their own stream, the speaker label is
simply which stream the audio arrived on -- bookkeeping rather than machine
learning, and effectively perfect. That is why meeting tools do not struggle with
this. HearWrite already gives each connection its own independent session, so
three people on three laptops each get a clean transcript today; what is missing
is merging those streams into one timeline. See the roadmap.

`speakers=SOLO` skips the frontend entirely. Not "clustering with one cluster" --
the models never run. Diarizing a single voice occasionally splits that person in
two, which is worse than making no distinction, and it costs two models on the
hot path for the privilege.

## Policies

Speaker mode and endpoint aggressiveness are orthogonal, because they vary
independently: a solo voice agent wants one speaker and an impatient endpoint,
a meeting wants many speakers and a patient one.

| Preset | Speakers | Endpoint | Cuts you off |
|---|---|---|---|
| `dictation` | solo | conservative | 5.0% |
| `conversation` | auto | balanced | 15.0% |
| `agent` | solo | aggressive | 22.5% |

Being cut off and being left to the timeout are not symmetric failures. Cutting
someone off mid sentence puts a turn boundary in the wrong place, permanently.
Failing to notice they finished only means waiting, which is latency, not loss.

## Install

```sh
pip install hearwrite                    # no dependencies at all
pip install 'hearwrite[onnx]'            # streaming ASR + diarization + VAD
pip install 'hearwrite[turn]'            # semantic endpointing
pip install 'hearwrite[server]'          # the WebSocket service
pip install 'hearwrite[whisper]'         # the second engine
```

The base install pulls **exactly one package and zero dependencies**. It gets the
event model, the Coordinator, scripted fakes and the CLI, and its test suite
passes in half a second with no GPU, no network and no model downloads. That is
asserted by CI, not hoped for.

## Use it

```sh
hearwrite serve --open                  # the browser UI, mic included
hearwrite demo                          # a full session, no models needed
hearwrite models                        # what is available, and its licence
hearwrite transcribe recording.wav      # 16kHz mono WAV
hearwrite transcribe rec.wav --model zipformer-en   # 5x cheaper, no punctuation
hearwrite serve --port 8080             # WebSocket: binary PCM up, JSON down
hearwrite bench fixture.wav             # score diarization against ground truth
hearwrite endpoints midthought/         # score endpointing on mid thought clips
hearwrite models --prune                # reclaim 354MB of unread model files
```

As a library:

```python
from hearwrite import CONVERSATION, Coordinator
from hearwrite.pipeline import build

coordinator = Coordinator(CONVERSATION, **build(CONVERSATION).as_kwargs())
for chunk in stream_of_pcm:
    for event in coordinator.push(chunk):
        print(event.kind, event.payload)
```

## Deploying it

No GPU anywhere. 172MB of packages, 265MB of weights after
`hearwrite models --prune`, ~340MB of memory for the models shared across every
session and about 3MB per session on top. A `Dockerfile` is included.

Models load once and are shared. Five concurrent sessions cost 339MB in total,
not 1.1GB, and connecting takes 0.01s after the first.

See [docs/deployment.md](./docs/deployment.md) for sizing, systemd, and what to
turn off when you need it smaller.

## Models, and their licences

Nothing is gated. No account, no access token, no licence to accept before you
can try it. Every model is downloaded from its own publisher and verified
against a pinned SHA-256, and both rules are enforced by tests.

| Model | Role | Licence |
|---|---|---|
| Nemotron 3.5 ASR (160ms) | streaming ASR, punctuated | OpenMDW-1.1 |
| sherpa-onnx punct-en | punctuation and casing | Apache-2.0 |
| sherpa-onnx zipformer | streaming ASR, lightweight | Apache-2.0 |
| NeMo TitaNet small | speaker embeddings | CC-BY-4.0 |
| Silero VAD | acoustic gate | MIT |
| smart-turn v3.1 | semantic gate | BSD-2-Clause |

`hearwrite models` prints this at runtime. Full detail, including what is
deliberately *not* used and why, is in [NOTICE](./NOTICE).

## Layout

```
src/hearwrite/
  events.py            the append only event log
  clock.py             audio relative time; nothing reads the system clock
  protocol.py          wire schema, frozen
  features.py          Whisper style log mel, checked against an independent build
  models.py            model registry: public URLs, pinned checksums, licences
  loaders.py           what is shared between sessions, and what must not be
  pipeline.py          the one place the four models are assembled
  metrics.py           scoring; abstention and error counted apart, never summed
  coordinator/         all state and all policy lives here
    commit.py            what text is final, and when
    speakers.py          online clustering and word to speaker alignment
    endpoint.py          the conjunctive acoustic and semantic gate
    policy.py            speaker mode and endpoint mode, orthogonal
    backpressure.py      drop partials, never commits
  engines/             ASR wrappers: sherpa (default) and whisper
  speakers/            speaker frontend wrappers
  vad/                 acoustic gate wrappers
  turn/                semantic gate wrappers
  server/              WebSocket service, admission control, and the browser UI
tests/tier1/           fast, deterministic, no models, no network
tests/tier2/           real models and real metrics, opt in
```

## Local development

Python 3.11.4 or newer required. Nothing else.

(3.11.4, not 3.11.0: model archives are extracted with `tarfile`'s `data`
filter, a security backport that landed in that patch release.)

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
bin/check                      # lint, typecheck, Tier 1: the definition of done
```

`bin/check` runs 183 tests in half a second, needs no GPU, no network and no
model downloads, and is exactly what CI runs. If it is green locally the
pipeline will be green.

Tier 2 exercises the real models and is opt in, because it needs weights:

```sh
python -m pytest tests/tier2 -m tier2
```

## Roadmap

In the order I would do them.

**1. Multi stream rooms.** Several clients feeding one merged transcript, with a
shared room id and a common clock. This is the real answer to multi speaker: with
per speaker capture the label is which stream it came from, so it sidesteps the
clustering problem rather than fighting it, and it gives a better result than
single microphone diarization ever will. Ordinary engineering rather than
research.

**2. A shared microphone diarizer, or an honest refusal.** NVIDIA's Streaming
Sortformer is purpose built for online diarization and is already in the model
registry. It caps at 4 speakers and needs torch and NeMo, roughly 2GB, which
breaks the one package and zero dependencies property this is built around. It
may not beat clustering on a shared mic either. Worth trying, behind an extra,
and worth abandoning if it does not measurably help.

**3. A real diarization corpus.** Every diarization number here comes from
concatenated LibriSpeech, which is one speaker per recording with a clean pause
at every boundary. AMI or VoxConverse would say what the numbers actually are on
meeting audio. I expect them to be much worse, and knowing by how much is worth
more than any tuning done without it.

### Deliberately not scheduled

The design this was built from ends with a delay penalty fine tune and a learned
per word delay, trained with a combined word error and delay reward. Both need a
GPU, and the streaming transducer already beats the latency target they were
meant to reach. Until a measurement says otherwise, they are not worth the
compute.

## Constraints

Things that are true by design, and will stay true.

* **CPU only.** No CUDA path is tested. `--provider` exposes alternatives, and
  CoreML measured *slower* than CPU here for these int8 models.
* **English in practice.** The default recogniser advertises 40 locales, but
  every number here is English, the punctuation model is English only, and
  inverse text normalisation is English rules.
* **16kHz mono PCM.** No resampling: naive interpolation aliases and costs word
  error rate invisibly, so the wrong format is refused with the ffmpeg command
  to fix it.
* **No authentication, no TLS.** The audio path expects a short lived token from
  your own application and a TLS terminator in front. See
  [SECURITY.md](./SECURITY.md).
* **Overlapping speech is detected, not separated.** Overlap is labelled `null`
  rather than guessed at.
* **No orthography beyond punctuation and numbers.** "half deserted" will not
  become "half-deserted"; hyphenation needs a lexicon and would misfire more than
  it fixed.
* **Weights are downloaded, never redistributed.** Every model is fetched from
  its publisher against a pinned checksum, and nothing gated is accepted.

## Contributing

Issues and PRs welcome. Please run `bin/check` before opening one, and see
[CONTRIBUTING.md](./CONTRIBUTING.md). Security reports go through
[SECURITY.md](./SECURITY.md) rather than a public issue.

## License

MIT — see [LICENSE](./LICENSE).

Third party model weights carry their own terms and are downloaded, never
redistributed here. [NOTICE](./NOTICE) records every one of them.
