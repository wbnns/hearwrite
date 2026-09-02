# CLAUDE.md — the map for building on HearWrite

Read this first. It is a map, not a readme: it tells you the one obvious way to
do each thing and points at the source-of-truth files.

## Step 0: run it

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
bin/check                        # must be green before you change anything
.venv/bin/hearwrite demo         # a full session through the real Coordinator
```

`hearwrite demo` uses scripted fakes for the models, but everything downstream of
them is production code. It is the fastest way to see what the system actually
emits.

## The one rule that keeps the output trustworthy

**Committed output is append-only.** Once a `commit` event is emitted, no later
event may contradict it — not a correction, not a re-segmentation, not a speaker
relabel.

This is not a style preference. It is the property that lets a consumer ignore
every `partial` and still hold a correct transcript, which is what keeps a simple
integration simple. It also has a consequence that shapes half the codebase: **a
wrong value is permanent, a missing one is not.** That is why the clustering
abstains instead of guessing, and why `speaker: null` is a supported outcome
rather than a failure.

If you are about to write code that "fixes up" something already committed, stop.
The answer is to not have committed it yet.

## The second rule: no wall-clock time

Every timestamp comes from `StreamClock` (`src/hearwrite/clock.py`), which only
advances when audio is pushed. Nothing in the pipeline calls `time.time()` to
produce an event value.

That is what makes a fixture replayed at 10x produce a byte-identical log to real
time, which is what makes CI trustworthy. `test_replay_speed_does_not_change_the_log`
will catch a violation, but by then you have already designed something that
needs unpicking.

Wall-clock lag is real and useful — it is passed in as `push(..., lag=...)` and
only ever gates whether partials are dropped. It never reaches a timestamp.

## The anatomy of a component (copy this shape to add a backend)

Every model sits behind a narrow `Protocol`, with a scripted fake beside it:

```
src/hearwrite/<layer>/
  base.py     the Protocol + its value types    <- the contract
  fake.py     a scripted double for Tier 1      <- how it gets tested
  <impl>.py   one real backend                  <- swappable
```

To add, say, a new ASR engine:

1. Read `src/hearwrite/engines/base.py` in full, including the docstring. The
   interface is defined to a **transducer's** shape on purpose — `push()`
   returning `None` is the blank/wait decision, not an error. Adapt your engine
   up to that shape. Do not reshape the interface to suit an easier model.
2. Write `src/hearwrite/engines/<name>.py`. It translates; it decides nothing.
   No thresholds, no timing policy, no committing.
3. Add the dependency to an **extra** in `pyproject.toml`, never to
   `[project.dependencies]`.
4. Add a `NOTICE` entry with the licence of the **weights**, not just the code.
5. The existing Tier 1 suite should pass unchanged. If it does not, either the
   adapter is wrong or the interface leaked — investigate before "fixing" a test.

## Source of truth (read these, don't guess)

| Question | File |
|---|---|
| What events exist and what may contradict what | `src/hearwrite/events.py` |
| The wire format | `src/hearwrite/protocol.py` |
| How time works | `src/hearwrite/clock.py` |
| What is tunable, and the preset axes | `src/hearwrite/coordinator/policy.py` |
| When text becomes final (incl. C1 gating) | `src/hearwrite/coordinator/commit.py` |
| How speakers are discovered and aligned | `src/hearwrite/coordinator/speakers.py` |
| When an utterance has ended | `src/hearwrite/coordinator/endpoint.py` |
| What happens under load | `src/hearwrite/coordinator/backpressure.py` |
| Which model has which licence | `NOTICE` |

## What is already built (don't rebuild it)

- The event model, the log with its append-only and time-ordering enforcement,
  and the protocol.
- The full Coordinator: commit policy with C1 confidence gating, online
  clustering with threshold-and-margin assignment, bounded history, deferred
  merge and LRU eviction, the conjunctive endpoint gate with timeout fallback,
  and backpressure.
- Scripted fakes for all four model layers.
- **The sherpa-onnx transducer engine and the ONNX Silero VAD**, both working on
  CPU. `src/hearwrite/models.py` resolves, downloads and checksums weights.
- **The speaker frontend** (`src/hearwrite/speakers/sherpa.py`) and the metrics
  module, plus `hearwrite bench`. Diarization works end to end.
- **Two ASR engines**: the sherpa transducer (default) and faster-whisper with
  LocalAgreement. `tests/tier1/test_engine_parity.py` holds them to the same
  contract; if a new engine needs a Coordinator change to pass it, the
  abstraction has leaked and the fix belongs in the adapter.
- **The WebSocket service** with admission control, in `src/hearwrite/server/`.
- `hearwrite demo | policies | models | transcribe | serve`.
- 86 Tier 1 tests and 11 Tier 2 tests, `bin/check`, CI.

**Not built yet:** semantic endpointing (Phase 3). The endpoint gate runs
acoustic only today, because no turn detector is wired in, so `_completeness()`
returns 1.0 and the conjunction reduces to the silence timer.

## The clustering threshold is calibration, not taste

`SpeakerPolicy.threshold` is a property of the EMBEDDING MODEL, not of speech.
It was measured on 40 LibriSpeech speakers and the numbers live in
`docs/evaluation.md`. Two consequences:

- **Change the embedding model and the threshold is wrong**, silently. Redo the
  measurement. `test_defaults_match_the_measured_calibration` is the tripwire.
- **Never calibrate on synthetic speech.** The first attempt used macOS `say`
  voices and was worthless: the same voice reading different text scored 0.54
  while two different voices reading the same text scored 0.83. The embedding
  was tracking the words, not the speaker.

Window length is part of the same calibration. Same speaker similarity climbs
with window length while different speaker similarity stays flat, so a short
window looks unlike itself. That is why windows are a fixed 2s and anything
under 1.5s is not embedded at all.

## Three things about the real engine that cost us bugs

Each of these was found by running audio, not by reasoning, and each has a test:

- **A transducer releases its last words late.** It needs trailing audio first.
  So `flush()` feeds silence, or the end of every session is lost, and turn
  membership is decided by a word's AUDIO position rather than by which turn
  happened to be open when it arrived.
- **The VAD and the ASR disagree about where speech ends.** The acoustic gate
  called silence at 1.04s on a clip whose next word ran to 1.32s. Do not assume
  an endpoint means the transcript is complete up to that point.
- **`at` is when an event was emitted**, never what audio it describes. Audio
  positions belong in the payload. `EventLog` now rejects an event from the past.

## Two things that look like bugs and are not

- **`speaker: null` on a committed word.** The clustering declined to guess. A
  later `speaker` event supplies the label if it becomes unambiguous. Do not
  "fix" this by lowering the margin.
- **A merged speaker's older words keep the old label.** Merges apply to open and
  future turns only; already-committed labels stand, because rewriting them would
  break the one rule. It is a documented, bounded imperfection.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Module docstrings explain **why** the module exists and what breaks if it is
  wrong, not what the functions are named.
- Comments state the failure mode. "Raise rather than clamp, because a clamped
  timestamp produces a plausible-looking metric that is quietly wrong" is the
  register.
- Stdlib-first. The base install has zero dependencies and that is a feature.
- No emoji.

## Done means

`bin/check` is green — lint, format, types, and the whole Tier 1 suite. It needs
no GPU, no network and no downloads, and it is exactly what CI runs.
