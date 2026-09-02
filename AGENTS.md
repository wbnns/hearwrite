# Working in this repo

HearWrite is tuned for Claude Code, but the structure is clean enough for any
coding agent. The full guide is in **[CLAUDE.md](./CLAUDE.md)** — read it. The
essentials:

## What it is

A real-time audio perception pipeline: streaming ASR, online speaker labels and
semantic endpointing, coordinated into one append-only event stream. The models
are swappable and are not the product. The **Coordinator** is the product.

## What it is not

Not a Whisper wrapper. Not a speech separation system. Not a batch transcriber.
Not tied to any one model.

## Seven rules

1. **Committed output is append-only.** Once a `commit` is emitted, nothing may
   contradict it. `partial` may be revised freely. Enforced by
   `tests/tier1/test_append_only.py` and by `EventLog.emit` itself.
2. **Never read the system clock to make a timestamp.** Every time value comes
   from `StreamClock`, which only moves when audio is pushed. This is what makes
   replay at 10x byte-identical to real time.
   (`tests/tier1/test_determinism.py`)
3. **The base install pulls no model runtime.** `pip install hearwrite` gets the
   Coordinator, the fakes and the CLI. Backends go behind an extra. CI asserts
   this.
4. **When in doubt, abstain.** A missing speaker label is recoverable; a wrong
   one is permanent. Ambiguous clustering returns `None`, never a guess.
5. **The ASR interface is defined to the TRANSDUCER's shape**, not Whisper's.
   `push()` returning `None` is the blank/wait decision. Adapt other engines up
   to it; never reshape the interface downward.
6. **Solo mode is a bypass.** `speakers=SOLO` must not call the speaker frontend
   at all. Asserted by `tests/tier1/test_solo.py`.
7. **Every model wrapper is stateless and has a fake.** Policy lives in
   `coordinator/`, never in a wrapper.

## Where things are

- `src/hearwrite/coordinator/` — all state, all policy. The interesting code.
- `src/hearwrite/{engines,speakers,vad,turn}/` — `base.py` is the interface,
  `fake.py` is the scripted test double, anything else is a real backend.
- `src/hearwrite/events.py`, `protocol.py` — the frozen contract.
- `tests/tier1/` — fast, deterministic, no models. Most bugs belong here.
- `NOTICE` — the licence map for every model. Update it when adding one.

## Do not

- **Do not add a dependency to `[project.dependencies]`.** It goes in an extra.
- **Do not use a gated model** (Hugging Face acceptance) or a non-commercial
  licence. It breaks clone-and-run. See the "Deliberately NOT used" section of
  `NOTICE`.
- **Do not make `bin/check` need a GPU, a network or a download.** If a test
  needs a real model it is Tier 2.
- **Do not put policy in a model wrapper.** Wrappers translate; they never
  decide.
- **Do not clamp a bad timestamp.** Raise. A clamped value produces a
  plausible-looking metric that is quietly wrong.

## Done means

`bin/check` is green. That is lint, format, types and the whole Tier 1 suite,
and it is exactly what CI runs.
