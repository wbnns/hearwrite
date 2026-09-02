# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added -- Phase 1a: it transcribes

**A working streaming pipeline on CPU.** sherpa-onnx transducer, ONNX Silero
VAD, and the Coordinator, end to end on a laptop with no GPU and no torch.
Measured on an M-series Mac with `zipformer-en`: p50 emission delay 0.40s, p90
0.60s, real time factor 0.03. The design doc set p50 under 600ms as a Phase 3
target; the transducer clears it in Phase 1a, which is the payoff for making it
the default engine rather than a later milestone.

**The engine interface held.** The ASR `Protocol` was written against a
transducer's shape before any real engine existed, and the sherpa-onnx adapter
needed no change to it. `push()` returning `None` maps onto the blank decision
exactly as intended.

**A model registry that records what each weight costs you.** Every URL is a
plain public download with a pinned SHA-256 and a recorded licence, and
`hearwrite models` prints them. Nothing is gated, so there is no token in CI and
no acceptance click between cloning and running.

**The WebSocket service.** Binary PCM up, JSON events down, one session per
connection with a hard admission limit. Audio never passes through an
application server.

**Silero VAD via sherpa-onnx rather than the `silero-vad` package**, which is a
torch model. Using the ONNX build keeps the acoustic gate in the same runtime as
the recognizer and the default install at roughly two gigabytes smaller.

### Fixed

**Event times could run backwards.** `turn_start` was stamped with its first
word's audio position instead of the position it was emitted at, so it could
claim a moment earlier than the endpoint before it. `at` now means emission
everywhere, audio positions live in payloads, and `EventLog` rejects an event
from the past.

**A late word opened a spurious turn.** A streaming transducer releases its last
words only after hearing trailing audio, so words routinely arrive after the
endpoint that followed them. Turn membership is now decided by audio position:
a word whose audio precedes the endpoint belongs to the turn that closed, not a
new one. Found by running real audio, not by reasoning about it.

**Flushing lost the end of every session.** The decoder needs trailing audio
before it emits its final words. Without padding on flush, a test clip lost
"THIS AFTERNOON" entirely; 0.3s recovered "THIS AFTER" and 0.5s recovered all of
it. `flush()` now feeds silence.

### Added -- Phase 0

**The event model and wire protocol.** One merged, ordered, append-only log
where speaker and endpoint events interleave with text, mirroring the way the
reference system emits special tokens in a shared vocabulary. The point is that
the client contract does not encode how many models sit behind it, so replacing
three models with one joint model later is an internal change rather than a
protocol break.

**The Coordinator.** All state and all policy in one place: commit policy
including C1 confidence gating, online speaker clustering with word alignment,
the conjunctive endpoint gate, and backpressure. Pure, synchronous, no I/O, and
no reads of the system clock.

**Solo mode as a bypass.** `speakers=SOLO` does not run the speaker frontend at
all. Diarizing a single voice occasionally splits that person into two labels,
which is worse than making no distinction, and skipping it drops two models off
the hot path.

**Clustering that abstains.** A segment joins a cluster only if it clears the
similarity threshold *and* beats the runner-up by a margin. Ambiguous segments
produce `speaker: null`, filled in later by a `speaker` event once the identity
is unambiguous. Under the append-only rule a wrong label is permanent and a
missing one is not, so abstention is the only safe failure.

**The Tier 1 suite.** 65 tests, no GPU, no network, no model downloads, running
in well under a second. Covers the append-only invariant, chunk-size invariance,
replay determinism, alignment edge cases, cluster eviction and merging, the
endpoint state machine including the trailing-off case, and backpressure.

**`bin/check` as the definition of done**, with CI as a thin wrapper over it,
plus a job asserting that a bare install pulls in no model runtime.

### Notes

Component selection was re-verified against the 2026 landscape rather than taken
from the design doc. Two findings changed the plan: `faster-whisper` has been
dormant since November 2025, and `sherpa-onnx` now offers a natively streaming
transducer that is Apache-2.0, CPU-first and needs no torch. The streaming
transducer is therefore the default engine rather than a later milestone, and
Whisper becomes the second engine that proves the interface generalises.
