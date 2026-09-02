# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
