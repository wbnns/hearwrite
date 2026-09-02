# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added -- Phase 3: it knows when you have finished

**Semantic endpointing.** smart-turn v3.1 closes the gap an acoustic VAD cannot:
"what's the weather in..." and "what's the weather in Menlo Park" are identical
to a silence timer if the pauses match. The gate is conjunctive, so an endpoint
needs both, with a timeout so a speaker who trails off cannot hang the session.
Measured on a 40 pair mid thought corpus: the conservative policy wrongly ends a
turn on 5.0% of mid phrase pauses. The design doc asked for under 5%, so this
lands at the target rather than under it.

**Whisper style log mel features in numpy**, rather than adding `transformers`
for an 8MB model. Reimplementing a feature extractor is a good way to be
silently wrong, so the filterbank is checked against faster-whisper's
independent implementation: it matches to 0.0, and the full spectrogram to
1.2e-07.

**C1 confidence gating, in both directions.** Hold a stable word the engine is
unsure about, or take a tentative word it is very sure about without waiting.
Which helps depends on the engine, and that is the interesting part: a greedy
transducer settles a word as it emits it, so there is nothing to skip, while
LocalAgreement over Whisper holds everything until two passes agree. Early
commit cuts Whisper's median emission delay from 1.62s to 0.64s with no change
to the transcript. Off by default, because an early committed word can still be
revised.

**`hearwrite endpoints`**, and a mid thought corpus builder. The trick that
makes the corpus cheap is the cut point: a clip ending on "the", "of" or "and"
cannot be a finished sentence, so the negative labels need no human.

### Fixed

**Early commit could delete words from the front of the transcript.** Filtering
tentative words by confidence individually let a confident later word advance
the commit frontier past an unconfident earlier one, which then sat behind the
frontier forever. "The build is green" came back as "is green". Only a
contiguous prefix may be committed, and a held word now blocks everything after
it for the same reason.

**Features were normalised after zero padding rather than before.** Whisper
normalises under an attention mask so only real samples count; folding the
padding into the mean and variance shifted every value the model saw, and the
model responded by returning 0.72 for absolutely everything.

**The turn detector ran at frame rate, which quietly changed the threshold.**
The completeness score is noisy between frames and the gate fires on the first
crossing, so scoring fifty times a second tested "the maximum of fifty samples"
rather than "the score", which is much weaker than the calibrated number. It is
now sampled a few times a second and never before the acoustic gate could fire.
That also cut the real time factor of a 106 second file from 0.330 to 0.046.

### Findings

**smart-turn v3.2 does not work with the documented feature pipeline.** v3.0 and
v3.1 separate finished from unfinished utterances by about 0.20 of probability
on our corpus. v3.2, given identical input, separates them by -0.006. Its
preprocessing must differ in some unpublished way; smart-turn's own inference
script still pins v3.1. HearWrite defaults to v3.1 and keeps v3.2 registered so
the finding stays reproducible.

**The registry guard was testing a proxy.** It asserted every model URL began
with github.com, which is not the property that matters. What matters is that a
download needs no account, token or licence acceptance, and the only way to
check that is to try: there is now a network marked test that fetches every
registry URL anonymously.

### Added -- Phase 1c: a second engine, and the proof the interface holds

**faster-whisper behind the same interface**, using LocalAgreement: transcribe a
growing buffer repeatedly, and treat the prefix two consecutive passes agree on
as settled. `push()` returning None is the same blank decision a transducer
makes per chunk, here meaning no new pass has run.

**Adding it required zero changes to `coordinator/`, `protocol.py`, `events.py`
or `engines/base.py`.** One new file. That is the payoff for defining the ASR
interface against a transducer's shape in Phase 0, before any real engine
existed, on the argument that designing to the easier case first guarantees a
rewrite. `tests/tier1/test_engine_parity.py` runs one scenario through both
engine shapes and asserts the contract holds identically for each.

**Measured, on the same clip.** The transducer commits at p50 0.52s with a real
time factor of 0.029 and installs one package. Whisper commits at p50 1.62s with
a real time factor of 0.230 and installs twenty, and returns punctuation, casing
and roughly a hundred languages. Neither is strictly better, which is why both
ship and why `--engine` exists.

**The Whisper buffer is trimmed at the agreement point.** Cost per pass is
roughly constant regardless of how much real audio arrived, so an untrimmed
buffer makes a three minute session degrade in a way a ten second demo never
does. That is the structural limit the design doc names, and it is guarded by a
test rather than a comment.

### Note

The Whisper extra downloads from the Hugging Face hub rather than from
`hearwrite.models`, so the pinned checksum guarantee that covers every other
model does not extend to it. Recorded in NOTICE.

### Added -- Phase 1b: it hears who is talking

**Online speaker labels, with no fixed speaker count.** A speaker frontend built
on TitaNet embeddings feeds the incremental clustering that was written in Phase
0. Measured on LibriSpeech speakers: the count is discovered exactly at 2, 4, 8,
16 and 24 speakers, with word level confusion from 3.1% to 7.6% and a null rate
under 1%. Read docs/evaluation.md before trusting those numbers -- the corpus is
clean read speech with a pause at every turn and no overlap, which is much
easier than real conversation.

**A threshold that was measured rather than chosen.** The clustering threshold
is a property of the embedding model, so it was calibrated on 40 real speakers
and the measurement is written down next to the constant. Same speaker
similarity is 0.69 and different speaker similarity 0.06 at a 2s window, giving
under 1% error at a threshold of 0.40.

**`hearwrite bench`**, and a metrics module that reports abstention and error
separately. A word the clustering declined to label and a word it labelled
wrongly are different outcomes with different costs, and folding them into one
number would make a system that abstains look identical to one that guesses.

**`scripts/build_fixtures.py`**, which assembles multi speaker conversations
with known turn boundaries from LibriSpeech. The corpus is downloaded on demand
and nothing from it is vendored.

### Changed

**TitaNet replaced WeSpeaker as the default embedding model**, on measurement
rather than reputation. WeSpeaker CAM++ and 3D-Speaker CAM++ both place
*different* speakers at 0.74 to 0.78 cosine, where no threshold separates them;
TitaNet places them at 0.06. WeSpeaker stays available and the registry says why
it is not the default.

**Speaker regions now survive a short pause, and windows are emitted during
speech rather than at the end of a turn.** The first version closed a region at
the first quiet frame, which fragmented ordinary speech into pieces too short to
embed and left 87% of words unlabelled. It also waited for a turn to finish
before emitting anything, which would have made turn label latency scale with
turn length.

**A word in a gap between segments borrows a label from its neighbours**, unless
the neighbours disagree, in which case it sits on a speaker change and gets no
label. Speech regions never tile the timeline perfectly, and committing null for
want of a window was costing 22% of words.

### Known limitations

**Turn label latency misses its target.** The design doc asked for p90 under
1.5s; measured p90 is 2.2s to 4.1s. A label cannot exist before a 2s embedding
window has closed, so this is a floor set by the window length, and shorter
windows trade it directly against accuracy.

**Speaker changes are localised to a window, not to the instant.** sherpa-onnx
does not expose its segmentation model on its own, so the frontend cuts on voice
activity plus a fixed window. A change with no pause is caught at window
granularity.

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
