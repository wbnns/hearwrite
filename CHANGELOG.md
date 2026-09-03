# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Link metadata and a preview image.** A descriptive title and meta
  description on both copies of the page, and OpenGraph plus Twitter card tags
  on the published copy only, since they carry absolute URLs that a page served
  from localhost has no business advertising. The preview image is
  `scripts/og.html` rendered by `scripts/build_og.py`, so it inherits the site's
  palette and font stack and the figures on it are the figures the page
  publishes, rather than a hand drawn asset that goes stale on its own.
- **A published page at hearwrite.wbnns.com**, built by `scripts/build_site.py`
  from the page the server itself serves, so the public copy cannot drift from
  the live one. The interactive card becomes a recording of a real session and
  the capture script is stripped, because a page with no service behind it must
  not try to open a WebSocket. The recording is labelled as a recording, and the
  provenance line names the machine the measurements came from rather than
  pointing at the machine serving the page, which is no longer the same one.

### Fixed

- **The last word of an utterance no longer waits for the session to end.** A
  transducer holds its trailing word until a new word starts, so a short
  utterance left its final word tentative until the user pressed stop: measured
  at 5.90s from spoken to final. The commit policy now settles the held word
  when the acoustic gate reports silence, which is the signal that the word
  itself is finished, rather than waiting for the endpoint, which also needs the
  semantic gate. The same word now commits in 0.60s.
- The demo page reported the wrong percentiles for short sessions. Indexing at
  `floor(n * q)` returns the maximum for n = 2, so a two word session displayed
  its worst delay as both p50 and p90. Now nearest rank.
- Silence in the demo waveform drew as a row of 2px bars, which read as a dotted
  rule rather than as quiet. It is a continuous baseline now.
- The last committed word kept its "just landed" highlight indefinitely, because
  the highlight was only cleared by the next commit. An endpoint clears it too.
- The demo's speaker gutter no longer indents the transcript by 116px before any
  speaker has been identified, which is every solo and every short session.

### Added

**A demo page rather than a demo.** The live transcription card now sits in a
page that presents what this project measures: hero figures, the trade between
the two recognisers as small multiples, the speaker curve, and -- given equal
weight -- what does not work.

The charts follow one rule that decided most of their design: the palette was
VALIDATED rather than chosen. Slots 1 and 2 of a reference categorical palette,
stepped separately for the light and dark surfaces, run through a contrast and
colour vision deficiency check on both: worst all-pairs CVD delta E 26.8 dark and
24.7 light against a floor of 8. Dark mode is its own steps, not an inverted
light mode. Every chart carries a table view, because colour is not an accessible
way to read a value, and every marker carries a hit target larger than itself.

Two sections exist to stop the page overselling. "What does not work" states the
shared microphone failure with the similarity numbers that explain it, on the
same page as the flat speaker curve, so the curve cannot be read as a general
claim. "How it compares" says plainly that HearWrite cannot be placed on the
published leaderboards: the streaming word error index is a private corpus, and
AMI and VoxConverse are public but have not been run. Tests assert both sections
stay there.


**`hearwrite wer`**, so the accuracy claim is reproducible rather than asserted.
Corpus WER over a LibriSpeech style directory, reporting substitutions,
deletions and insertions separately because they mean different things: a
deletion is audio the model missed, an insertion is a word it invented, and one
percentage hides which.

The normalisation is where the work is. Case and punctuation are stripped from
both sides, so a model is not charged three errors for writing "The Times,
January" against a bare uppercase reference. Digits are spoken back out, so
inverse text normalisation -- a feature that improves the transcript -- stops
being scored as three errors. And the "and" in "three hundred and forty two"
is dropped between number words, because keeping it measures which side of the
Atlantic a corpus came from.

Measured: `zipformer-en` 4.40% and `nemotron-3.5-160ms` 6.81% on LibriSpeech
dev-clean, streaming at 160ms lookahead on CPU.

**A section on comparing against published leaderboards**, which says plainly
that HearWrite cannot claim a place on the AA-WER Streaming Index because its
corpus is not reproducible, and has not run AMI or VoxConverse for DER although
they are public. Running our own audio and calling it comparable would be a
category error -- WER moves more between corpora than between good systems.

It also records the bias in the number we do have: zipformer was trained on
LibriSpeech, and its wins concentrate in the proper nouns of the books it is
read from. Nemotron heard "KALIKO" as "Callagh"; zipformer knows the name
because it has read the book. The honest reading is "both are in the 4 to 7%
range on easy read speech", not "zipformer is better".

**A rebuilt demo page.** A centred card, the speaker on the left, large type,
the most recent word highlighted the way a caption marks what is being said, and
a live waveform drawn from the same samples that go to the recogniser rather than
a second analyser -- so what you see is what it is hearing, silences included.

## [0.1.0] -- 2026-09-03

Not 1.0. The event protocol is frozen and the transcription pipeline is solid,
but "1.0" would claim the design is settled, and diarization is not. Shipping
0.1.0 says what is true: this does streaming transcription well and does multi
speaker capture badly on one microphone.


### Changed -- the documentation now says which half is ready

Streaming transcription and diarization are at different maturities and the
README was reporting them as though they were not. "Speaker count: discovered,
exact at 2, 4, 8, 16 and 24" sat in the headline table with the caveat
underneath, which is the wrong way round: a number in a table gets read and a
caveat under it does not. Someone tried it on three people in a room and got two
speakers, which is exactly what the caveat had warned about and exactly what the
table had promised away.

There is now a Status section that says plainly what works and what does not, the
diarization figures are reported next to their failure case rather than under a
disclaimer, and `docs/evaluation.md` carries the real conversation result with
the similarity numbers that explain it.

Also added a Constraints section: CPU only, English in practice, 16kHz mono with
no resampling, no authentication or TLS, overlap detected rather than separated,
no orthography beyond punctuation and numbers, and weights downloaded rather than
redistributed. All true by design and intended to stay true.

The roadmap is now ordered by what would actually help. Multi stream rooms first,
because with per speaker capture the label is which stream the audio arrived on
-- bookkeeping rather than machine learning, and better than single microphone
diarization will ever be. Sortformer second and conditionally. A real diarization
corpus third, because every number here comes from concatenated LibriSpeech and
nobody knows what they are on meeting audio.


### Added -- it fits on a small VPS

**Models are loaded once and shared across sessions.** Every session used to
load its own copy: about 230MB and a second of startup each, so the default
admission limit of four wanted roughly a gigabyte of resident memory before
anyone had said a word. Five concurrent sessions now cost 339MB total instead of
about 1.1GB, and connecting takes 0.01s instead of 0.86s after the first.

The split is verified, not assumed. The recognizer, the speaker embedder and the
turn detector's ONNX session hold no per-stream state -- two streams interleaved
chunk by chunk on one recognizer produce byte identical output to running each
alone. The VAD is not shared, because it has a `reset()` and therefore carries
state; at 629KB a copy per session costs nothing.

**`hearwrite models --prune`** deletes the float builds of every model, which
are downloaded and never loaded. That is 354MB of a 602MB cache.

**A Dockerfile and `docs/deployment.md`** with measured numbers rather than
estimates: 172MB of packages, 265MB of pruned weights, ~340MB of shared memory
plus ~3MB per session, real time factor 0.047 for one stream and 0.133 each for
eight concurrent. A 1GB VPS runs it.

**The admission limit is derived from the machine** rather than hardcoded to
four. Memory is the binding constraint well before CPU, so the core count is
conservative by a wide margin, which is the right side to err on.

### Fixed

**Speaker labelling got slower the longer a session ran.** `label_for` scanned
every resolved segment, so it cost 1us per word in the first minutes and 61us
per word four hours in -- linear per word is quadratic over a session. Segments
older than the labelling window are now dropped, and the cost is flat at 0.7us
regardless of session length.


### Fixed -- the entrances had drifted

**The WebSocket service had been running without diarization or a semantic
gate.** It was written in Phase 1a and never updated, so `serve --policy
conversation` quietly delivered no speaker labels and no semantic endpointing
while `transcribe` delivered both on the same audio. Nothing ever failed;
the output was simply worse over one entrance than the other. Both now call a
single `hearwrite.pipeline.build`, and a test asserts the server constructs no
components of its own.

**`requires-python` claimed 3.11 and meant 3.11.4.** Model archives are
extracted with `tarfile`'s `data` filter, a security backport that landed in
3.11.4; on 3.11.0 through 3.11.3 the call raises TypeError. The floor is now
honest, and extraction refuses to run without the filter rather than falling
back to unprotected extraction.

**The log mel tests were skipping in CI.** numpy was not in the `dev` extra, so
the most safety critical numeric code in the project -- a feature extractor that
is quietly wrong degrades accuracy invisibly -- was never exercised by the
pipeline that is supposed to catch that. numpy is now a dev dependency, and the
filterbank has a checked in golden reference so the check runs without pulling
faster-whisper in.

### Added

`CODE_OF_CONDUCT.md` and `SECURITY.md`. The security policy is specific about
what is actually worth reporting: checksum bypass, archive extraction escapes,
and the WebSocket service. It is equally specific that the absence of
authentication on the audio path is a documented design decision, not a bug.

Verified on Python 3.11.15 as well as 3.14: 169 tests green on both.

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
