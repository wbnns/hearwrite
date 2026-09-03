# Evaluation

Measured numbers, and what they are worth.

## How to reproduce

Tier 2 needs real weights and a labelled corpus, so it is opt in:

```sh
pip install -e '.[onnx]'
hearwrite models                     # downloads on first use, all checksummed
python -m pytest tests/tier2 -m tier2
```

The diarization fixtures are built from LibriSpeech `dev-clean` (CC BY 4.0),
which is not vendored. `scripts/build_fixtures.py` downloads it and assembles
conversations with known speakers and known turn boundaries.

## Streaming ASR

Apple silicon, CPU only, `zipformer-en`, 20ms chunks:

| Metric | Value |
|---|---|
| Emission delay p50 | 0.40s |
| Emission delay p90 | 0.60s |
| Real time factor | 0.03 transcript only, 0.05 with all four models |

The design doc set p50 under 600ms as a Phase 3 target. A streaming transducer
clears it with no confidence gating at all, which is the argument for making it
the default engine rather than a later milestone.

## Engines compared

The same 5.4s clip, same machine, same Coordinator. `zipformer-en` against
Whisper `tiny.en`:

| | sherpa (transducer) | whisper (LocalAgreement) |
|---|---|---|
| Transcript | `TO BUILD IS GREEN I THINK WE CAN SHIP AT THIS AFTERNOON` | `The build is green. I think we can ship at this afternoon.` |
| Emission delay p50 | 0.52s | 1.62s |
| Emission delay p90 | 1.21s | 1.99s |
| Real time factor | 0.029 | 0.230 |
| Install | 1 package | 20 packages |

This is the whole Track A versus Track B argument, measured. The transducer
commits three times sooner and costs eight times less. Whisper gets punctuation,
casing and about a hundred languages. Neither is strictly better, which is why
both ship and why the interface had to accommodate both from the start.

The Whisper numbers also understate its problem. Its cost per pass is roughly
constant regardless of how much real audio arrived, so a three minute session
degrades in a way a five second clip does not. The buffer is trimmed at the
agreement point to bound that, and `test_whisper_buffer_stays_bounded_over_a_long_session`
is the guard.

## Diarization

Word level speaker confusion against known turns. LibriSpeech `dev-clean`
speakers, one utterance per turn, 0.7s of silence between turns, no overlap.

| Speakers | Found | Confusion | Null rate | Turn label p50 | p90 |
|---|---|---|---|---|---|
| 2  | 2  | 3.1% | 0.0% | 1.44s | 4.09s |
| 4  | 4  | 6.1% | 0.0% | 1.41s | 2.58s |
| 8  | 8  | 6.7% | 0.9% | 1.46s | 2.98s |
| 16 | 16 | 7.2% | 0.6% | 1.42s | 2.17s |
| 24 | 24 | 7.6% | 0.0% | 1.41s | 2.19s |

Nothing tells the pipeline how many people are talking. The count is discovered,
and it was exact at every size tested.

### What this does NOT show

**This is an easy corpus, and the numbers should be read that way.** LibriSpeech
is clean read speech recorded close to the microphone, one speaker at a time,
with a clear pause at every turn boundary. Real conversation has overlap,
interruptions, crosstalk, room noise and back channels ("mm hm", "right") that
are too short to embed. Expect materially worse results on AMI or VoxConverse,
and do not compare these figures to a published DER: DER also counts missed
speech and false alarms, and it is usually measured on much harder audio.

**Turn label latency misses its target.** The design doc asked for p90 under
1.5s; the measured p90 is 2.2s to 4.1s. A label cannot exist before a full 2s
embedding window has closed, so this is a floor set by the window length, not a
tuning problem. Shorter windows trade it directly against accuracy.

**Confusion is not the same as being right.** A word is scored against the turn
its midpoint falls in, so a word straddling a real speaker change can be marked
wrong even when the labelling is defensible.

## Endpointing

Whether a pause is the end of a thought. The corpus is 40 pairs: each full
LibriSpeech utterance alongside the same audio cut immediately after a function
word ("the", "of", "and", "to"). A clip ending there cannot be a finished
sentence, so the negative labels need no human.

| Policy | Threshold | Cuts the speaker off | Left to the timeout |
|---|---|---|---|
| conservative | 0.70 | 5.0% | 60.0% |
| balanced | 0.60 | 15.0% | 32.5% |
| aggressive | 0.55 | 22.5% | 22.5% |

The design doc asked for a false endpoint rate under 5% on mid thought pauses.
Conservative lands **at** 5.0%, not under it.

The two failures are not symmetric. Cutting someone off mid sentence produces a
turn boundary in the wrong place, permanently. Failing to notice that they
finished only means waiting for the acoustic timeout, which is latency, not
loss. That asymmetry is why the thresholds sit where they do and why the numbers
are reported apart.

Reproduce with `hearwrite endpoints ~/.cache/hearwrite/corpora/fixtures/midthought`.

### Scoring rate is part of the threshold

The completeness score is noisy frame to frame. Measured across one pause it
read 0.58, 0.61, 0.67, 0.66, 0.61, 0.65, 0.61, 0.65, then spiked to 0.71 before
falling back.

The gate fires on the first frame that crosses, so running the detector at frame
rate does not test "is the score above 0.70". It tests "is the maximum of fifty
samples a second above 0.70", which is a much weaker condition and would put the
real false endpoint rate well above the table above. The calibration measured one
window per utterance; the runtime samples a few times a second, which keeps the
two comparable.

`turn_interval` is therefore a correctness dial, not just a cost one. It also
cuts the real time factor of a 106 second multi speaker file from 0.076 to
0.046, on top of the larger saving from not scoring at all until the acoustic
gate is satisfied (0.330 to 0.076).

### The positive labels are noisy

LibriSpeech utterances are audiobook chunks and do not always end on a sentence
boundary, so some "complete" clips are not complete. That inflates the
"left to the timeout" column and leaves the false endpoint column trustworthy.
Quote the first, not the second.

### smart-turn v3.2 does not work with the documented features

v3.0, v3.1 and v3.2 all take the same 80 by 800 log mel input. Fed the features
Whisper defines and smart-turn's own reference computes, v3.0 and v3.1 separate
finished from unfinished by about 0.20 of probability. v3.2 separates them by
**-0.006** -- it returns the same answer for a finished sentence and one cut off
after "and". Its preprocessing must differ in a way that is not published, and
smart-turn's own `inference.py` still pins v3.1. HearWrite defaults to v3.1 and
keeps v3.2 in the registry so the finding stays reproducible.

## C1 confidence gating

The reference system learns a per word delay. C1 approximates it with policy and
no training, in two directions: hold a stable word the engine is unsure about,
or take a tentative word it is very sure about without waiting.

Which one helps depends entirely on the engine, and that is the interesting
part. A greedy transducer settles a word the moment it emits it, so there is no
tentative state to skip. LocalAgreement over Whisper holds every word until two
passes agree, so early commit removes a whole pass of latency:

| `early_commit_confidence` | p50 delay | p90 delay | Transcript |
|---|---|---|---|
| off | 1.62s | 1.99s | unchanged |
| 0.95 | 1.38s | 1.99s | unchanged |
| 0.85 | 0.64s | 1.99s | unchanged |
| 0.70 | 0.64s | 1.99s | unchanged |

A 2.5x reduction in median emission delay at no cost to this transcript. p90 is
unmoved because the final words still wait for the last pass.

It is off by default. The gain is real but it is a gamble: an early committed
word can still be revised, and under the append only rule the revision is
dropped rather than emitted, so the cost of being wrong is a wrong word rather
than a contradiction.

**A bug worth recording.** The first implementation filtered tentative words by
confidence individually. Committing a confident later word advanced the commit
frontier past an unconfident earlier one, which deleted it: the transcript came
back as "is green. I think..." with "The build" silently gone. Only a contiguous
prefix may be taken, and the same rule applies when holding a word -- a held
word blocks everything after it, because emitting word five while word four is
pending leaves a hole the append only rule makes permanent.

## The polish pass

A streaming recogniser tuned for latency emits bare words, and bare words read
as wrong even when every word is right. A 31MB text model can put punctuation
and casing back about 6ms after a sentence finishes, which is nothing next to
the recogniser.

Measured on one real recording, the same 11 seconds of speech:

| Path | Output | Real time factor |
|---|---|---|
| `nemotron-3.5-160ms` | `The Times January third two thousand nine Chancellor on Brink of Second Bailout for Banks` | 0.235 |
| `zipformer-en` + polish | `The times January, third, two thousand nine chancellor on Brink of second bail out for banks` | **0.037** |

Six times cheaper, and it gains commas. It also loses: "Times" and "Chancellor"
become lowercase, and "bailout" splits in two. Neither path dominates, which is
why both ship.

### It must not run behind a model that already punctuates

Not a judgement call, a measurement. Given Nemotron's own output the punctuation
model returns `The times January three, two thousand nine chancellor on Brink of
second Bailout for Banks` -- it lowercases two proper nouns and capitalises a
common one. Given text that is already punctuated it produces `stairs..`. And
given UPPERCASE text it returns it unchanged, which would make the whole stage a
silent no-op behind exactly the recogniser it exists to help.

So the model's own registry entry records whether it punctuates, and the polish
is built only when it does not. `--punctuate` and `--no-punctuate` override it.

### Why a second model may touch committed text at all

Rewriting a committed word is precisely what the append only rule forbids. The
polish gets to exist because it is held to a narrower promise: it may change
RENDERING and nothing else.

`polished` is a separate event that supersedes for display, and the Coordinator
verifies the result contains the same words in the same order before emitting
it. A model that drops, adds or substitutes a word has its output discarded and
the committed text stands. A consumer that ignores `polished` still gets a
correct transcript, which is the same promise `partial` carries.

### Utterances are fragments, so the model gets context

An endpoint can fall mid clause, so polishing each utterance alone capitalised
the middle of sentences -- "the Stairs". The previous utterance's last six words
are passed as context and stripped from the result, which tells the model the
clause continues.

## Why abstention is reported separately

A word HearWrite declines to label and a word it labels wrongly are different
outcomes with different costs. Under the append only rule a wrong speaker label
can never be corrected, so the clustering is built to abstain instead of guess.
Folding both into one number would make a system that abstains look identical to
one that guesses, which would punish exactly the behaviour the design requires.

## Calibration

The clustering threshold is a property of the embedding model, not of speech.
It was measured, not chosen: 40 LibriSpeech speakers, four windows each.

| Embedding model | Same speaker | Different speakers | Best error |
|---|---|---|---|
| TitaNet small (192d) | 0.69 | 0.06 | under 1% |
| WeSpeaker CAM++ (512d) | 0.92 | 0.78 | 15% merges |
| 3D-Speaker CAM++ (512d) | 0.85 | 0.74 | 15% merges |

TitaNet is the default because its embeddings are well conditioned: different
speakers sit near orthogonal, so a threshold means something. The other two put
unrelated speakers at 0.74 to 0.78 cosine, where no threshold separates them.

Window length matters as much as the model. Same speaker similarity climbs with
window length while different speaker similarity stays flat near 0.06, so a
short window looks unlike itself. At a fixed threshold of 0.40:

| Window | Merges | Splits |
|---|---|---|
| 1.0s | 0.28% | 19.6% |
| 1.5s | 0.48% | 2.5% |
| 2.0s | 0.55% | 0.4% |
| 3.0s | 0.50% | 0.0% |

Hence a fixed 2.0s window and a 1.5s minimum. Anything shorter is not embedded,
and the words in it borrow a label from their neighbours or commit as null.

**If you change the embedding model, redo this measurement.** The threshold
travels with the model and a stale one fails quietly.

## A warning about synthetic speech

The first attempt at this calibration used macOS `say` voices, and it was
worthless. On TTS audio the same voice reading different text scored 0.54 while
two different voices reading the same text scored 0.83 -- the embedding was
tracking the words, not the speaker. Speaker embedding models are trained on
human speech and do not transfer to a synthesiser. Use real recordings.
