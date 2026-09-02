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
| Real time factor | 0.03 |

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
