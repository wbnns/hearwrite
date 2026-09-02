"""Online clustering and word-to-speaker alignment.

This is the module carrying the most risk, so it gets the most tests. The theme
throughout: under the append-only rule a missing label is recoverable and a wrong
label is not, so every ambiguous case must resolve to `None`.
"""

from __future__ import annotations

import math

from hearwrite.coordinator.policy import SpeakerPolicy
from hearwrite.coordinator.speakers import SpeakerTracker, cosine
from hearwrite.speakers.base import Segment
from hearwrite.speakers.fake import embedding


def seg(start, end, emb, overlap=False):
    return Segment(start, end, emb, overlap=overlap)


A = embedding(1.0, 0.0, 0.0)
B = embedding(0.0, 1.0, 0.0)
C = embedding(0.0, 0.0, 1.0)


def tracker(**kw):
    """A tracker for testing CLUSTERING, not the duration policy.

    These tests use short fixture segments, so the minimum duration is relaxed
    unless a test is specifically about it. Coupling every clustering test to
    the production default would mean recalibrating the embedding model breaks
    twenty unrelated assertions.
    """
    kw.setdefault("min_duration", 0.4)
    return SpeakerTracker(SpeakerPolicy(**kw))


# -- assignment --------------------------------------------------------------


def test_distinct_voices_get_distinct_labels():
    t = tracker()
    assert t.assign(seg(0.0, 1.0, A)).label == "A"
    assert t.assign(seg(1.0, 2.0, B)).label == "B"
    assert t.assign(seg(2.0, 3.0, A)).label == "A"
    assert t.speaker_count == 2


def test_no_fixed_speaker_count():
    """Twenty speakers is twenty centroids. Nothing caps this but max_speakers."""
    t = tracker(max_speakers=64)
    dim = 24
    for i in range(20):
        vec = tuple(1.0 if j == i else 0.0 for j in range(dim))
        assert t.assign(seg(i, i + 0.9, vec)).label is not None
    assert t.speaker_count == 20


def test_ambiguous_segment_abstains_rather_than_guessing():
    """A voice sitting between two clusters gets no label and creates no cluster."""
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    t.assign(seg(1.0, 2.0, B))

    between = embedding(0.72, 0.70, 0.0)
    result = t.assign(seg(2.0, 3.0, between))

    assert result.label is None
    assert result.reason == "ambiguous"
    assert t.speaker_count == 2, "abstaining must not invent a third speaker"


def test_overlap_is_detected_not_separated():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    result = t.assign(seg(1.0, 2.0, A, overlap=True))
    assert result.label is None
    assert result.reason == "overlap"


def test_short_segments_are_not_clustered():
    """Segments below the minimum give embeddings too noisy to trust."""
    t = tracker(min_duration=0.4)
    result = t.assign(seg(0.0, 0.1, A))
    assert result.label is None
    assert result.reason == "too_short"
    assert t.speaker_count == 0


# -- alignment ---------------------------------------------------------------


def test_word_straddling_a_boundary_goes_to_the_larger_overlap():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    t.assign(seg(1.0, 2.0, B))

    assert t.label_for(0.8, 1.1) == "A"  # 0.2 in A, 0.1 in B
    assert t.label_for(0.9, 1.4) == "B"  # 0.1 in A, 0.4 in B


def test_word_overlapping_nothing_is_unlabeled():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    assert t.label_for(5.0, 5.5) is None


def test_segment_containing_no_words_is_harmless():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    t.assign(seg(1.0, 2.0, B))  # nothing was said here
    assert t.label_for(0.1, 0.5) == "A"
    assert t.speaker_count == 2


# -- merging -----------------------------------------------------------------


def test_merge_redirects_future_lookups():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    near_a = embedding(0.2, 1.0, 0.0)
    t.assign(seg(1.0, 2.0, near_a))
    assert t.speaker_count == 2

    t.merge("A", "B")
    assert t.speaker_count == 1
    assert t.resolve("B") == "A"
    assert t.label_for(1.1, 1.5) == "A", "a merged label must resolve to its survivor"


def test_merge_candidates_only_fire_for_genuinely_close_centroids():
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    t.assign(seg(1.0, 2.0, B))
    assert t.merge_candidates() == [], "orthogonal voices must never be merged"


def test_merge_survivor_is_the_earlier_label():
    """Stable merging: the outcome cannot depend on dict iteration order."""
    t = tracker()
    t.assign(seg(0.0, 1.0, A))
    almost = embedding(0.99, 0.14, 0.0)
    t.assign(seg(1.0, 2.0, almost))
    for survivor, absorbed in t.merge_candidates():
        assert survivor < absorbed


# -- bounded memory ----------------------------------------------------------


def test_history_is_bounded():
    t = tracker(history=4)
    for i in range(20):
        t.assign(seg(i, i + 0.9, A))
    cluster = next(iter(t._clusters.values()))
    assert len(cluster.embeddings) == 4
    assert len(cluster.weights) == 4


def test_eviction_caps_cluster_count_and_mints_a_fresh_label():
    t = tracker(max_speakers=2)
    dim = 8
    for i in range(4):
        vec = tuple(1.0 if j == i else 0.0 for j in range(dim))
        t.assign(seg(i, i + 0.9, vec))
    assert t.speaker_count == 2, "max_speakers was not enforced"
    # The first speaker was evicted, so returning gets a new identity. That is
    # the documented trade for bounded memory, not a bug.
    again = t.assign(seg(10.0, 10.9, embedding(1.0, 0.0, 0.0)))
    assert again.label not in ("A",)


def test_centroid_is_not_captured_by_one_long_turn():
    """A bounded ring buffer means an old voice cannot dominate forever."""
    t = tracker(history=4)
    t.assign(seg(0.0, 30.0, A))  # one very long turn
    for i in range(6):  # then several normal ones
        t.assign(seg(31.0 + i, 31.9 + i, embedding(0.95, 0.31, 0.0)))
    cluster = next(iter(t._clusters.values()))
    assert cosine(cluster.centroid, embedding(0.95, 0.31, 0.0)) > cosine(cluster.centroid, A)


# -- primitives --------------------------------------------------------------


def test_cosine_handles_a_zero_vector():
    assert cosine(embedding(0.0), embedding(1.0)) == 0.0


def test_defaults_match_the_measured_calibration():
    """The shipped defaults are what the LibriSpeech calibration produced.

    Measured over 40 real speakers with TitaNet embeddings: different speakers
    sit at about 0.06 cosine, the same speaker at 0.62 to 0.76 for windows of
    1.5s and up. If someone changes the embedding model without redoing that
    measurement, this test is the reminder that the numbers travel together.
    """
    p = SpeakerPolicy()
    assert p.threshold == 0.40
    assert p.margin == 0.10
    assert p.min_duration == 1.5


def test_cosine_matches_the_definition():
    a, b = embedding(1.0, 1.0), embedding(1.0, 0.0)
    assert math.isclose(cosine(a, b), 1 / math.sqrt(2), rel_tol=1e-9)


def test_resolved_segments_are_pruned_so_labelling_stays_flat():
    """Otherwise the cost per word grows with the length of the session.

    `label_for` scans the resolved list, so an unpruned tracker charges 1us per
    word in the first minutes and 61us per word four hours in. Linear per word
    is quadratic over a session, and nothing needs a segment from two minutes
    ago: words are labelled moments after their audio, and an unlabelled one is
    abandoned when its turn closes.
    """
    t = tracker(segment_memory=60.0)
    for i in range(400):
        t.assign(seg(i * 2.0, i * 2.0 + 1.9, A if i % 2 else B))
    assert len(t._resolved) < 60, f"kept {len(t._resolved)} segments"


def test_pruning_keeps_everything_a_recent_word_could_need():
    t = tracker(segment_memory=60.0)
    for i in range(200):
        t.assign(seg(i * 2.0, i * 2.0 + 1.9, A))
    # A word inside the most recent segment must still resolve.
    assert t.label_for(398.0, 398.5) is not None
    # And one inside a segment from ten seconds ago.
    assert t.label_for(390.0, 390.5) is not None


def test_pruning_never_drops_a_span_a_word_still_overlaps():
    """The prune horizon is measured from the newest segment, not wall time."""
    t = tracker(segment_memory=30.0)
    t.assign(seg(0.0, 1.9, A))
    t.assign(seg(2.0, 3.9, B))
    assert t.label_for(0.5, 0.9) == "A"
    assert t.label_for(2.5, 2.9) == "B"
