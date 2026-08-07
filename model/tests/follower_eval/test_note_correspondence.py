# model/tests/follower_eval/test_note_correspondence.py
"""Unit tests for the per-note correspondence baseline (#108 / #148).

The load-bearing claim of this module is that its TRUTH is not its system under
test, so most of these tests pin the truth derivation on constructed cases with
a known answer -- including the cases where the rule must refuse to decide."""

from __future__ import annotations

import pytest
from follower_bench.score_notes import ScoreNote
from follower_bench.segments import PerfNote
from follower_eval import note_correspondence as nc
from follower_eval.asap_eval import BeatTruth


def _truth(perf_times, score_secs, spacing=1.0):
    return BeatTruth(
        perf_times=tuple(perf_times),
        score_secs=tuple(score_secs),
        beat_spacing=tuple([spacing] * len(score_secs)),
    )


def _perf(times_pitches):
    return [
        PerfNote(onset=t, offset=t + 0.2, pitch=p, velocity=64)
        for t, p in times_pitches
    ]


def _score(pos_pitches):
    return [ScoreNote(pitch=p, position=t) for t, p in pos_pitches]


# --- interpolation ----------------------------------------------------------


def test_interp_is_linear_between_anchors():
    assert nc._interp((0.0, 10.0), (0.0, 20.0), 5.0) == pytest.approx(10.0)


def test_interp_clamps_rather_than_extrapolating():
    """A note before the first anchor or after the last must map to the
    endpoint. Extrapolation would invent score time outside the alignment and
    silently fabricate truth there."""
    xs, ys = (5.0, 10.0), (100.0, 200.0)
    assert nc._interp(xs, ys, 0.0) == 100.0
    assert nc._interp(xs, ys, 99.0) == 200.0


# --- truth derivation -------------------------------------------------------


def test_truth_pairs_each_note_with_its_same_pitch_score_note():
    truth = _truth([0.0, 4.0], [0.0, 4.0])
    perf = _perf([(0.0, 60), (1.0, 62), (2.0, 64)])
    score = _score([(0.0, 60), (1.0, 62), (2.0, 64)])
    pairs, no_truth, ambiguous = nc.derive_truth(perf, score, truth)
    assert pairs == {0: 0, 1: 1, 2: 2}
    assert no_truth == 0 and ambiguous == 0


def test_truth_follows_the_alignment_not_the_raw_clock():
    """The performance runs at half score speed. Truth must follow ASAP's beat
    anchors, so perf 2.0s maps to score 1.0s -- a rule that ignored the
    alignment would pair the wrong notes."""
    truth = _truth([0.0, 8.0], [0.0, 4.0])
    perf = _perf([(0.0, 60), (2.0, 62), (4.0, 64)])
    score = _score([(0.0, 60), (1.0, 62), (2.0, 64)])
    pairs, _, _ = nc.derive_truth(perf, score, truth)
    assert pairs == {0: 0, 1: 1, 2: 2}


def test_a_note_absent_from_the_score_has_no_truth():
    """A wrong note is a real category. It must be counted, not forced onto the
    nearest score note of a different pitch."""
    truth = _truth([0.0, 4.0], [0.0, 4.0])
    perf = _perf([(0.0, 60), (1.0, 61)])  # 61 is not in the score
    score = _score([(0.0, 60), (1.0, 62)])
    pairs, no_truth, _ = nc.derive_truth(perf, score, truth)
    assert pairs == {0: 0}
    assert no_truth == 1


def test_a_same_pitch_note_outside_tolerance_has_no_truth():
    truth = _truth([0.0, 20.0], [0.0, 20.0])
    perf = _perf([(0.0, 60)])
    score = _score([(10.0, 60)])  # same pitch, 10 beats away
    pairs, no_truth, _ = nc.derive_truth(perf, score, truth)
    assert pairs == {} and no_truth == 1


def test_two_indistinguishable_candidates_are_excluded_not_guessed():
    """A trill or tremolo puts two same-pitch score notes inside tolerance. The
    rule cannot separate them, so the note is excluded and COUNTED -- silently
    picking one would manufacture truth and flatter whichever choice the
    follower happened to make."""
    truth = _truth([0.0, 4.0], [0.0, 4.0])
    perf = _perf([(1.0, 60)])
    score = _score([(0.95, 60), (1.05, 60)])
    pairs, no_truth, ambiguous = nc.derive_truth(perf, score, truth)
    assert pairs == {} and ambiguous == 1 and no_truth == 0


def test_two_separable_candidates_resolve_to_the_nearer():
    truth = _truth([0.0, 4.0], [0.0, 4.0])
    perf = _perf([(1.0, 60)])
    score = _score([(1.02, 60), (1.45, 60)])  # 0.43 beats apart > AMBIGUITY_BEATS
    pairs, _, ambiguous = nc.derive_truth(perf, score, truth)
    assert pairs == {0: 0} and ambiguous == 0


def test_tolerance_scales_with_the_local_beat_length():
    """ASAP's beat spacing varies; the tolerance is in BEATS, so a slow passage
    must accept a wider absolute window than a fast one."""
    slow = BeatTruth(
        perf_times=(0.0, 8.0), score_secs=(0.0, 8.0), beat_spacing=(4.0, 4.0)
    )
    perf = _perf([(0.0, 60)])
    score = _score([(1.5, 60)])  # 1.5s away: >0.5 beat if beat=1s, <0.5 beat if beat=4s
    assert nc.derive_truth(perf, score, slow)[0] == {0: 0}

    fast = BeatTruth(
        perf_times=(0.0, 8.0), score_secs=(0.0, 8.0), beat_spacing=(1.0, 1.0)
    )
    assert nc.derive_truth(perf, score, fast)[0] == {}


# --- metric arithmetic ------------------------------------------------------


def _clip(correct, predicted, truth_pairs, **kw):
    return nc.ClipCorrespondence(
        asap_piece=kw.get("piece", "x"),
        n_perf_notes=kw.get("n_perf", truth_pairs),
        n_truth_pairs=truth_pairs,
        n_no_truth=0,
        n_ambiguous=0,
        n_predicted=predicted,
        n_correct=correct,
        precision=None,
        recall=None,
        f1=None,
    )


def test_pooled_precision_and_recall_have_different_denominators():
    """Precision is over notes the follower PAIRED; recall is over notes that
    HAVE truth. A follower that pairs only what it is sure of scores high
    precision and low recall, and collapsing the two would hide that."""
    p, r, f = nc._pooled([_clip(correct=50, predicted=50, truth_pairs=100)])
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f == pytest.approx(2 / 3)


def test_bootstrap_resamples_performances_not_notes():
    """A cluster bootstrap over 2 very different performances must produce a
    WIDE interval. A note-level bootstrap over the same 200 notes would report
    a spuriously tight one."""
    clips = [
        _clip(correct=100, predicted=100, truth_pairs=100, piece="a"),
        _clip(correct=0, predicted=100, truth_pairs=100, piece="b"),
    ]
    ci = nc.bootstrap_ci(clips, n=500)["precision"]["ci95"]
    assert ci[0] == pytest.approx(0.0, abs=1e-6)
    assert ci[1] == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_is_deterministic_under_its_seed():
    clips = [
        _clip(correct=90, predicted=100, truth_pairs=100, piece=f"p{i}")
        for i in range(5)
    ]
    assert nc.bootstrap_ci(clips, n=300) == nc.bootstrap_ci(clips, n=300)


def test_bootstrap_raises_on_no_clips():
    with pytest.raises(nc.NoteCorrespondenceError, match="no clips"):
        nc.bootstrap_ci([])
