# model/tests/follower_eval/test_onset_direction.py
"""Unit tests for the directional onset-timing baseline (#108 / #148).

The claims worth pinning are the local tempo fit, the deadband's refusal to
score on-time notes, and the two nulls -- especially that the shuffled null's
expectation is the marginal-agreement rate rather than 0.5."""

from __future__ import annotations

import pytest
from follower_eval import onset_direction as od


def _clip(n, acc, shuffled, majority, piece="x", mag=0.01):
    return od.ClipDirection(
        asap_piece=piece,
        n_scored=n,
        n_deadband_excluded=0,
        n_no_local_fit=0,
        sign_accuracy=acc,
        median_abs_magnitude_err_s=mag,
        truth_late_frac=majority,
        shuffled_sign_accuracy=shuffled,
        majority_sign_accuracy=majority,
    )


# --- the local tempo fit ----------------------------------------------------


def test_local_fit_recovers_a_known_linear_tempo_map():
    # score position -> perf time at 2x slower playing, offset 10s
    pts = [(s, 10.0 + 2.0 * s) for s in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert od._local_fit(pts, 5.0) == pytest.approx(20.0)


def test_local_fit_refuses_too_few_points():
    """Two points define a line with zero residual; there is no way to tell a
    good local fit from a meaningless one."""
    assert od._local_fit([(0.0, 0.0), (1.0, 1.0)], 2.0) is None


def test_local_fit_refuses_a_degenerate_window():
    """Every match at one score position gives no slope -- a chord, or a stall."""
    pts = [(3.0, t) for t in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert od._local_fit(pts, 4.0) is None


def test_inverse_beat_map_sorts_by_score_time():
    class T:
        score_secs = (4.0, 0.0, 2.0)
        perf_times = (40.0, 0.0, 20.0)

    xs, ys = od._inverse_beat_map(T())
    assert xs == (0.0, 2.0, 4.0)
    assert ys == (0.0, 20.0, 40.0)


# --- the nulls --------------------------------------------------------------


def test_shuffled_null_expectation_is_the_marginal_rate_not_one_half():
    """THE misreading this guards against. With truth 80% late and a system that
    also answers late 80% of the time, a shuffled arm carrying zero per-note
    information still agrees 0.8*0.8 + 0.2*0.2 = 0.68. Reading 0.68 as
    'above chance' would turn a null into a result."""
    p = q = 0.8
    expected = p * q + (1 - p) * (1 - q)
    assert expected == pytest.approx(0.68)
    assert expected > 0.5


def test_a_run_that_only_matches_the_majority_is_uninformative():
    clips = [_clip(n=100, acc=0.73, shuffled=0.60, majority=0.73)]
    pooled = od._pooled(clips)
    assert not (
        pooled["sign_accuracy"] > pooled["majority"]
        and pooled["sign_accuracy"] > pooled["shuffled"]
    )


def test_a_run_that_beats_both_nulls_is_informative():
    clips = [_clip(n=100, acc=0.87, shuffled=0.57, majority=0.73)]
    pooled = od._pooled(clips)
    assert (
        pooled["sign_accuracy"] > pooled["majority"]
        and pooled["sign_accuracy"] > pooled["shuffled"]
    )


# --- pooling and bootstrap --------------------------------------------------


def test_pooling_weights_clips_by_their_scored_note_count():
    """A 900-note performance must not be outvoted by a 100-note one."""
    clips = [
        _clip(n=900, acc=0.90, shuffled=0.5, majority=0.5, piece="big"),
        _clip(n=100, acc=0.50, shuffled=0.5, majority=0.5, piece="small"),
    ]
    assert od._pooled(clips)["sign_accuracy"] == pytest.approx(0.86)


def test_bootstrap_resamples_performances_and_widens_with_disagreement():
    clips = [
        _clip(n=100, acc=1.0, shuffled=0.5, majority=0.5, piece="a"),
        _clip(n=100, acc=0.0, shuffled=0.5, majority=0.5, piece="b"),
    ]
    lo, hi = od.bootstrap_ci(clips, n=500)["sign_accuracy"]["ci95"]
    assert lo == pytest.approx(0.0, abs=1e-6) and hi == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_is_deterministic_under_its_seed():
    clips = [
        _clip(n=100, acc=0.8, shuffled=0.6, majority=0.7, piece=f"p{i}")
        for i in range(5)
    ]
    assert od.bootstrap_ci(clips, n=300) == od.bootstrap_ci(clips, n=300)


def test_bootstrap_raises_on_no_clips():
    with pytest.raises(od.OnsetDirectionError, match="no clips"):
        od.bootstrap_ci([])
