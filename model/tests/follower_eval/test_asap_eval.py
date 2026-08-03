# model/tests/follower_eval/test_asap_eval.py
"""Unit tests for Track A's own logic (ASAP ground-truth eval, #133). The
follower and the ASAP loaders are covered elsewhere; here we pin the beat-error
math, the cold-start windowing, and the deterministic random-start sampling on
constructed inputs with KNOWN answers."""

from __future__ import annotations

import pytest

from follower_bench.follower import MatchedNote
from follower_eval import asap_eval as ae


def _mn(perf_time: float, score_position: float) -> MatchedNote:
    return MatchedNote(
        perf_index=0,
        score_index=0,
        perf_time=perf_time,
        score_position=score_position,
        confidence=0.9,
    )


# perf beats at 0,10,20,30,40s map to score beats 0,2,4,6,8s (perf 5x slower).
TRUTH = ae.BeatTruth(
    perf_times=(0.0, 10.0, 20.0, 30.0, 40.0),
    score_secs=(0.0, 2.0, 4.0, 6.0, 8.0),
    beat_spacing=(2.0, 2.0, 2.0, 2.0, 2.0),
)


def test_beat_errors_zero_when_follower_matches_truth():
    matches = [_mn(0, 0), _mn(10, 2), _mn(20, 4), _mn(30, 6), _mn(40, 8)]
    errs = ae._beat_errors(matches, TRUTH, window=None)
    assert len(errs) == 5
    assert all(abs(sec) < 1e-9 for sec, _ in errs)
    assert all(abs(beats) < 1e-9 for _, beats in errs)


def test_beat_errors_known_half_beat_offset():
    # follower decodes 1.0s high everywhere -> 1.0s / 2.0s spacing = 0.5 beat error
    matches = [_mn(0, 1), _mn(10, 3), _mn(20, 5), _mn(30, 7), _mn(40, 9)]
    errs = ae._beat_errors(matches, TRUTH, window=None)
    assert all(sec == pytest.approx(1.0) for sec, _ in errs)
    assert all(beats == pytest.approx(0.5) for _, beats in errs)


def test_beat_errors_window_filters_beats():
    matches = [_mn(0, 0), _mn(40, 8)]
    errs = ae._beat_errors(matches, TRUTH, window=(15.0, 35.0))
    # only perf beats at 20 and 30 fall in the window
    assert len(errs) == 2


def test_summarize_within_tolerance_rates():
    # 3 beats at 0.2 bar error, 1 at 1.5 -> within-1 = 3/4, within-.5 = 3/4
    errs = [(0.4, 0.2), (0.4, 0.2), (0.4, 0.2), (3.0, 1.5)]
    r = ae._summarize("full", 0.0, errs, transpose=0)
    assert r.n_beats_eval == 4
    assert r.within_1beat_frac == pytest.approx(0.75)
    assert r.within_half_beat_frac == pytest.approx(0.75)
    assert r.median_abs_err_beats == pytest.approx(0.2)


def test_summarize_empty():
    r = ae._summarize("start@5s", 5.0, [], transpose=0)
    assert r.n_beats_eval == 0
    assert r.median_abs_err_beats is None
    assert r.within_1beat_frac is None


def test_rng_starts_deterministic_and_bounded():
    a = ae._rng_starts(TRUTH, n=4, window_sec=10.0, seed=7)
    b = ae._rng_starts(TRUTH, n=4, window_sec=10.0, seed=7)
    assert a == b  # reproducible
    assert len(a) == 4
    assert a == sorted(a)  # returned sorted
    assert all(0.0 <= t <= 30.0 for t in a)  # leaves a 10s window before last beat (40)


def test_rng_starts_no_room_returns_empty():
    # window longer than the whole performance -> no valid start
    assert ae._rng_starts(TRUTH, n=4, window_sec=100.0, seed=0) == []
