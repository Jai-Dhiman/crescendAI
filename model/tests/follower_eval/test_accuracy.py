# model/tests/follower_eval/test_accuracy.py
"""Unit tests for the gold (accuracy) track (issue #133 S3). Pins the tap<->score
comparison math on constructed trajectories with KNOWN answers -- the HMM itself
is covered by follower_bench tests; here we prove the localization error, the
within-tolerance rate, restart detection, and relock latency are computed right,
and that mislabels/empties fail loudly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from follower_bench.follower import EstimatedTrajectory, MatchedNote

from follower_eval import accuracy as acc
from follower_eval.realaudio import RealAudioEvalError


def _mn(perf_time: float, score_position: float) -> MatchedNote:
    return MatchedNote(perf_index=0, score_index=0, perf_time=perf_time,
                       score_position=score_position, confidence=0.9)


# 5 bars, 2s each, in score-render seconds -> bar N starts at 2*(N-1).
MEASURE_TABLE = [{"bar_number": n, "start_sec": 2.0 * (n - 1), "start_tick": 0} for n in range(1, 6)]


def test_bar_second_table_starts_and_durations():
    starts, durs = acc.bar_second_table(MEASURE_TABLE)
    assert starts == {1: 0.0, 2: 2.0, 3: 4.0, 4: 6.0, 5: 8.0}
    assert durs[1] == pytest.approx(2.0)
    assert durs[5] == pytest.approx(2.0)   # last bar reuses the prior span


def test_decode_at_clamps_and_interpolates():
    pt = [0.0, 10.0, 20.0]
    sp = [0.0, 2.0, 4.0]
    assert acc.decode_at(pt, sp, -5.0) == 0.0     # before first match -> clamp low
    assert acc.decode_at(pt, sp, 30.0) == 4.0     # after last match -> clamp high
    assert acc.decode_at(pt, sp, 5.0) == pytest.approx(1.0)   # halfway 0->10 -> score 1.0
    assert acc.decode_at(pt, sp, 15.0) == pytest.approx(3.0)
    assert acc.decode_at([], [], 5.0) is None


def test_evaluate_taps_zero_error_when_follower_tracks_perfectly():
    # follower maps audio t -> score t/5 (performer 5x slower than score render).
    matches = [_mn(0, 0), _mn(10, 2), _mn(20, 4), _mn(30, 6), _mn(40, 8)]
    taps = [acc.BarTap(n, 10.0 * (n - 1)) for n in range(1, 6)]
    starts, durs = acc.bar_second_table(MEASURE_TABLE)
    errs = acc.evaluate_taps(taps, matches, starts, durs)
    assert [round(e.abs_err_sec, 6) for e in errs] == [0.0] * 5
    assert all(e.abs_err_bars == 0.0 for e in errs)
    assert not any(e.is_restart for e in errs)


def test_evaluate_taps_known_half_bar_error():
    matches = [_mn(0, 0), _mn(10, 2)]
    # tap bar 1 at audio 5s: decoded score = 1.0, true = 0.0 -> 1.0s error = 0.5 bar
    taps = [acc.BarTap(1, 5.0)]
    starts, durs = acc.bar_second_table(MEASURE_TABLE)
    errs = acc.evaluate_taps(taps, matches, starts, durs)
    assert errs[0].abs_err_sec == pytest.approx(1.0)
    assert errs[0].abs_err_bars == pytest.approx(0.5)


def test_evaluate_taps_flags_restart():
    matches = [_mn(0, 0), _mn(40, 8)]
    taps = [acc.BarTap(1, 0), acc.BarTap(2, 5), acc.BarTap(3, 10),
            acc.BarTap(1, 15), acc.BarTap(2, 20)]  # bar drops 3->1 = restart
    starts, durs = acc.bar_second_table(MEASURE_TABLE)
    errs = acc.evaluate_taps(taps, matches, starts, durs)
    assert [e.is_restart for e in errs] == [False, False, False, True, False]


def test_evaluate_taps_loud_on_unknown_bar():
    starts, durs = acc.bar_second_table(MEASURE_TABLE)
    with pytest.raises(RealAudioEvalError, match="not in score measure_table"):
        acc.evaluate_taps([acc.BarTap(99, 1.0)], [_mn(0, 0), _mn(10, 2)], starts, durs)


def test_relock_latency_and_no_relock():
    # Build TapErrors by hand: a restart at tap idx 1 that relocks 2 taps later.
    def te(bar, t, err_bars, restart):
        return acc.TapError(bar_number=bar, audio_sec=t, true_score_sec=0.0,
                            decoded_score_sec=0.0, abs_err_sec=err_bars * 2.0,
                            abs_err_bars=err_bars, local_bar_sec=2.0, is_restart=restart)
    # restart at t=10 with big error 3.0 bars; back within 1 bar at t=16
    errs = [te(3, 5, 0.1, False), te(1, 10, 3.0, True), te(2, 13, 2.0, False),
            te(3, 16, 0.4, False)]
    latencies, no_relock = acc._relock(errs, tol_bars=1.0)
    assert latencies == [pytest.approx(6.0)]   # 16 - 10
    assert no_relock == 0

    # a restart that never comes back within tolerance
    errs2 = [te(3, 5, 0.1, False), te(1, 10, 3.0, True), te(2, 13, 2.5, False)]
    latencies2, no_relock2 = acc._relock(errs2, tol_bars=1.0)
    assert latencies2 == []
    assert no_relock2 == 1

    # a restart already within tolerance -> latency 0
    errs3 = [te(3, 5, 0.1, False), te(1, 10, 0.3, True)]
    latencies3, no_relock3 = acc._relock(errs3, tol_bars=1.0)
    assert latencies3 == [pytest.approx(0.0)]


def test_pctl():
    assert acc._pctl([1.0], 0.9) == 1.0
    assert acc._pctl([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert acc._pctl([0.0, 1.0, 2.0, 3.0, 4.0], 0.9) == pytest.approx(3.6)


def test_load_gold_loud_on_empty(tmp_path: Path):
    p = tmp_path / "x.gold.json"
    p.write_text(json.dumps({"bar_taps": []}))
    with pytest.raises(RealAudioEvalError, match="no 'bar_taps'"):
        acc.load_gold(p)


def test_load_gold_sorts_by_audio_sec(tmp_path: Path):
    p = tmp_path / "x.gold.json"
    p.write_text(json.dumps({"bar_taps": [
        {"bar_number": 2, "audio_sec": 10.0},
        {"bar_number": 1, "audio_sec": 3.0},
    ]}))
    taps = acc.load_gold(p)
    assert [t.audio_sec for t in taps] == [3.0, 10.0]
    assert [t.bar_number for t in taps] == [1, 2]


def test_evaluate_clip_end_to_end(tmp_path: Path, monkeypatch):
    """Full evaluate_clip: real bundle + gold loaders, follow_hmm stubbed to a
    known trajectory so the asserted numbers are exact."""
    bundle = tmp_path / "vid.json"
    bundle.write_text(json.dumps({"notes": [
        {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 50},
        {"onset": 40.0, "offset": 40.5, "pitch": 62, "velocity": 50},
    ]}))
    gold = tmp_path / "vid.gold.json"
    gold.write_text(json.dumps({"bar_taps": [
        {"bar_number": n, "audio_sec": 10.0 * (n - 1)} for n in range(1, 6)
    ]}))
    traj = EstimatedTrajectory(
        transpose_semitones=0,
        matches=tuple(_mn(10.0 * (n - 1), 2.0 * (n - 1)) for n in range(1, 6)),
        unmatched_perf_indices=(),
    )
    monkeypatch.setattr(acc, "follow_hmm", lambda *a, **k: traj)

    res = acc.evaluate_clip("bach_prelude_c_wtc1", bundle, gold,
                            score_notes=[], bar_boundaries=(), measure_table=MEASURE_TABLE)
    assert res.n_taps == 5
    assert res.n_decoded == 5
    assert res.median_abs_err_sec == 0.0
    assert res.within_1bar_frac == 1.0
    assert res.within_half_bar_frac == 1.0
    assert res.n_restarts == 0
