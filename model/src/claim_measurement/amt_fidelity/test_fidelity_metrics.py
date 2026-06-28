"""Unit tests for the pure AMT-fidelity metrics core.

Dependency-free (hand-rolled stats) so they run under any python3 with no model
or scipy install. Run:  python3 test_fidelity_metrics.py   (or via pytest).

These TDD the two things the orchestration shell must NOT get wrong: greedy
AMT<->GT note matching (pitch-exact, nearest-onset, within-window, unique) and
the onset / duration fidelity reductions that gate timing and articulation.
"""
import math

from fidelity_metrics import (
    spearman,
    match_notes,
    onset_fidelity,
    duration_fidelity,
)


def _note(pitch, onset, offset, velocity=64):
    return {"pitch": pitch, "onset": onset, "offset": offset, "velocity": velocity}


# ---- spearman ---------------------------------------------------------------

def test_spearman_monotone_increasing():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9


def test_spearman_monotone_decreasing():
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) - (-1.0)) < 1e-9


def test_spearman_partial_known_value():
    # ranks x=[1,2,3], y=[1,3,2] -> pearson on ranks = 0.5 (computed by hand)
    assert abs(spearman([10, 20, 30], [10, 30, 20]) - 0.5) < 1e-9


def test_spearman_handles_ties_without_crash():
    r = spearman([1, 1, 2, 3], [2, 2, 3, 4])
    assert -1.0 <= r <= 1.0


# ---- match_notes ------------------------------------------------------------

def test_match_identical_notes_all_pair():
    gt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0), _note(64, 1.0, 1.5)]
    amt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0), _note(64, 1.0, 1.5)]
    res = match_notes(gt, amt)
    assert res["n_matched"] == 3
    assert res["recall"] == 1.0


def test_match_tolerates_small_onset_shift():
    gt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0)]
    amt = [_note(60, 0.02, 0.52), _note(62, 0.52, 1.02)]  # +20ms shift
    res = match_notes(gt, amt, onset_window_s=0.1)
    assert res["n_matched"] == 2


def test_match_excludes_out_of_window():
    gt = [_note(60, 0.0, 0.5)]
    amt = [_note(60, 0.3, 0.8)]  # 300ms off, outside 100ms window
    res = match_notes(gt, amt, onset_window_s=0.1)
    assert res["n_matched"] == 0


def test_match_requires_exact_pitch():
    gt = [_note(60, 0.0, 0.5)]
    amt = [_note(61, 0.0, 0.5)]  # wrong pitch
    res = match_notes(gt, amt)
    assert res["n_matched"] == 0


def test_match_counts_dropped_note_as_recall_miss():
    gt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0), _note(64, 1.0, 1.5)]
    amt = [_note(60, 0.0, 0.5), _note(64, 1.0, 1.5)]  # 62 dropped
    res = match_notes(gt, amt)
    assert res["n_matched"] == 2
    assert abs(res["recall"] - 2.0 / 3.0) < 1e-9


def test_match_unique_amt_use():
    # two GT notes same pitch close together, only one AMT note -> only one matches
    gt = [_note(60, 0.00, 0.05), _note(60, 0.04, 0.09)]
    amt = [_note(60, 0.02, 0.07)]
    res = match_notes(gt, amt, onset_window_s=0.1)
    assert res["n_matched"] == 1


# ---- onset_fidelity ---------------------------------------------------------

def test_onset_fidelity_constant_bias():
    gt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0), _note(64, 1.0, 1.5)]
    amt = [_note(60, 0.03, 0.53), _note(62, 0.53, 1.03), _note(64, 1.03, 1.53)]
    pairs = match_notes(gt, amt)["pairs"]
    f = onset_fidelity(pairs)
    assert abs(f["bias_ms"] - 30.0) < 1e-6   # +30ms systematic latency
    assert f["noise_ms"] < 1e-6              # zero scatter around the bias
    assert abs(f["median_abs_ms"] - 30.0) < 1e-6


def test_onset_fidelity_scatter():
    # errors of -10, 0, +10 ms -> bias 0, population std = sqrt(200/3) ~ 8.16ms
    gt = [_note(60, 0.0, 0.5), _note(62, 0.5, 1.0), _note(64, 1.0, 1.5)]
    amt = [_note(60, -0.01, 0.49), _note(62, 0.5, 1.0), _note(64, 1.01, 1.51)]
    pairs = match_notes(gt, amt)["pairs"]
    f = onset_fidelity(pairs)
    assert abs(f["bias_ms"] - 0.0) < 1e-6
    assert abs(f["noise_ms"] - math.sqrt(200.0 / 3.0)) < 1e-6


# ---- duration_fidelity ------------------------------------------------------

def test_duration_fidelity_half_length():
    gt = [_note(60, 0.0, 1.0), _note(62, 1.0, 2.0), _note(64, 2.0, 3.0)]
    # AMT halves every duration (offset pulled in)
    amt = [_note(60, 0.0, 0.5), _note(62, 1.0, 1.5), _note(64, 2.0, 2.5)]
    pairs = match_notes(gt, amt)["pairs"]
    f = duration_fidelity(pairs)
    assert abs(f["median_ratio"] - 0.5) < 1e-9


def test_duration_fidelity_spearman_preserves_order():
    # GT durations 0.2,0.4,0.6 ; AMT scaled but order-preserving -> spearman 1.0
    gt = [_note(60, 0.0, 0.2), _note(62, 1.0, 1.4), _note(64, 2.0, 2.6)]
    amt = [_note(60, 0.0, 0.25), _note(62, 1.0, 1.45), _note(64, 2.0, 2.65)]
    pairs = match_notes(gt, amt)["pairs"]
    f = duration_fidelity(pairs)
    assert abs(f["spearman_dur"] - 1.0) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner surfaces all
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
