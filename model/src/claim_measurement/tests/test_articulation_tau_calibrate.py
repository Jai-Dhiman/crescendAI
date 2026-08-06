"""Articulation tau calibration (#101 FRONT 10) -- the IOI-floor selection rule.

The rule under test is the one that got this calibration wrong on the first pass:
maximising corpus-spread-over-noise picks the LARGEST floor, because discarding notes
both narrows the measured population and inflates its apparent spread. The shipped rule
minimises substrate error subject to a note-retention bar. """

from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.articulation_tau_calibrate import (
    MINIMUM_PAIR_RETENTION,
    articulation_ratio,
    calibrate,
    conditioning_row,
    duration_ioi_pairs,
)


def _bundle(amt_ratio: float, gt_ratio: float, n: int = 40, ioi: float = 0.2) -> dict:
    def notes(r):
        return [
            {"onset": i * ioi, "offset": i * ioi + r * ioi, "pitch": 60, "velocity": 64}
            for i in range(n)
        ]

    return {
        "video_id": f"w{amt_ratio}",
        "notes": notes(amt_ratio),
        "gt_notes": notes(gt_ratio),
    }


def test_ioi_floor_excludes_pairs_below_it():
    notes = [
        {"onset": 0.0, "offset": 0.5},
        {"onset": 0.01, "offset": 0.5},
        {"onset": 0.5, "offset": 0.7},
    ]
    assert len(duration_ioi_pairs(notes, 0.001)) == 2
    assert len(duration_ioi_pairs(notes, 0.05)) == 1  # the 10ms chord pair drops out


def test_articulation_ratio_none_below_minimum_pairs():
    notes = [{"onset": 0.2 * i, "offset": 0.2 * i + 0.1} for i in range(3)]
    assert articulation_ratio(notes, 0.05) is None


def test_conditioning_row_reports_retention_not_just_error():
    """Retention is the guard that stops the floor sweep from being won by discarding
    notes.
    """
    bundles = [_bundle(1.0, 1.0, n=20, ioi=0.2)]
    row = conditioning_row(bundles, 0.05)
    assert row["pair_retention"] == pytest.approx(1.0)
    assert conditioning_row(bundles, 0.5)["pair_retention"] == 0.0


def test_calibration_rejects_floors_that_discard_too_many_notes():
    """A floor above every IOI must not be chosen even though its error would be
    trivially small -- it is rejected by the retention bar, and with no eligible floor
    the run fails loudly."""
    # all IOIs are 0.5ms -- below even the smallest swept floor -- so no floor retains
    # anything
    bundles = [_bundle(1.0 + 0.1 * i, 1.0, n=40, ioi=0.0005) for i in range(5)]
    with pytest.raises(SystemExit, match="not conditionable"):
        calibrate(bundles)


def test_calibration_picks_the_reference_off_the_amt_statistic():
    """d = AMT statistic - reference, so anchoring the reference to GT would bake the
    substrate's systematic release bias into every measurement (FRONT 8d Cause 1)."""
    bundles = [_bundle(0.6, 1.0), _bundle(0.6, 1.0), _bundle(0.6, 1.0)]
    res = calibrate(bundles)
    assert res["reference_ratio"] == pytest.approx(0.6, abs=1e-6)
    assert res["reference_ratio"] != res["ioi_floor_sweep"][0]["gt_median"]


def test_minimum_pair_retention_is_a_real_bar():
    assert 0.0 < MINIMUM_PAIR_RETENTION <= 1.0
