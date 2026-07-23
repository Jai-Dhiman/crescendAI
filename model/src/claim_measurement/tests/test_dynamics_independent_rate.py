"""GT-anchored independent dynamics rate (#101 FRONT 8b) -- pure aggregation logic, no verify()."""
from __future__ import annotations

from claim_measurement.dynamics_supply.independent_rate import (
    aggregate,
    bootstrap_ci,
    gt_polarity,
)


def _rec(seg, gt_label, gt_pol, verdict, committed=True, reason="ok"):
    return {"segment": seg, "gt_label": gt_label, "gt_polarity": gt_pol,
            "verdict": verdict, "committed": committed, "reason": reason}


def test_gt_polarity_uses_gt_median_and_deadband():
    assert gt_polarity(80.0, 60.0, 6.5) == "+"      # +20 > tau
    assert gt_polarity(40.0, 60.0, 6.5) == "-"      # -20 < -tau
    assert gt_polarity(63.0, 60.0, 6.5) == "neutral"  # +3 within deadband
    assert gt_polarity(66.5, 60.0, 6.5) == "neutral"  # boundary inclusive-neutral


def test_aggregate_rate_confusion_and_histogram():
    records = [
        _rec("s1", "loud", "+", "SUPPORTED"),
        _rec("s2", "loud", "+", "REFUTED"),                     # AMT disagreed w/ GT
        _rec("s3", "balanced", "neutral", "SUPPORTED"),
        _rec("s4", "soft", "-", "UNVERIFIABLE", committed=False, reason="near_threshold"),
    ]
    agg = aggregate(records)
    assert agg["n_committed"] == 3
    assert agg["n_supported"] == 2
    assert agg["faithfulness_rate"] == 2 / 3
    assert agg["confusion_gt_label_x_verdict"]["loud->SUPPORTED"] == 1
    assert agg["confusion_gt_label_x_verdict"]["loud->REFUTED"] == 1
    assert agg["confusion_gt_label_x_verdict"]["soft->ABSTAIN:near_threshold"] == 1
    assert agg["abstention_histogram"] == {"near_threshold": 1}
    assert agg["polarity_breakdown"]["+"] == {"supported": 1, "refuted": 1}


def test_aggregate_all_abstain_is_unmeasurable():
    records = [_rec("s1", "balanced", "neutral", "UNVERIFIABLE", committed=False, reason="x")]
    agg = aggregate(records)
    assert agg["n_committed"] == 0
    assert agg["faithfulness_rate"] is None


def test_bootstrap_ci_degenerate_all_supported():
    committed = [_rec(f"s{i}", "loud", "+", "SUPPORTED") for i in range(20)]
    ci = bootstrap_ci(committed, n_boot=200)
    assert ci["lo"] == 1.0 and ci["hi"] == 1.0 and ci["half_width"] == 0.0


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n_boot=100)
    assert ci["half_width"] is None
