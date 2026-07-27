"""Pedaling oracle pure functions (#101 FRONT 9) -- GT polarity + AMT reference recalibration."""
from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.pedaling_independent_rate import (
    amt_corpus_reference,
    gt_pedal_polarity,
)


def test_gt_pedal_polarity_thresholds():
    # median 0.5, tau 0.25 -> over above 0.75, under below 0.25
    assert gt_pedal_polarity(0.9, 0.5, 0.25) == "+"
    assert gt_pedal_polarity(0.1, 0.5, 0.25) == "-"
    assert gt_pedal_polarity(0.5, 0.5, 0.25) == "neutral"
    assert gt_pedal_polarity(0.74, 0.5, 0.25) == "neutral"  # just inside the deadband


def test_amt_corpus_reference_is_median_on_fraction():
    # bundle A: pedal down [0,5] of 10s window -> on_fraction 0.5
    # bundle B: pedal down whole window from a prior-held state -> 1.0
    a = {"duration_sec": 10.0, "pedal_events": [{"time": 0.0, "value": 127}, {"time": 5.0, "value": 0}]}
    b = {"duration_sec": 10.0, "pedal_events": [{"time": 0.0, "value": 127}]}
    # median of [0.5, 1.0] = 0.75
    assert amt_corpus_reference([a, b]) == pytest.approx(0.75)


def test_amt_corpus_reference_empty_raises():
    with pytest.raises(ValueError, match="no bundles contain a measurable"):
        amt_corpus_reference([{"duration_sec": 0.0, "pedal_events": []}])
