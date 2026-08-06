"""Articulation oracle pure functions (#101 FRONT 10) -- GT polarity + reference
recalibration.
"""

from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.articulation_independent_rate import (
    amt_corpus_reference,
    gt_articulation_polarity,
    gt_corpus_reference,
    gt_ratio,
)


def _notes(ratio: float, n: int = 10, ioi: float = 0.2, key: str = "notes") -> dict:
    return {
        key: [
            {
                "onset": i * ioi,
                "offset": i * ioi + ratio * ioi,
                "pitch": 60,
                "velocity": 64,
            }
            for i in range(n)
        ]
    }


def test_gt_articulation_polarity_thresholds():
    # corpus median 1.0, tau 0.163 -> legato above 1.163, detached below 0.837
    assert gt_articulation_polarity(1.5, 1.0, 0.163) == "+"
    assert gt_articulation_polarity(0.5, 1.0, 0.163) == "-"
    assert gt_articulation_polarity(1.0, 1.0, 0.163) == "neutral"
    assert (
        gt_articulation_polarity(1.16, 1.0, 0.163) == "neutral"
    )  # just inside the deadband


def test_gt_ratio_reads_gt_notes_not_amt_notes():
    """The oracle is only non-circular if truth comes from gt_notes. A bundle whose AMT
    and GT disagree must yield the GT value, or the rate silently scores the substrate
    against itself."""
    bundle = {**_notes(0.4, key="notes"), **_notes(1.9, key="gt_notes")}
    assert gt_ratio(bundle) == pytest.approx(1.9, abs=1e-6)


def test_amt_and_gt_corpus_references_are_window_medians():
    bundles = [
        {**_notes(0.5, key="notes"), **_notes(0.9, key="gt_notes")},
        {**_notes(1.5, key="notes"), **_notes(1.1, key="gt_notes")},
    ]
    assert amt_corpus_reference(bundles) == pytest.approx(1.0)  # median of [0.5, 1.5]
    assert gt_corpus_reference(bundles) == pytest.approx(1.0)  # median of [0.9, 1.1]


def test_corpus_reference_raises_when_no_bundle_is_measurable():
    # every IOI below the 50ms floor -> no ratios -> explicit failure, never a silent
    # default
    unmeasurable = [_notes(1.0, n=10, ioi=0.01)]
    with pytest.raises(ValueError, match="no bundle yields a measurable AMT"):
        amt_corpus_reference(unmeasurable)
