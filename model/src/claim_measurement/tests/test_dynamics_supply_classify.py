"""Two-arm dynamics supply + sign-fidelity scorer (#101 / #67) -- pure, no LLM."""
from __future__ import annotations

from claim_measurement.dynamics_supply.classify_supply import (
    arm_summary,
    expected_polarity,
    level_claims,
    sign_fidelity,
    supply_interpretation,
)


def _claim(rid, subtype, polarity="+", location="whole_piece"):
    return {"recording_id": rid, "dynamics_subtype": subtype,
            "polarity": polarity, "location": location}


def test_level_claims_filters_subtype_and_localizes_regions_out():
    claims = [
        _claim("p1", "level", "+"),
        _claim("p1", "contrast", "+"),                       # dropped: contrast
        _claim("p2", "level", "-", {"bar_start": 3, "bar_end": 5}),  # dropped: region
        _claim("p3", "ambiguous"),                           # dropped: ambiguous
    ]
    lvl = level_claims(claims)
    assert [c["recording_id"] for c in lvl] == ["p1"]


def test_arm_summary_counts_perfs_and_polarity():
    claims = [_claim("p1", "level", "+"), _claim("p2", "level", "neutral"),
              _claim("p2", "contrast", "-")]
    s = arm_summary(claims)
    assert s["n_level_whole_piece"] == 2
    assert s["n_perfs_with_level"] == 2
    assert s["level_polarity_histogram"] == {"+": 1, "neutral": 1}
    assert s["subtype_histogram"] == {"level": 2, "contrast": 1}


def test_expected_polarity_maps_cue_labels():
    assert expected_polarity("loud") == "+"
    assert expected_polarity("soft") == "-"
    assert expected_polarity("balanced") == "neutral"


def test_sign_fidelity_matches_polarity_to_cue():
    level_b = [
        _claim("loudperf", "level", "+"),       # cue loud -> + : correct
        _claim("softperf", "level", "+"),        # cue soft -> - : WRONG (teacher inverted)
        _claim("balperf", "level", "neutral"),   # cue balanced -> neutral : correct
        _claim("orphan", "level", "+"),          # no cue label -> skipped
    ]
    cue = {"loudperf": "loud", "softperf": "soft", "balperf": "balanced"}
    f = sign_fidelity(level_b, cue)
    assert f["n_scored"] == 3
    assert f["n_correct"] == 2
    assert f["sign_fidelity"] == 2 / 3
    assert f["n_no_cue"] == 1


def test_supply_interpretation_three_regimes():
    assert supply_interpretation(0, 0).startswith("fundamental")
    assert supply_interpretation(0, 12).startswith("input")     # the promptable-gap result
    assert supply_interpretation(9, 12).startswith("voice")
