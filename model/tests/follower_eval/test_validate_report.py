"""Tests for the Track B report's outcome and confidence semantics."""

from __future__ import annotations

import pytest

from follower_eval import validate_report as vr


def _validation(verdict: str, confidence: float | None) -> dict:
    return {
        "piece": "piece",
        "video_id": verdict,
        "verdict": verdict,
        "fraction_wrong": 0.25,
        "follower_confidence": confidence,
    }


def test_summarize_keeps_verdicts_separate_and_uses_resolved_confidence():
    summary = vr.summarize(
        [
            _validation("tracked", 0.8),
            _validation("recovered", 0.3),
            _validation("wrong", 0.9),
            _validation("junk", 0.2),
        ]
    )

    assert summary["verdicts"] == {
        "tracked": 1,
        "recovered": 1,
        "wrong": 1,
        "junk": 1,
    }
    assert "success_frac" not in summary
    assert summary["confidence_outcomes"]["low"] == {
        "tracked": 0,
        "recovered": 1,
        "wrong": 0,
        "junk": 1,
    }
    assert summary["confidence_outcomes"]["high"] == {
        "tracked": 1,
        "recovered": 0,
        "wrong": 1,
        "junk": 0,
    }


def test_summarize_keeps_missing_confidence_visible():
    summary = vr.summarize([_validation("tracked", None)])

    assert summary["confidence_outcomes"]["unscored"]["tracked"] == 1


def test_summarize_rejects_old_validation_without_score_corrected_confidence():
    validation = _validation("tracked", 0.8)
    del validation["follower_confidence"]

    with pytest.raises(vr.ValidateReportError, match="re-save it"):
        vr.summarize([validation])


def test_format_does_not_report_collapsed_success():
    output = vr._format(vr.summarize([_validation("junk", 0.9)]))

    assert "success" not in output.lower()
    assert "high-confidence junk" in output
