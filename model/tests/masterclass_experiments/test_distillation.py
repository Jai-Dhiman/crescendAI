"""Tests for LLM distillation pilot."""

import numpy as np
import pytest


def test_build_rubric():
    from masterclass_experiments.distillation import build_rubric

    taxonomy = {
        "dynamics": {
            "description": "Volume control and dynamic contrast",
        },
        "phrasing": {
            "description": "Musical line shape and direction",
        },
    }
    quote_bank = {
        "dynamics": [
            {"feedback_summary": "Too loud", "severity": "significant"},
            {"feedback_summary": "Good crescendo", "severity": "minor"},
        ],
        "phrasing": [
            {"feedback_summary": "No direction", "severity": "critical"},
            {"feedback_summary": "Beautiful line", "severity": "minor"},
        ],
    }

    rubric = build_rubric(taxonomy, quote_bank)

    assert "dynamics" in rubric
    assert "phrasing" in rubric
    # Each rubric entry should have anchors for scale 1-5
    assert "anchors" in rubric["dynamics"]
    assert len(rubric["dynamics"]["anchors"]) == 5


def test_build_scoring_prompt():
    from masterclass_experiments.distillation import build_scoring_prompt

    rubric = {
        "dynamics": {
            "description": "Volume control",
            "anchors": {
                1: "No dynamic variation",
                2: "Minimal dynamics",
                3: "Adequate dynamics",
                4: "Good dynamic range",
                5: "Excellent dynamic control",
            },
        },
    }
    segment_id = "Chopin_Ballade_1_3"

    prompt = build_scoring_prompt(rubric, segment_id)

    assert "dynamics" in prompt
    assert "1 -" in prompt or "1:" in prompt
    assert "5 -" in prompt or "5:" in prompt
    assert segment_id in prompt


def test_parse_scores_valid():
    from masterclass_experiments.distillation import parse_scores

    response = '{"dynamics": 3, "phrasing": 4}'
    dimensions = ["dynamics", "phrasing"]
    scores = parse_scores(response, dimensions)
    assert scores == {"dynamics": 3, "phrasing": 4}


def test_parse_scores_out_of_range_clamped():
    from masterclass_experiments.distillation import parse_scores

    response = '{"dynamics": 7, "phrasing": 0}'
    dimensions = ["dynamics", "phrasing"]
    scores = parse_scores(response, dimensions)
    assert scores["dynamics"] == 5
    assert scores["phrasing"] == 1


def test_calibration_analysis():
    from masterclass_experiments.distillation import calibration_analysis

    rng = np.random.RandomState(42)
    n = 50
    # Teacher scores correlate with composite labels + noise
    composite = rng.rand(n)
    teacher = composite * 4 + 1 + rng.randn(n) * 0.3  # scale to 1-5ish

    result = calibration_analysis(teacher, composite, dim_name="dynamics")

    assert "pearson_r" in result
    assert "mean_offset" in result
    assert "passed_correlation" in result
    # With this construction, correlation should be high
    assert result["pearson_r"] > 0.5


def test_go_no_go_decision():
    from masterclass_experiments.distillation import go_no_go

    per_dim = {
        "dynamics": {"pearson_r": 0.65, "passed_correlation": True},
        "phrasing": {"pearson_r": 0.55, "passed_correlation": True},
        "timing": {"pearson_r": 0.35, "passed_correlation": False},
        "pedaling": {"pearson_r": 0.60, "passed_correlation": True},
        "voicing": {"pearson_r": 0.40, "passed_correlation": False},
    }
    stop_auc = 0.78
    spot_check_accuracy = 0.55

    decision = go_no_go(per_dim, stop_auc, spot_check_accuracy)

    # 3/5 = 60% pass correlation -> meets 60% threshold
    assert decision["go"] is True
    assert decision["dims_passing"] == 3
    assert decision["dims_total"] == 5
