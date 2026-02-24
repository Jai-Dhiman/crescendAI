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


def test_build_rubric_positive_uses_praise():
    """Positive anchor should use praise quotes, not require minor+praise."""
    from masterclass_experiments.distillation import build_rubric

    taxonomy = {"dynamics": {"description": "Dynamic control"}}
    quote_bank = {
        "dynamics": [
            {"feedback_summary": "Wonderful crescendo", "severity": "moderate", "feedback_type": "praise"},
            {"feedback_summary": "Needs work", "severity": "significant", "feedback_type": "correction"},
        ],
    }

    rubric = build_rubric(taxonomy, quote_bank)
    # Should pick the praise quote, not fall back to "Excellent quality"
    assert "Wonderful crescendo" in rubric["dynamics"]["anchors"][5]


def test_build_rubric_positive_uses_minor_severity():
    """Positive anchor should also accept minor/moderate severity even without praise type."""
    from masterclass_experiments.distillation import build_rubric

    taxonomy = {"dynamics": {"description": "Dynamic control"}}
    quote_bank = {
        "dynamics": [
            {"feedback_summary": "Slight timing issue", "severity": "minor", "feedback_type": "suggestion"},
            {"feedback_summary": "Very poor dynamics", "severity": "critical", "feedback_type": "correction"},
        ],
    }

    rubric = build_rubric(taxonomy, quote_bank)
    # Minor severity should qualify as positive (dimension done relatively well)
    assert "Slight timing issue" in rubric["dynamics"]["anchors"][5]


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


def test_build_scoring_prompt_with_context():
    from masterclass_experiments.distillation import build_scoring_prompt

    rubric = {
        "dynamics": {
            "description": "Volume control",
            "anchors": {1: "Poor", 2: "Below avg", 3: "Adequate", 4: "Good", 5: "Excellent"},
        },
    }
    context = "Piece: Ballade No.1 by Chopin\nTeacher feedback for this segment:\n  1. [dynamics, significant] Too loud"

    prompt = build_scoring_prompt(rubric, "seg_1", context=context)

    assert "Performance context:" in prompt
    assert "Ballade No.1 by Chopin" in prompt
    assert "Too loud" in prompt
    # Context should appear before the rubric dimensions
    ctx_pos = prompt.index("Performance context:")
    dim_pos = prompt.index("## dynamics")
    assert ctx_pos < dim_pos


def test_build_scoring_prompt_empty_context():
    """Empty context string should not add context section."""
    from masterclass_experiments.distillation import build_scoring_prompt

    rubric = {
        "dynamics": {
            "description": "Volume control",
            "anchors": {1: "Poor", 2: "Below avg", 3: "Adequate", 4: "Good", 5: "Excellent"},
        },
    }
    prompt = build_scoring_prompt(rubric, "seg_1", context="")
    assert "Performance context:" not in prompt


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


def test_build_moment_context():
    from masterclass_experiments.distillation import build_moment_context

    moments = [
        {
            "piece": "Ballade No.1",
            "composer": "Chopin",
            "musical_dimension": "dynamics",
            "severity": "significant",
            "feedback_summary": "Too loud in the climax",
        },
        {
            "piece": "Ballade No.1",
            "composer": "Chopin",
            "musical_dimension": "phrasing",
            "severity": "minor",
            "feedback_summary": "Good overall line",
        },
    ]

    ctx = build_moment_context(moments)

    assert "Ballade No.1" in ctx
    assert "Chopin" in ctx
    assert "1. [dynamics, significant] Too loud in the climax" in ctx
    assert "2. [phrasing, minor] Good overall line" in ctx


def test_build_moment_context_empty():
    from masterclass_experiments.distillation import build_moment_context

    assert build_moment_context([]) == ""


def test_compute_spot_check():
    from masterclass_experiments.distillation import compute_spot_check

    # LLM gave dynamics the lowest score for seg_1
    llm_scores = {
        "seg_1": {"dynamics": 1, "timing": 3, "phrasing": 4},
        "seg_2": {"dynamics": 4, "timing": 2, "phrasing": 3},
    }
    # seg_1 has a dynamics moment -> match; seg_2 has articulation -> no match
    segment_moments = {
        "seg_1": [{"musical_dimension": "dynamics"}],
        "seg_2": [{"musical_dimension": "articulation"}],
    }
    raw_to_taxonomy = {
        "dynamics": "dynamics",
        "timing": "timing",
        "articulation": "articulation",
    }

    accuracy = compute_spot_check(llm_scores, segment_moments, raw_to_taxonomy)
    assert accuracy == 0.5  # 1 match out of 2


def test_compute_spot_check_technique_excluded():
    """Moments with technique-only dims should be skipped (no taxonomy mapping)."""
    from masterclass_experiments.distillation import compute_spot_check

    llm_scores = {"seg_1": {"dynamics": 2, "timing": 3}}
    segment_moments = {"seg_1": [{"musical_dimension": "technique"}]}
    raw_to_taxonomy = {"dynamics": "dynamics", "timing": "timing"}

    # No valid taxonomy dims -> segment skipped, 0 checked
    accuracy = compute_spot_check(llm_scores, segment_moments, raw_to_taxonomy)
    assert accuracy == 0.0


def test_compute_spot_check_multi_moment_match():
    """Segment with multiple moments: match if any moment's dim matches lowest."""
    from masterclass_experiments.distillation import compute_spot_check

    llm_scores = {"seg_1": {"dynamics": 1, "timing": 4, "phrasing": 3}}
    segment_moments = {
        "seg_1": [
            {"musical_dimension": "timing"},
            {"musical_dimension": "dynamics"},
        ],
    }
    raw_to_taxonomy = {"dynamics": "dynamics", "timing": "timing"}

    accuracy = compute_spot_check(llm_scores, segment_moments, raw_to_taxonomy)
    assert accuracy == 1.0
