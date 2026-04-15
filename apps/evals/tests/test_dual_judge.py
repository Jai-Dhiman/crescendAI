from __future__ import annotations

from teaching_knowledge.scripts.dual_judge import (
    _spearman,
    compute_agreement,
)


def test_spearman_perfect_positive() -> None:
    assert abs(_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9


def test_spearman_perfect_negative() -> None:
    assert abs(_spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) - (-1.0)) < 1e-9


def test_spearman_uncorrelated_roughly_zero() -> None:
    rho = _spearman([1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 9])
    assert -0.5 < rho < 0.95


def test_compute_agreement_groups_by_criterion() -> None:
    judge_a_rows = [
        {"recording_id": "r1", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r2", "judge_dimensions": [
            {"criterion": "Style", "score": 2},
            {"criterion": "Tone", "score": 3},
        ]},
        {"recording_id": "r3", "judge_dimensions": [
            {"criterion": "Style", "score": 1},
            {"criterion": "Tone", "score": 1},
        ]},
        {"recording_id": "r4", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r5", "judge_dimensions": [
            {"criterion": "Style", "score": 0},
            {"criterion": "Tone", "score": 3},
        ]},
    ]
    judge_b_rows = [
        {"recording_id": "r1", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 3},
        ]},
        {"recording_id": "r2", "judge_dimensions": [
            {"criterion": "Style", "score": 2},
            {"criterion": "Tone", "score": 1},
        ]},
        {"recording_id": "r3", "judge_dimensions": [
            {"criterion": "Style", "score": 1},
            {"criterion": "Tone", "score": 2},
        ]},
        {"recording_id": "r4", "judge_dimensions": [
            {"criterion": "Style", "score": 3},
            {"criterion": "Tone", "score": 0},
        ]},
        {"recording_id": "r5", "judge_dimensions": [
            {"criterion": "Style", "score": 0},
            {"criterion": "Tone", "score": 3},
        ]},
    ]

    agreements = compute_agreement(judge_a_rows, judge_b_rows)
    by_name = {a.name: a for a in agreements}

    assert "Style" in by_name
    assert "Tone" in by_name

    style_ag = by_name["Style"]
    assert style_ag.spearman > 0.9
    assert style_ag.trust_level == "high"

    tone_ag = by_name["Tone"]
    assert tone_ag.trust_level in {"uncertain", "low"}


def test_trust_level_thresholds() -> None:
    from teaching_knowledge.scripts.dual_judge import _classify_trust

    assert _classify_trust(0.85) == "high"
    assert _classify_trust(0.7) == "high"
    assert _classify_trust(0.69) == "uncertain"
    assert _classify_trust(0.4) == "uncertain"
    assert _classify_trust(0.39) == "low"
    assert _classify_trust(-0.2) == "low"
