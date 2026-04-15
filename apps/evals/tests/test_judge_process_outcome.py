from __future__ import annotations

import json

from shared.judge import DimensionScore, _parse_v2_response


def test_dimension_score_has_process_and_outcome_fields() -> None:
    d = DimensionScore(
        criterion="Audible-Specific Corrective Feedback",
        score=2,
        evidence="ok",
        reason="fine",
        process=3,
        outcome=2,
    )
    assert d.process == 3
    assert d.outcome == 2


def test_parse_legacy_single_score_response() -> None:
    legacy = json.dumps([
        {
            "criterion": "Specific Positive Praise",
            "score": 2,
            "evidence": "You nailed the crescendo.",
            "reason": "Concrete and warm.",
        }
    ])
    dims = _parse_v2_response(legacy)
    assert len(dims) == 1
    d = dims[0]
    assert d.score == 2
    assert d.process == 2
    assert d.outcome == 2


def test_parse_new_process_outcome_response() -> None:
    new = json.dumps([
        {
            "criterion": "Audible-Specific Corrective Feedback",
            "process": 3,
            "outcome": 1,
            "evidence": "bar 12",
            "reason": "noticed but incorrect",
        }
    ])
    dims = _parse_v2_response(new)
    assert len(dims) == 1
    d = dims[0]
    assert d.process == 3
    assert d.outcome == 1
    # composite score = min of the two signals (a conservative composite)
    assert d.score == 1


def test_parse_new_schema_with_na_process_or_outcome() -> None:
    new = json.dumps([
        {
            "criterion": "Autonomy-Supporting Motivation",
            "process": "N/A",
            "outcome": "N/A",
            "evidence": "",
            "reason": "Not applicable.",
        }
    ])
    dims = _parse_v2_response(new)
    assert dims[0].process is None
    assert dims[0].outcome is None
    assert dims[0].score is None


def test_parse_new_schema_with_only_process_na() -> None:
    new = json.dumps([
        {
            "criterion": "Concrete Artifact Provision",
            "process": "N/A",
            "outcome": 2,
            "evidence": "",
            "reason": "Praise-only, no artifact expected.",
        }
    ])
    dims = _parse_v2_response(new)
    assert dims[0].process is None
    assert dims[0].outcome == 2
    # When one side is N/A, composite = the other side
    assert dims[0].score == 2


def test_parse_failure_still_returns_one_dimension() -> None:
    dims = _parse_v2_response("not json at all")
    assert len(dims) == 1
    assert dims[0].criterion == "parse_failure"
