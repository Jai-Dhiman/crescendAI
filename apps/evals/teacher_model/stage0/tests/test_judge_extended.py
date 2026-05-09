"""Verify the extended judge parser returns 9 dims and the prompt enumerates all 9."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.judge_extended import (
    EXTENDED_DIMS,
    parse_extended_judge_response,
)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "judge_v2_extended.txt"


def test_extended_dims_lists_nine_criteria_in_canonical_order() -> None:
    expected = [
        "Audible-Specific Corrective Feedback",
        "Concrete Artifact Provision",
        "Specific Positive Praise",
        "Autonomy-Supporting Motivation",
        "Scaffolded Guided Discovery",
        "Style-Consistent Musical Language",
        "Appropriate Tone & Language",
        "Taste Defensibility",
        "Adaptation Specificity",
    ]
    assert EXTENDED_DIMS == expected


def test_prompt_file_mentions_all_nine_dimensions() -> None:
    text = _PROMPT_PATH.read_text()
    for dim in EXTENDED_DIMS:
        assert dim in text, f"prompt missing dimension: {dim!r}"


def test_parser_returns_nine_dimension_scores() -> None:
    response_payload = json.dumps([
        {"criterion": dim, "process": 2, "outcome": 2, "evidence": "e", "reason": "r"}
        for dim in [
            "Audible-Specific Corrective Feedback",
            "Concrete Artifact Provision",
            "Specific Positive Praise",
            "Autonomy-Supporting Motivation",
            "Scaffolded Guided Discovery",
            "Style-Consistent Musical Language",
            "Appropriate Tone & Language",
            "Taste Defensibility",
            "Adaptation Specificity",
        ]
    ])
    dims = parse_extended_judge_response(response_payload)
    assert len(dims) == 9
    assert [d.criterion for d in dims] == EXTENDED_DIMS
    for d in dims:
        assert d.process == 2 and d.outcome == 2


def test_parser_handles_na_strings() -> None:
    payload = json.dumps([
        {"criterion": dim, "process": "N/A", "outcome": "N/A", "evidence": "", "reason": "n/a"}
        for dim in [
            "Audible-Specific Corrective Feedback",
            "Concrete Artifact Provision",
            "Specific Positive Praise",
            "Autonomy-Supporting Motivation",
            "Scaffolded Guided Discovery",
            "Style-Consistent Musical Language",
            "Appropriate Tone & Language",
            "Taste Defensibility",
            "Adaptation Specificity",
        ]
    ])
    dims = parse_extended_judge_response(payload)
    assert all(d.process is None and d.outcome is None for d in dims)
