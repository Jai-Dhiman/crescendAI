"""Format extraction + schema validation across the 3 tolerated tool-call shapes."""
from __future__ import annotations

import pytest

from teacher_model.stage0.tool_scorer import (
    ToolCase,
    ToolProbeResult,
    score_response,
)

_SCHEMAS = {
    "search_catalog": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "create_exercise": {
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "exercises": {"type": "array"},
        },
        "required": ["skill", "exercises"],
    },
}


def _pos_case(tool: str = "search_catalog") -> ToolCase:
    return ToolCase(case_id="p1", expected_call=True, expected_tool=tool, category=None)


def _neg_case(category: str = "chitchat") -> ToolCase:
    return ToolCase(case_id="n1", expected_call=False, expected_tool=None, category=category)


def test_qwen_native_format_with_valid_args_scores_correct_and_format_valid() -> None:
    raw = (
        "Sure, let me look that up.\n"
        "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Op. 9 No. 2\"}}</tool_call>"
    )
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.discipline_correct is True
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True


def test_raw_json_format_with_valid_args_scores_correct() -> None:
    raw = '{"name": "search_catalog", "arguments": {"query": "Chopin"}}'
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True
    assert out.discipline_correct is True


def test_prose_with_embedded_json_extracts_call() -> None:
    raw = (
        "I'll search for that. Here's my call:\n"
        '```json\n{"name": "search_catalog", "arguments": {"query": "Chopin"}}\n```\n'
        "Let me know what you find."
    )
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "search_catalog"
    assert out.format_valid is True


def test_no_tool_call_on_negative_case_scores_disciplined() -> None:
    raw = "Let's just listen for another minute and see how the dynamics develop."
    out = score_response(raw, _neg_case(category="premature"), _SCHEMAS)
    assert out.called is False
    assert out.discipline_correct is True
    assert out.format_valid is None  # not applicable when no call


def test_tool_call_on_negative_case_scores_undisciplined() -> None:
    raw = '<tool_call>{"name": "search_catalog", "arguments": {"query": "anything"}}</tool_call>'
    out = score_response(raw, _neg_case(category="chitchat"), _SCHEMAS)
    assert out.called is True
    assert out.discipline_correct is False  # called when shouldn't have


def test_no_tool_call_on_positive_case_scores_undisciplined() -> None:
    raw = "I think we should talk about that more before searching."
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is False
    assert out.discipline_correct is False


def test_wrong_tool_name_scores_undisciplined_even_if_called() -> None:
    raw = '<tool_call>{"name": "create_exercise", "arguments": {"skill": "x", "exercises": []}}</tool_call>'
    out = score_response(raw, _pos_case(tool="search_catalog"), _SCHEMAS)
    assert out.called is True
    assert out.tool_name == "create_exercise"
    assert out.discipline_correct is False


def test_invalid_args_against_schema_marks_format_invalid() -> None:
    raw = '<tool_call>{"name": "search_catalog", "arguments": {}}</tool_call>'
    out = score_response(raw, _pos_case(), _SCHEMAS)
    assert out.called is True
    assert out.format_valid is False
    # Discipline depends on whether tool name was right; here it was, so:
    assert out.discipline_correct is True
