"""Degeneracy classification fixtures for the post-tool-result continuation probe."""
from __future__ import annotations

import pytest

from teacher_model.stage0.continuation_probe import (
    ContinuationResult,
    load_tool_result_fixture,
    score_continuation,
)

_INITIAL = (
    "Let me look that up for you."
    "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Ballade 1\"}}</tool_call>"
)
_RESULT = {"matches": [{"pieceId": "chopin.ballades.1"}]}


def test_clean_continuation_classified_clean() -> None:
    follow_up = (
        "Great — I found Chopin's Ballade No. 1. Try the second theme around bar 68 "
        "for that singing tone you're going for."
    )
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert isinstance(out, ContinuationResult)
    assert out.category == "clean"
    assert out.is_degenerate is False


def test_refusal_classified() -> None:
    follow_up = "I cannot continue. I am unable to help with this request."
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "refusal"
    assert out.is_degenerate is True


def test_repetition_classified_when_same_tool_call_re_emitted() -> None:
    follow_up = (
        "<tool_call>{\"name\": \"search_catalog\", \"arguments\": {\"query\": \"Chopin Ballade 1\"}}</tool_call>"
    )
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "repetition"
    assert out.is_degenerate is True


def test_format_collapse_classified_when_output_is_raw_json_dump() -> None:
    follow_up = '{"matches":[{"pieceId":"chopin.ballades.1"}]}'
    out = score_continuation(_INITIAL, _RESULT, follow_up)
    assert out.category == "format_collapse"
    assert out.is_degenerate is True


def test_empty_classified_when_below_length_threshold() -> None:
    out = score_continuation(_INITIAL, _RESULT, "ok")
    assert out.category == "empty"
    assert out.is_degenerate is True


def test_load_tool_result_fixture_returns_dict_for_each_tool() -> None:
    for tool in (
        "create_exercise",
        "score_highlight",
        "keyboard_guide",
        "show_session_data",
        "reference_browser",
        "search_catalog",
    ):
        fixture = load_tool_result_fixture(tool)
        assert isinstance(fixture, dict)
        assert fixture, f"fixture for {tool} must be non-empty"


def test_load_tool_result_fixture_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        load_tool_result_fixture("not_a_real_tool")
