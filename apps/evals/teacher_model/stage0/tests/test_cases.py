"""Verify the committed tool-probe cases file has the required structure."""
from __future__ import annotations

from pathlib import Path

from teacher_model.stage0.cases import load_cases

_CASES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tool_probe_cases.jsonl"
)

_VALID_TOOLS = {
    "create_exercise",
    "score_highlight",
    "keyboard_guide",
    "show_session_data",
    "reference_browser",
    "search_catalog",
}

_VALID_NEG_CATEGORIES = {
    "chitchat",
    "premature",
    "ambiguous",
    "already_recommended",
    "out_of_scope",
    "borderline_wrong_tool",
}


def test_file_has_exactly_40_cases() -> None:
    cases = load_cases(_CASES_PATH)
    assert len(cases) == 40


def test_split_is_20_positive_20_negative() -> None:
    cases = load_cases(_CASES_PATH)
    pos = [c for c in cases if c.expected_call]
    neg = [c for c in cases if not c.expected_call]
    assert len(pos) == 20 and len(neg) == 20


def test_positive_cases_use_only_known_tools() -> None:
    cases = load_cases(_CASES_PATH)
    for c in cases:
        if c.expected_call:
            assert c.expected_tool in _VALID_TOOLS, f"unknown tool: {c.expected_tool}"


def test_negative_cases_cover_all_six_categories() -> None:
    cases = load_cases(_CASES_PATH)
    cats = {c.category for c in cases if not c.expected_call and c.category}
    assert cats == _VALID_NEG_CATEGORIES, f"missing categories: {_VALID_NEG_CATEGORIES - cats}"


def test_each_case_has_a_briefing_field() -> None:
    cases = load_cases(_CASES_PATH)
    for c in cases:
        assert c.briefing, f"case {c.case_id} missing briefing"
