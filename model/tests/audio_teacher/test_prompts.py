"""Elicitation contract: axis-specific question + strict answer format."""
from __future__ import annotations

import pytest

from audio_teacher.prompts import build_question


@pytest.mark.parametrize(
    "axis,keyword",
    [("pedaling", "pedal"), ("dynamics", "dynamic"), ("phrasing", "phras")],
)
def test_question_names_the_axis_contrast_and_forces_ab_answer(axis, keyword):
    question = build_question(axis)
    assert keyword in question.lower()
    assert 'ANSWER: A' in question and 'ANSWER: B' in question
    with pytest.raises(KeyError):
        build_question("rubato")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The first clip blurs badly.\nANSWER: A", "a"),
        ("answer: b", "b"),
        ("ANSWER: A\nOn reflection...\nANSWER: B", "b"),  # last answer wins
        ("Both sound similar to me.", None),
        ("ANSWER: C", None),
        ("", None),
    ],
    ids=["plain", "lowercase", "last_wins", "no_answer", "invalid_letter", "empty"],
)
def test_parse_choice_extracts_forced_ab_or_none(text, expected):
    from audio_teacher.prompts import parse_choice

    assert parse_choice(text) == expected
