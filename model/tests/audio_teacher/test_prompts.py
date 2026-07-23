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
