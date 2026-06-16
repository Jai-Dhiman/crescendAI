"""Verify that SynthesisResult exposes the prescribed_exercise field."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

from shared.pipeline_client import SynthesisResult


def test_synthesis_result_has_prescribed_exercise_field():
    """SynthesisResult must carry prescribed_exercise parsed from eval_context."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={
            "prescribed_exercise": {
                "kind": "own_passage_loop",
                "target_dimension": "dynamics",
                "bar_range": [1, 4],
                "tempo_factor": 0.8,
            }
        },
    )
    assert result.prescribed_exercise is not None
    assert result.prescribed_exercise["kind"] == "own_passage_loop"
    assert result.prescribed_exercise["target_dimension"] == "dynamics"


def test_synthesis_result_prescribed_exercise_none_when_null():
    """prescribed_exercise is None when the artifact emitted null."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={"prescribed_exercise": None},
    )
    assert result.prescribed_exercise is None


def test_synthesis_result_prescribed_exercise_none_when_absent():
    """prescribed_exercise is None when eval_context has no key (legacy sessions)."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={},
    )
    assert result.prescribed_exercise is None
