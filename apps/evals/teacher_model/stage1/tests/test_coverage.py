from teacher_model.stage1.coverage import CoverageMatrix
from teacher_model.stage1.schema import (
    Stage1AssistantTurn,
    Stage1Example,
    Stage1ToolUseBlock,
)


def _example_calling(tool: str, payload: dict) -> Stage1Example:
    return Stage1Example(
        shape="synthesis",
        system_blocks=[],
        messages=[{"role": "user", "content": "stub"}],
        assistant=Stage1AssistantTurn(
            content=[Stage1ToolUseBlock(id="t1", name=tool, input=payload)]
        ),
    )


def test_coverage_matrix_records_and_reports_satisfaction():
    targets = {
        "create_exercise": {
            "focus_dimension:dynamics": 2,
            "focus_dimension:timing": 1,
        },
    }
    matrix = CoverageMatrix(targets=targets)

    assert not matrix.is_satisfied()
    assert {(c.tool, c.cell) for c in matrix.unfilled_cells()} == {
        ("create_exercise", "focus_dimension:dynamics"),
        ("create_exercise", "focus_dimension:timing"),
    }

    matrix.record(
        _example_calling(
            "create_exercise",
            {
                "source_passage": "x",
                "target_skill": "x",
                "exercises": [
                    {"title": "x", "instruction": "x", "focus_dimension": "dynamics"},
                ],
            },
        )
    )
    assert not matrix.is_satisfied()  # dynamics count 1, target 2

    matrix.record(
        _example_calling(
            "create_exercise",
            {
                "source_passage": "x",
                "target_skill": "x",
                "exercises": [
                    {"title": "x", "instruction": "x", "focus_dimension": "dynamics"},
                    {"title": "y", "instruction": "y", "focus_dimension": "timing"},
                ],
            },
        )
    )
    assert matrix.is_satisfied()  # dynamics: 2, timing: 1
    assert matrix.unfilled_cells() == []
