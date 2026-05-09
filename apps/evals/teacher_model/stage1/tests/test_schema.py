import pytest

from teacher_model.stage1.schema import (
    Stage1Example,
    Stage1AssistantTurn,
    Stage1ToolUseBlock,
    Stage1TextBlock,
    validate_tool_input,
)


def test_stage1_example_roundtrips_json():
    original = Stage1Example(
        shape="synthesis",
        system_blocks=["UNIFIED_TEACHER_SYSTEM", "<session_data>...</session_data>"],
        messages=[
            {"role": "user", "content": "Please provide your session synthesis."}
        ],
        assistant=Stage1AssistantTurn(
            content=[
                Stage1TextBlock(text="<analysis>brief</analysis>\n\nNice work."),
                Stage1ToolUseBlock(
                    id="toolu_01",
                    name="create_exercise",
                    input={
                        "source_passage": "bars 5-8",
                        "target_skill": "voice balance",
                        "exercises": [
                            {
                                "title": "LH only",
                                "instruction": "Play LH alone, listening for evenness.",
                                "focus_dimension": "dynamics",
                            }
                        ],
                    },
                ),
            ]
        ),
        metadata={"source": "distilled", "combo_rationale": None},
    )

    serialized = original.model_dump_json()
    parsed = Stage1Example.model_validate_json(serialized)
    assert parsed == original


def test_validate_tool_input_unknown_tool_name():
    errors = validate_tool_input("does_not_exist", {})
    assert len(errors) == 1
    assert "unknown tool" in errors[0].lower()
    assert "does_not_exist" in errors[0]


@pytest.mark.parametrize(
    "tool_input,expected_error_substring",
    [
        (
            {
                "source_passage": "bars 5-8",
                "target_skill": "voice balance",
                "exercises": [
                    {
                        "title": "LH only",
                        "instruction": "Play LH alone.",
                        "focus_dimension": "dynamics",
                    }
                ],
            },
            None,  # valid
        ),
        (
            {"target_skill": "x", "exercises": []},
            "source_passage",  # missing required
        ),
        (
            {
                "source_passage": "x",
                "target_skill": "x",
                "exercises": [],
            },
            "min",  # at least 1 exercise required
        ),
        (
            {
                "source_passage": "x",
                "target_skill": "x",
                "exercises": [
                    {
                        "title": "x",
                        "instruction": "x",
                        "focus_dimension": "loud",  # invalid enum
                    }
                ],
            },
            "focus_dimension",
        ),
    ],
)
def test_validate_tool_input_create_exercise(tool_input, expected_error_substring):
    errors = validate_tool_input("create_exercise", tool_input)
    if expected_error_substring is None:
        assert errors == []
    else:
        assert any(expected_error_substring in e for e in errors), errors


@pytest.mark.parametrize(
    "tool_input,expected_substring",
    [
        (
            {
                "piece_id": "chopin.ballades.1",
                "highlights": [
                    {"bars": [5, 8], "dimension": "phrasing", "annotation": "shape"},
                ],
            },
            None,
        ),
        (
            {
                "piece_id": "chopin.ballades.1",
                "highlights": [{"bars": [10, 5], "dimension": "phrasing"}],
            },
            "start",  # bars start must be <= end
        ),
        (
            {"highlights": [{"bars": [1, 2], "dimension": "phrasing"}]},
            "piece_id",
        ),
        (
            {
                "piece_id": "chopin.ballades.1",
                "highlights": [
                    {"bars": [i, i], "dimension": "phrasing"} for i in range(1, 7)
                ],
            },
            "max",  # >5 highlights
        ),
        (
            {
                "piece_id": "chopin.ballades.1",
                "highlights": [{"bars": [1, 2], "dimension": "rhythm"}],
            },
            "dimension",
        ),
    ],
)
def test_validate_tool_input_score_highlight(tool_input, expected_substring):
    errors = validate_tool_input("score_highlight", tool_input)
    if expected_substring is None:
        assert errors == []
    else:
        assert any(expected_substring in e for e in errors), errors
