from teacher_model.stage1.schema import (
    Stage1Example,
    Stage1AssistantTurn,
    Stage1ToolUseBlock,
    Stage1TextBlock,
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
