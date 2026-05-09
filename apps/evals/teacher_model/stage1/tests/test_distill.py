from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.distill import DistillResult, distill


class _StubSonnet:
    def __init__(self, content_blocks: list[dict]):
        self._blocks = content_blocks
        self.calls: list[dict] = []

    def messages_create(self, *, model, max_tokens, system, messages, tools, tool_choice):
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
            }
        )
        return type("Response", (), {"content": self._blocks})()


_STUB_TOOLS = [
    {
        "name": "create_exercise",
        "description": "stub",
        "input_schema": {"type": "object", "properties": {}},
    }
]


def test_distill_forwards_tools_and_returns_example():
    briefing = Briefing(
        briefing_id="rec_001",
        framing_text="<session_data>{...}</session_data>",
        composer="Chopin",
        skill_bucket="intermediate",
    )
    sonnet = _StubSonnet(
        content_blocks=[
            type(
                "TextBlock",
                (),
                {"type": "text", "text": "<analysis>brief</analysis>\n\nNice work."},
            )(),
            type(
                "ToolUseBlock",
                (),
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "create_exercise",
                    "input": {
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
                },
            )(),
        ]
    )

    result = distill(
        briefing, "synthesis", sonnet, "UNIFIED_TEACHER_SYSTEM", tools=_STUB_TOOLS
    )

    assert isinstance(result, DistillResult)
    assert result.rejection is None
    assert result.example is not None
    assert result.example.shape == "synthesis"
    assert len(result.example.assistant.content) == 2
    assert result.example.assistant.content[1].name == "create_exercise"
    assert result.example.metadata["source"] == "distilled"
    assert result.example.metadata["briefing_id"] == "rec_001"
    assert len(sonnet.calls) == 1
    assert sonnet.calls[0]["tools"] == _STUB_TOOLS


def test_distill_rejects_invalid_tool_input_with_structured_metadata():
    briefing = Briefing(
        briefing_id="rec_002",
        framing_text="...",
        composer="Bach",
        skill_bucket="beginner",
    )
    sonnet = _StubSonnet(
        content_blocks=[
            type(
                "TextBlock",
                (),
                {"type": "text", "text": "preamble"},
            )(),  # block_index 0 -- fine
            type(
                "ToolUseBlock",
                (),
                {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "score_highlight",
                    "input": {
                        # piece_id missing -- invalid
                        "highlights": [{"bars": [1, 4], "dimension": "phrasing"}],
                    },
                },
            )(),  # block_index 1 -- the offender
        ]
    )

    result = distill(
        briefing, "chat", sonnet, "UNIFIED_TEACHER_SYSTEM", tools=_STUB_TOOLS
    )

    assert result.example is None
    assert result.rejection is not None
    assert result.rejection.reason == "validation"
    assert result.rejection.tool_name == "score_highlight"
    assert result.rejection.block_index == 1
    assert any("piece_id" in e for e in result.rejection.errors), result.rejection.errors
