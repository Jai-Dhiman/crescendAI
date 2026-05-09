from dataclasses import dataclass
from typing import Any, Literal

from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.schema import (
    Stage1AssistantTurn,
    Stage1Example,
    Stage1Message,
    Stage1TextBlock,
    Stage1ToolUseBlock,
)

Shape = Literal["synthesis", "chat"]

_USER_STUB_SYNTHESIS = "Please provide your session synthesis."


@dataclass(frozen=True)
class Rejection:
    reason: str
    tool_name: str | None
    errors: list[str]
    block_index: int


@dataclass(frozen=True)
class DistillResult:
    example: Stage1Example | None
    rejection: Rejection | None


def distill(
    briefing: Briefing,
    shape: Shape,
    sonnet,
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> DistillResult:
    if shape == "synthesis":
        system_blocks = [system_prompt, briefing.framing_text]
        messages = [{"role": "user", "content": _USER_STUB_SYNTHESIS}]
    else:
        system_blocks = [system_prompt]
        messages = [{"role": "user", "content": briefing.framing_text}]

    response = sonnet.messages_create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=[{"type": "text", "text": s} for s in system_blocks],
        messages=messages,
        tools=tools,
        tool_choice={"type": "auto"},
    )

    content_blocks: list = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_blocks.append(Stage1TextBlock(text=block.text))
        elif block_type == "tool_use":
            content_blocks.append(
                Stage1ToolUseBlock(id=block.id, name=block.name, input=block.input)
            )

    example = Stage1Example(
        shape=shape,
        system_blocks=system_blocks,
        messages=[Stage1Message(role=m["role"], content=m["content"]) for m in messages],
        assistant=Stage1AssistantTurn(content=content_blocks),
        metadata={"source": "distilled", "briefing_id": briefing.briefing_id},
    )
    return DistillResult(example=example, rejection=None)
