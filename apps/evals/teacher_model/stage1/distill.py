import time
from dataclasses import dataclass
from typing import Any, Literal

from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.schema import (
    Stage1AssistantTurn,
    Stage1Example,
    Stage1Message,
    Stage1TextBlock,
    Stage1ToolUseBlock,
    validate_tool_input,
)

Shape = Literal["synthesis", "chat"]

_RETRY_MAX = 3
_RETRY_BACKOFF_SECONDS = 0.0  # zero in tests; production CLI sets via env


def _retryable_excs() -> tuple[type[BaseException], ...]:
    excs: tuple[type[BaseException], ...] = (ConnectionError,)
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return excs
    sdk_excs: list[type[BaseException]] = []
    for name in ("APIConnectionError", "RateLimitError", "APIStatusError"):
        cls = getattr(anthropic, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            sdk_excs.append(cls)
    return excs + tuple(sdk_excs)


def _call_with_retry(sonnet, **kwargs):
    retryable = _retryable_excs()
    last_exc: BaseException | None = None
    for attempt in range(_RETRY_MAX):
        try:
            return sonnet.messages_create(**kwargs)
        except retryable as exc:
            last_exc = exc
            if attempt < _RETRY_MAX - 1 and _RETRY_BACKOFF_SECONDS > 0:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
    assert last_exc is not None
    raise last_exc

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

    response = _call_with_retry(
        sonnet,
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=[{"type": "text", "text": s} for s in system_blocks],
        messages=messages,
        tools=tools,
        tool_choice={"type": "auto"},
    )

    content_blocks: list = []
    tool_uses_with_index: list[tuple[int, Stage1ToolUseBlock]] = []
    for idx, block in enumerate(response.content):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_blocks.append(Stage1TextBlock(text=block.text))
        elif block_type == "tool_use":
            tu = Stage1ToolUseBlock(id=block.id, name=block.name, input=block.input)
            content_blocks.append(tu)
            tool_uses_with_index.append((idx, tu))

    for block_index, tu in tool_uses_with_index:
        errors = validate_tool_input(tu.name, tu.input)
        if errors:
            return DistillResult(
                example=None,
                rejection=Rejection(
                    reason="validation",
                    tool_name=tu.name,
                    errors=errors,
                    block_index=block_index,
                ),
            )

    example = Stage1Example(
        shape=shape,
        system_blocks=system_blocks,
        messages=[Stage1Message(role=m["role"], content=m["content"]) for m in messages],
        assistant=Stage1AssistantTurn(content=content_blocks),
        metadata={"source": "distilled", "briefing_id": briefing.briefing_id},
    )
    return DistillResult(example=example, rejection=None)
