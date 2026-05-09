from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class Stage1TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Stage1ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


Stage1ContentBlock = Annotated[
    Union[Stage1TextBlock, Stage1ToolUseBlock],
    Field(discriminator="type"),
]


class Stage1AssistantTurn(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Stage1ContentBlock]


class Stage1Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Stage1Example(BaseModel):
    shape: Literal["synthesis", "chat"]
    system_blocks: list[str]
    messages: list[Stage1Message]
    assistant: Stage1AssistantTurn
    metadata: dict[str, Any] = Field(default_factory=dict)
