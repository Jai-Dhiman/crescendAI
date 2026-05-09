from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, Field, StringConstraints, ValidationError


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


class _CreateExerciseExercise(BaseModel):
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    instruction: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    focus_dimension: Literal[
        "dynamics",
        "timing",
        "pedaling",
        "articulation",
        "phrasing",
        "interpretation",
    ]
    hands: Literal["left", "right", "both"] | None = None


class _CreateExerciseInput(BaseModel):
    source_passage: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    target_skill: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    exercises: Annotated[list[_CreateExerciseExercise], Field(min_length=1, max_length=3)]


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        ctx = err.get("ctx", {})
        ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        detail = f"{err['msg']} ({ctx_str})" if ctx_str else err["msg"]
        out.append(f"{loc}: {detail}")
    return out


_VALIDATORS: dict[str, Callable[[dict[str, Any]], list[str]]] = {}


def _register(name: str, model: type[BaseModel]) -> None:
    def validator(payload: dict[str, Any]) -> list[str]:
        try:
            model.model_validate(payload)
            return []
        except ValidationError as exc:
            return _format_pydantic_errors(exc)

    _VALIDATORS[name] = validator


_register("create_exercise", _CreateExerciseInput)


def validate_tool_input(name: str, payload: dict[str, Any]) -> list[str]:
    validator = _VALIDATORS.get(name)
    if validator is None:
        return [f"unknown tool: {name}"]
    return validator(payload)
