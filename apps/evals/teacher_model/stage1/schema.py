import re
from typing import Annotated, Any, Literal, Union, get_args
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator, model_validator


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


_REGISTRY: dict[str, tuple[type[BaseModel], str]] = {}


def _register(name: str, model: type[BaseModel], description: str = "") -> None:
    _REGISTRY[name] = (model, description)


def validate_tool_input(name: str, payload: dict[str, Any]) -> list[str]:
    entry = _REGISTRY.get(name)
    if entry is None:
        return [f"unknown tool: {name}"]
    model, _ = entry
    try:
        model.model_validate(payload)
        return []
    except ValidationError as exc:
        return _format_pydantic_errors(exc)


_register(
    "create_exercise",
    _CreateExerciseInput,
    "Create one to three short practice exercises targeting a specific skill in a passage.",
)


_PIECE_SLUG = re.compile(r"^[a-z0-9._-]+$")


class _Highlight(BaseModel):
    bars: tuple[int, int]
    dimension: Literal[
        "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"
    ]
    annotation: Annotated[str, StringConstraints(max_length=500)] | None = None

    @field_validator("bars")
    @classmethod
    def _bars_ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1 or v[1] < 1:
            raise ValueError("bars must be >= 1")
        if v[0] > v[1]:
            raise ValueError("bars start must be <= end")
        return v


class _ScoreHighlightInput(BaseModel):
    piece_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    highlights: Annotated[list[_Highlight], Field(min_length=1, max_length=5)]

    @field_validator("piece_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _PIECE_SLUG.match(v):
            raise ValueError("piece_id must match catalog slug regex")
        return v


_register(
    "score_highlight",
    _ScoreHighlightInput,
    "Highlight bar ranges of a known catalog piece with a dimension annotation.",
)


class _KeyboardGuideInput(BaseModel):
    title: str
    description: str
    fingering: str | None = None
    hands: Literal["left", "right", "both"]


_register(
    "keyboard_guide",
    _KeyboardGuideInput,
    "Show a small on-screen keyboard guide with optional fingering for one hand or both.",
)


class _ShowSessionDataInput(BaseModel):
    query_type: Literal["dimension_history", "recent_sessions", "session_detail"]
    dimension: Literal[
        "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"
    ] | None = None
    session_id: str | None = None
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("session_id")
    @classmethod
    def _uuid(cls, v: str | None) -> str | None:
        if v is None:
            return v
        UUID(v)  # raises ValueError if not a valid UUID
        return v


_register(
    "show_session_data",
    _ShowSessionDataInput,
    "Surface the student's recent practice data for one of three query shapes.",
)


class _ReferenceBrowserInput(BaseModel):
    piece_id: str | None = None
    passage: str | None = None
    description: str

    @field_validator("piece_id")
    @classmethod
    def _slug(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _PIECE_SLUG.match(v):
            raise ValueError("piece_id must match catalog slug regex")
        return v


_register(
    "reference_browser",
    _ReferenceBrowserInput,
    "Suggest a reference recording or excerpt; piece_id and passage are optional.",
)


class _SearchCatalogInput(BaseModel):
    composer: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    opus_number: int | None = Field(default=None, ge=1, le=9999)
    piece_number: int | None = Field(default=None, ge=1, le=9999)
    title_keywords: Annotated[str, StringConstraints(min_length=3, max_length=200)] | None = None
    query: Annotated[str, StringConstraints(min_length=1, max_length=300)] | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "_SearchCatalogInput":
        provided = [
            self.composer,
            self.opus_number,
            self.piece_number,
            self.title_keywords,
            self.query,
        ]
        if all(p is None for p in provided):
            raise ValueError("at least one search field is required")
        if self.title_keywords is not None:
            tokens = [t for t in self.title_keywords.strip().split() if len(t) >= 2]
            if not tokens:
                raise ValueError(
                    "title_keywords must contain at least one token of 2+ characters"
                )
        return self


_register(
    "search_catalog",
    _SearchCatalogInput,
    "Search the piece catalog by composer, opus/piece number, title keywords, or free-form query.",
)


NegativeCategory = Literal[
    "chitchat",
    "premature",
    "ambiguous",
    "already_recommended",
    "out_of_scope",
    "borderline_wrong_tool",
]

NEGATIVE_CATEGORIES: tuple[str, ...] = get_args(NegativeCategory)


class Stage1Negative(BaseModel):
    shape: Literal["synthesis", "chat"]
    system_blocks: list[str]
    messages: list[Stage1Message]
    assistant: Stage1AssistantTurn
    category: NegativeCategory
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchedContrastPair(BaseModel):
    contrast_id: str
    positive: Stage1Example
    negative: Stage1Negative

    @model_validator(mode="after")
    def _ids_match(self) -> "MatchedContrastPair":
        pos_id = self.positive.metadata.get("contrast_id")
        neg_id = self.negative.metadata.get("contrast_id")
        if pos_id != self.contrast_id or neg_id != self.contrast_id:
            raise ValueError(
                f"contrast_id mismatch: pair={self.contrast_id} "
                f"positive={pos_id} negative={neg_id}"
            )
        return self


def build_anthropic_tool_schemas() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, (model, description) in _REGISTRY.items():
        json_schema = model.model_json_schema()
        json_schema.pop("title", None)
        out.append(
            {
                "name": name,
                "description": description,
                "input_schema": json_schema,
            }
        )
    return out
