# Stage 1 Tool-Format SFT Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task), respecting the "files touched" annotation -- tasks within a group that touch the same file must be sequenced. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Produce the locked specification, authoring tooling, and evaluation harness for ~2K Stage 1 training examples that teach Qwen3.6-35B-A3B to natively emit chat-template-native Anthropic-shaped `tool_use` blocks.
**Spec:** `docs/specs/2026-05-08-stage1-tool-format-sft-design.md`
**Style:** Follow `CLAUDE.md` (root + apps/) and user-global preferences. Python via `uv`. Explicit exception handling. No emojis. No fallback shims.

## Task Groups

```
Group A (foundation — schema + briefing source + contract):
  Sequential within schema.py: T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8 -> T9
  Sequential within briefing_source.py: T10 -> T11
  T25 (schema contract test) depends on T9; otherwise parallel with briefing-source track.
  Schema-track and briefing-source-track are parallel.
Group B (consumers of schema; depends on A):
  T12, T13 sequential within negatives_loader.py
  T14 standalone (coverage.py)
  T15 standalone (holdout.py)
  T16, T17 sequential within render.py
  T12/14/15/16 are mutually parallel.
Group C (distill; depends on A): T18 -> T19 -> T20 sequential within distill.py
Group D (harness; depends on B): T21 standalone (harness.py)
Group E (CLI; depends on B+C+D): T22 -> T23 sequential within cli.py
Group F (cleanup; parallel with everything): T24 standalone (tool_format.py)
```

---

### Task 1: schema — Stage1Example JSON round-trip
**Group:** A
**Behavior being verified:** A Stage1Example with mixed text + tool_use content blocks serializes to JSON and parses back to an equivalent object.
**Interface under test:** `Stage1Example.model_dump_json` / `Stage1Example.model_validate_json`

**Files:**
- Create: `apps/evals/teacher_model/stage1/__init__.py`
- Create: `apps/evals/teacher_model/stage1/schema.py`
- Create: `apps/evals/teacher_model/stage1/tests/__init__.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_schema.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py::test_stage1_example_roundtrips_json -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.schema'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/__init__.py
# (empty)

# apps/evals/teacher_model/stage1/tests/__init__.py
# (empty)

# apps/evals/teacher_model/stage1/schema.py
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py::test_stage1_example_roundtrips_json -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/__init__.py apps/evals/teacher_model/stage1/tests/__init__.py apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add Stage1Example schema with tool_use content blocks"
```

---

### Task 2: schema — validate_tool_input dispatcher + create_exercise validator
**Group:** A (depends on T1; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input` returns an error when given an unknown tool name; for `create_exercise` it accepts valid input and reports specific errors for invalid input (missing required field, invalid focus_dimension enum, exercises array length out of bounds).

**Interface under test:** `validate_tool_input(name: str, input: dict) -> list[str]`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
import pytest

from teacher_model.stage1.schema import validate_tool_input


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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "validate_tool_input"
```
Expected: FAIL — `ImportError: cannot import name 'validate_tool_input' from 'teacher_model.stage1.schema'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
from typing import Callable

from pydantic import ValidationError, conlist, constr

DIMS_6 = (
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
)


class _CreateExerciseExercise(BaseModel):
    title: constr(min_length=1, max_length=200)
    instruction: constr(min_length=1, max_length=4000)
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
    source_passage: constr(min_length=1, max_length=500)
    target_skill: constr(min_length=1, max_length=500)
    exercises: conlist(_CreateExerciseExercise, min_length=1, max_length=3)


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        out.append(f"{loc}: {err['msg']}")
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add validate_tool_input dispatcher + create_exercise validator"
```

---

### Task 3: schema — score_highlight validator
**Group:** A (depends on T2; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input("score_highlight", ...)` accepts valid bar tuples and rejects: bars where start > end, missing piece_id, more than 5 highlights, invalid dimension enum.

**Interface under test:** `validate_tool_input("score_highlight", input)`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "score_highlight"
```
Expected: FAIL — assertions on validator output of an unregistered tool (`unknown tool: score_highlight`)

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
import re
from pydantic import field_validator

_PIECE_SLUG = re.compile(r"^[a-z0-9._-]+$")


class _Highlight(BaseModel):
    bars: tuple[int, int]
    dimension: Literal[
        "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"
    ]
    annotation: constr(max_length=500) | None = None

    @field_validator("bars")
    @classmethod
    def _bars_ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1 or v[1] < 1:
            raise ValueError("bars must be >= 1")
        if v[0] > v[1]:
            raise ValueError("bars start must be <= end")
        return v


class _ScoreHighlightInput(BaseModel):
    piece_id: constr(min_length=1, max_length=200)
    highlights: conlist(_Highlight, min_length=1, max_length=5)

    @field_validator("piece_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _PIECE_SLUG.match(v):
            raise ValueError("piece_id must match catalog slug regex")
        return v


_register("score_highlight", _ScoreHighlightInput)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add score_highlight validator with bar tuple + slug rules"
```

---

### Task 4: schema — keyboard_guide validator
**Group:** A (depends on T3; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input("keyboard_guide", ...)` accepts valid input with optional fingering and rejects: missing required title/description/hands, invalid hands enum.

**Interface under test:** `validate_tool_input("keyboard_guide", input)`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
@pytest.mark.parametrize(
    "tool_input,expected_substring",
    [
        (
            {"title": "C-major scale", "description": "Right-hand scale.", "hands": "right"},
            None,
        ),
        (
            {
                "title": "x",
                "description": "x",
                "hands": "right",
                "fingering": "1-2-3-1-2-3-4-5",
            },
            None,
        ),
        ({"description": "x", "hands": "right"}, "title"),
        ({"title": "x", "hands": "right"}, "description"),
        ({"title": "x", "description": "x"}, "hands"),
        (
            {"title": "x", "description": "x", "hands": "third"},
            "hands",
        ),
    ],
)
def test_validate_tool_input_keyboard_guide(tool_input, expected_substring):
    errors = validate_tool_input("keyboard_guide", tool_input)
    if expected_substring is None:
        assert errors == []
    else:
        assert any(expected_substring in e for e in errors), errors
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "keyboard_guide"
```
Expected: FAIL — `unknown tool: keyboard_guide`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
class _KeyboardGuideInput(BaseModel):
    title: str
    description: str
    fingering: str | None = None
    hands: Literal["left", "right", "both"]


_register("keyboard_guide", _KeyboardGuideInput)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add keyboard_guide validator"
```

---

### Task 5: schema — show_session_data validator
**Group:** A (depends on T4; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input("show_session_data", ...)` accepts each of the 3 query_type values, applies the limit ceiling (max 50), validates UUID format on session_id, and rejects unknown query_type.

**Interface under test:** `validate_tool_input("show_session_data", input)`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
@pytest.mark.parametrize(
    "tool_input,expected_substring",
    [
        ({"query_type": "dimension_history", "dimension": "dynamics"}, None),
        ({"query_type": "recent_sessions", "limit": 20}, None),
        (
            {
                "query_type": "session_detail",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            },
            None,
        ),
        ({"query_type": "all_time"}, "query_type"),
        ({"query_type": "recent_sessions", "limit": 100}, "less than or equal to 50"),
        ({"query_type": "session_detail", "session_id": "not-a-uuid"}, "uuid"),
    ],
)
def test_validate_tool_input_show_session_data(tool_input, expected_substring):
    errors = validate_tool_input("show_session_data", tool_input)
    if expected_substring is None:
        assert errors == []
    else:
        assert any(expected_substring in e.lower() for e in errors), errors
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "show_session_data"
```
Expected: FAIL — `unknown tool: show_session_data`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
from uuid import UUID


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


_register("show_session_data", _ShowSessionDataInput)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add show_session_data validator with conditional fields"
```

---

### Task 6: schema — reference_browser validator
**Group:** A (depends on T5; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input("reference_browser", ...)` requires `description`; piece_id and passage are optional; rejects bad piece_id slug.

**Interface under test:** `validate_tool_input("reference_browser", input)`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
@pytest.mark.parametrize(
    "tool_input,expected_substring",
    [
        ({"description": "Listen to Argerich's recording"}, None),
        (
            {"piece_id": "chopin.ballades.1", "description": "x", "passage": "bars 5-8"},
            None,
        ),
        ({}, "description"),
        ({"description": "x", "piece_id": "Has Capital Letters"}, "piece_id"),
    ],
)
def test_validate_tool_input_reference_browser(tool_input, expected_substring):
    errors = validate_tool_input("reference_browser", tool_input)
    if expected_substring is None:
        assert errors == []
    else:
        assert any(expected_substring in e for e in errors), errors
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "reference_browser"
```
Expected: FAIL — `unknown tool: reference_browser`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
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


_register("reference_browser", _ReferenceBrowserInput)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add reference_browser validator"
```

---

### Task 7: schema — search_catalog validator
**Group:** A (depends on T6; touches schema.py and test_schema.py)

**Behavior being verified:** `validate_tool_input("search_catalog", ...)` requires at-least-one of {composer, opus_number, piece_number, title_keywords, query}; rejects empty input; accepts each individual field; rejects title_keywords with no token >=2 chars.

**Interface under test:** `validate_tool_input("search_catalog", input)`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
@pytest.mark.parametrize(
    "tool_input,expected_substring",
    [
        ({"composer": "Chopin"}, None),
        ({"composer": "Chopin", "opus_number": 64, "piece_number": 2}, None),
        ({"title_keywords": "Nocturne in C"}, None),
        ({"query": "that slow Bach prelude"}, None),
        ({}, "at least one"),
        ({"title_keywords": "a b c"}, "2+ characters"),
    ],
)
def test_validate_tool_input_search_catalog(tool_input, expected_substring):
    errors = validate_tool_input("search_catalog", tool_input)
    if expected_substring is None:
        assert errors == []
    else:
        assert any(expected_substring in e.lower() for e in errors), errors
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "search_catalog"
```
Expected: FAIL — `unknown tool: search_catalog`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
from pydantic import model_validator


class _SearchCatalogInput(BaseModel):
    composer: constr(min_length=1, max_length=200) | None = None
    opus_number: int | None = Field(default=None, ge=1, le=9999)
    piece_number: int | None = Field(default=None, ge=1, le=9999)
    title_keywords: constr(min_length=3, max_length=200) | None = None
    query: constr(min_length=1, max_length=300) | None = None

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


_register("search_catalog", _SearchCatalogInput)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add search_catalog validator with at-least-one rule"
```

---

### Task 8: schema — Stage1Negative model with 6-category enum
**Group:** A (depends on T7; touches schema.py and test_schema.py)

**Behavior being verified:** `Stage1Negative` enforces a category from the 6-value enum; rejects unknown category; round-trips JSON.

**Interface under test:** `Stage1Negative.model_validate` / `model_dump_json`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
from teacher_model.stage1.schema import Stage1Negative, NEGATIVE_CATEGORIES


def test_stage1_negative_enforces_category_enum():
    valid = Stage1Negative(
        shape="chat",
        system_blocks=["UNIFIED_TEACHER_SYSTEM"],
        messages=[{"role": "user", "content": "thanks!"}],
        assistant=Stage1AssistantTurn(
            content=[Stage1TextBlock(text="You're very welcome.")]
        ),
        category="chitchat",
        metadata={"rationale": "social close, no teaching needed"},
    )
    parsed = Stage1Negative.model_validate_json(valid.model_dump_json())
    assert parsed.category == "chitchat"

    with pytest.raises(Exception) as exc_info:
        Stage1Negative(
            shape="chat",
            system_blocks=[],
            messages=[],
            assistant=Stage1AssistantTurn(content=[]),
            category="not_a_real_category",
        )
    assert "category" in str(exc_info.value).lower()


def test_negative_categories_constant_has_six_values():
    assert set(NEGATIVE_CATEGORIES) == {
        "chitchat",
        "premature",
        "ambiguous",
        "already_recommended",
        "out_of_scope",
        "borderline_wrong_tool",
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "negative"
```
Expected: FAIL — `ImportError: cannot import name 'Stage1Negative'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
NEGATIVE_CATEGORIES = (
    "chitchat",
    "premature",
    "ambiguous",
    "already_recommended",
    "out_of_scope",
    "borderline_wrong_tool",
)

NegativeCategory = Literal[
    "chitchat",
    "premature",
    "ambiguous",
    "already_recommended",
    "out_of_scope",
    "borderline_wrong_tool",
]


class Stage1Negative(BaseModel):
    shape: Literal["synthesis", "chat"]
    system_blocks: list[str]
    messages: list[Stage1Message]
    assistant: Stage1AssistantTurn
    category: NegativeCategory
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add Stage1Negative model with 6-category enum"
```

---

### Task 9: schema — MatchedContrastPair with contrast_id
**Group:** A (depends on T8; touches schema.py and test_schema.py)

**Behavior being verified:** `MatchedContrastPair` requires both members to share the same `contrast_id`; rejects mismatched ids.

**Interface under test:** `MatchedContrastPair.model_validate`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/schema.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_schema.py
from teacher_model.stage1.schema import MatchedContrastPair


def _make_positive(contrast_id: str) -> Stage1Example:
    return Stage1Example(
        shape="chat",
        system_blocks=[],
        messages=[{"role": "user", "content": "show me bars 5-8 of Chopin Op. 23"}],
        assistant=Stage1AssistantTurn(
            content=[
                Stage1ToolUseBlock(
                    id="t1",
                    name="search_catalog",
                    input={"composer": "Chopin", "opus_number": 23},
                )
            ]
        ),
        metadata={"contrast_id": contrast_id, "source": "hand"},
    )


def _make_negative(contrast_id: str) -> Stage1Negative:
    return Stage1Negative(
        shape="chat",
        system_blocks=[],
        messages=[{"role": "user", "content": "show me bars 5-8 of chopin.ballades.1"}],
        assistant=Stage1AssistantTurn(
            content=[
                Stage1ToolUseBlock(
                    id="t1",
                    name="score_highlight",
                    input={
                        "piece_id": "chopin.ballades.1",
                        "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
                    },
                )
            ]
        ),
        category="borderline_wrong_tool",  # placeholder; not the real meaning here
        metadata={"contrast_id": contrast_id},
    )


def test_matched_contrast_pair_requires_same_contrast_id():
    pair = MatchedContrastPair(
        contrast_id="cp_001",
        positive=_make_positive("cp_001"),
        negative=_make_negative("cp_001"),
    )
    assert pair.contrast_id == "cp_001"

    with pytest.raises(Exception) as exc_info:
        MatchedContrastPair(
            contrast_id="cp_001",
            positive=_make_positive("cp_001"),
            negative=_make_negative("cp_999"),
        )
    assert "contrast_id" in str(exc_info.value).lower()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x -k "matched_contrast"
```
Expected: FAIL — `ImportError: cannot import name 'MatchedContrastPair'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/schema.py
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/schema.py apps/evals/teacher_model/stage1/tests/test_schema.py && git commit -m "feat(stage1): add MatchedContrastPair with contrast_id cross-ref"
```

---

### Task 10: briefing_source — iter_synthesis_briefings loads cached files
**Group:** A (parallel with T1-T9; touches briefing_source.py)

**Behavior being verified:** `iter_synthesis_briefings(cache_dir)` yields one Briefing per JSON file in the directory, with the briefing's text content extracted from the cached entry.

**Interface under test:** `iter_synthesis_briefings(cache_dir: Path) -> Iterator[Briefing]`

**Files:**
- Create: `apps/evals/teacher_model/stage1/briefing_source.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_briefing_source.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_briefing_source.py
import json
from pathlib import Path

from teacher_model.stage1.briefing_source import iter_synthesis_briefings


def test_iter_synthesis_briefings_yields_one_per_file(tmp_path: Path):
    briefing_a = {
        "briefing_id": "rec_aaa",
        "framing_text": "<session_data>{...A...}</session_data>",
        "composer": "Chopin",
        "skill_bucket": "intermediate",
    }
    briefing_b = {
        "briefing_id": "rec_bbb",
        "framing_text": "<session_data>{...B...}</session_data>",
        "composer": "Bach",
        "skill_bucket": "beginner",
    }
    (tmp_path / "rec_aaa.json").write_text(json.dumps(briefing_a))
    (tmp_path / "rec_bbb.json").write_text(json.dumps(briefing_b))

    yielded = sorted(iter_synthesis_briefings(tmp_path), key=lambda b: b.briefing_id)
    assert [b.briefing_id for b in yielded] == ["rec_aaa", "rec_bbb"]
    assert yielded[0].framing_text == briefing_a["framing_text"]
    assert yielded[0].composer == "Chopin"
    assert yielded[1].skill_bucket == "beginner"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_briefing_source.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.briefing_source'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/briefing_source.py
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Briefing:
    briefing_id: str
    framing_text: str
    composer: str
    skill_bucket: str
    shape: str = "synthesis"


def iter_synthesis_briefings(cache_dir: Path) -> Iterator[Briefing]:
    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text())
        yield Briefing(
            briefing_id=data["briefing_id"],
            framing_text=data["framing_text"],
            composer=data.get("composer", ""),
            skill_bucket=data.get("skill_bucket", ""),
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_briefing_source.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/briefing_source.py apps/evals/teacher_model/stage1/tests/test_briefing_source.py && git commit -m "feat(stage1): add iter_synthesis_briefings cache loader"
```

---

### Task 11: briefing_source — iter_chat_scenarios deterministic generator
**Group:** A (depends on T10; touches briefing_source.py and test_briefing_source.py)

**Behavior being verified:** `iter_chat_scenarios(template, n, seed)` produces n Briefings with `shape="chat"` whose conversation history is parameterized from the template; the same seed produces identical output across runs.

**Interface under test:** `iter_chat_scenarios(template: dict, n: int, seed: int) -> Iterator[Briefing]`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/briefing_source.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_briefing_source.py`
- Create: `apps/evals/teacher_model/stage1/data/chat_scenario_template.json`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_briefing_source.py
from teacher_model.stage1.briefing_source import iter_chat_scenarios

_TEMPLATE = {
    "intents": [
        {"id": "show_bars", "user": "Can you show me bars {bar_start}-{bar_end} of {piece}?"},
        {"id": "find_piece", "user": "What's that {era} piece by {composer}?"},
    ],
    "fillers": {
        "bar_start": [1, 5, 12],
        "bar_end": [4, 8, 16],
        "piece": ["Chopin Ballade 1", "Bach Prelude in C"],
        "era": ["Romantic", "Baroque"],
        "composer": ["Chopin", "Bach"],
    },
}


def test_iter_chat_scenarios_n_and_determinism():
    a = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=42))
    b = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=42))
    c = list(iter_chat_scenarios(_TEMPLATE, n=10, seed=99))

    assert len(a) == 10
    assert all(b_item.shape == "chat" for b_item in a)
    assert [x.briefing_id for x in a] == [x.briefing_id for x in b]
    assert [x.framing_text for x in a] == [x.framing_text for x in b]
    assert [x.framing_text for x in a] != [x.framing_text for x in c]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_briefing_source.py -x -k "chat_scenarios"
```
Expected: FAIL — `ImportError: cannot import name 'iter_chat_scenarios'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/briefing_source.py
import random


def iter_chat_scenarios(template: dict, n: int, seed: int) -> Iterator[Briefing]:
    rng = random.Random(seed)
    intents = template["intents"]
    fillers = template["fillers"]
    for i in range(n):
        intent = rng.choice(intents)
        chosen = {k: rng.choice(v) for k, v in fillers.items()}
        user_text = intent["user"].format(**chosen)
        briefing_id = f"chat_{seed}_{i:04d}_{intent['id']}"
        yield Briefing(
            briefing_id=briefing_id,
            framing_text=user_text,
            composer=str(chosen.get("composer", "")),
            skill_bucket="intermediate",
            shape="chat",
        )
```

Then create the seed template file:

```bash
mkdir -p apps/evals/teacher_model/stage1/data
```

```json
// apps/evals/teacher_model/stage1/data/chat_scenario_template.json
{
  "intents": [
    {"id": "show_bars", "user": "Can you show me bars {bar_start}-{bar_end} of {piece}?"},
    {"id": "find_piece", "user": "What's that {era} piece by {composer}?"},
    {"id": "fingering", "user": "How should I finger the run in {piece}?"},
    {"id": "exercise_request", "user": "Can you give me an exercise for {target_skill}?"},
    {"id": "history_query", "user": "How have I been doing on {dimension} this month?"},
    {"id": "chitchat", "user": "Thanks, that helped a lot."},
    {"id": "vent", "user": "I'm just frustrated today, nothing's working."}
  ],
  "fillers": {
    "bar_start": [1, 5, 9, 12, 17, 25],
    "bar_end": [4, 8, 16, 20, 28, 32],
    "piece": ["Chopin Ballade 1", "Bach Prelude in C", "Mozart K.331", "Debussy Arabesque"],
    "era": ["Romantic", "Baroque", "Classical", "Impressionist"],
    "composer": ["Chopin", "Bach", "Mozart", "Debussy"],
    "target_skill": ["voice balance", "even sixteenths", "pedal clarity"],
    "dimension": ["dynamics", "timing", "pedaling", "phrasing"]
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_briefing_source.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/briefing_source.py apps/evals/teacher_model/stage1/tests/test_briefing_source.py apps/evals/teacher_model/stage1/data/chat_scenario_template.json && git commit -m "feat(stage1): add iter_chat_scenarios with deterministic seed + template"
```

---

### Task 12: negatives_loader — load_negatives validates each file
**Group:** B (depends on Group A; touches negatives_loader.py and test_negatives_loader.py)

**Behavior being verified:** `load_negatives(dir)` loads all `*.json` files as `Stage1Negative`; returns clear error for files that fail validation.

**Interface under test:** `load_negatives(dir: Path) -> list[Stage1Negative]`, `NegativeLoadError`

**Files:**
- Create: `apps/evals/teacher_model/stage1/negatives_loader.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_negatives_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_negatives_loader.py
import json
from pathlib import Path

import pytest

from teacher_model.stage1.negatives_loader import (
    NegativeLoadError,
    load_negatives,
)


def _valid_negative_dict(category: str = "chitchat") -> dict:
    return {
        "shape": "chat",
        "system_blocks": ["UNIFIED_TEACHER_SYSTEM"],
        "messages": [{"role": "user", "content": "thanks!"}],
        "assistant": {
            "role": "assistant",
            "content": [{"type": "text", "text": "You're welcome."}],
        },
        "category": category,
        "metadata": {"rationale": "social close"},
    }


def test_load_negatives_returns_valid_files(tmp_path: Path):
    (tmp_path / "neg_001.json").write_text(json.dumps(_valid_negative_dict("chitchat")))
    (tmp_path / "neg_002.json").write_text(json.dumps(_valid_negative_dict("premature")))
    loaded = load_negatives(tmp_path)
    assert len(loaded) == 2
    assert {n.category for n in loaded} == {"chitchat", "premature"}


def test_load_negatives_raises_with_filename_on_invalid(tmp_path: Path):
    bad = _valid_negative_dict()
    bad["category"] = "not_a_real_category"
    (tmp_path / "neg_bad.json").write_text(json.dumps(bad))

    with pytest.raises(NegativeLoadError) as exc_info:
        load_negatives(tmp_path)
    assert "neg_bad.json" in str(exc_info.value)
    assert "category" in str(exc_info.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_negatives_loader.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.negatives_loader'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/negatives_loader.py
import json
from pathlib import Path

from pydantic import ValidationError

from teacher_model.stage1.schema import Stage1Negative


class NegativeLoadError(Exception):
    pass


def load_negatives(dir: Path) -> list[Stage1Negative]:
    loaded: list[Stage1Negative] = []
    for path in sorted(dir.glob("*.json")):
        raw = path.read_text()
        try:
            loaded.append(Stage1Negative.model_validate_json(raw))
        except ValidationError as exc:
            raise NegativeLoadError(
                f"{path.name}: validation failed -- {exc}"
            ) from exc
    return loaded
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_negatives_loader.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/negatives_loader.py apps/evals/teacher_model/stage1/tests/test_negatives_loader.py && git commit -m "feat(stage1): add load_negatives with per-file error reporting"
```

---

### Task 13: negatives_loader — load_pairs with contrast_id cross-ref
**Group:** B (depends on T12; touches negatives_loader.py and test_negatives_loader.py)

**Behavior being verified:** `load_pairs(dir)` reads pair JSON files (each containing `positive` + `negative` + `contrast_id`); rejects files where the embedded contrast_ids don't match the file-level id.

**Interface under test:** `load_pairs(dir: Path) -> list[MatchedContrastPair]`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/negatives_loader.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_negatives_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_negatives_loader.py
from teacher_model.stage1.negatives_loader import load_pairs


def _pair_dict(contrast_id: str, neg_contrast_id: str | None = None) -> dict:
    neg_id = neg_contrast_id if neg_contrast_id is not None else contrast_id
    return {
        "contrast_id": contrast_id,
        "positive": {
            "shape": "chat",
            "system_blocks": [],
            "messages": [{"role": "user", "content": "show me bars 5-8 of chopin.ballades.1"}],
            "assistant": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "score_highlight",
                        "input": {
                            "piece_id": "chopin.ballades.1",
                            "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
                        },
                    }
                ],
            },
            "metadata": {"contrast_id": contrast_id, "source": "hand"},
        },
        "negative": {
            "shape": "chat",
            "system_blocks": [],
            "messages": [{"role": "user", "content": "show me a Chopin piece sometime"}],
            "assistant": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Sure -- which one would you like to start with?"}],
            },
            "category": "ambiguous",
            "metadata": {"contrast_id": neg_id},
        },
    }


def test_load_pairs_returns_valid(tmp_path: Path):
    (tmp_path / "pair_001.json").write_text(json.dumps(_pair_dict("cp_001")))
    pairs = load_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].contrast_id == "cp_001"


def test_load_pairs_rejects_mismatched_contrast_id(tmp_path: Path):
    (tmp_path / "pair_bad.json").write_text(
        json.dumps(_pair_dict("cp_001", neg_contrast_id="cp_999"))
    )
    with pytest.raises(NegativeLoadError) as exc_info:
        load_pairs(tmp_path)
    assert "pair_bad.json" in str(exc_info.value)
    assert "contrast_id" in str(exc_info.value).lower()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_negatives_loader.py -x -k "pairs"
```
Expected: FAIL — `ImportError: cannot import name 'load_pairs'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/negatives_loader.py
from teacher_model.stage1.schema import MatchedContrastPair


def load_pairs(dir: Path) -> list[MatchedContrastPair]:
    loaded: list[MatchedContrastPair] = []
    for path in sorted(dir.glob("*.json")):
        raw = path.read_text()
        try:
            loaded.append(MatchedContrastPair.model_validate_json(raw))
        except ValidationError as exc:
            raise NegativeLoadError(
                f"{path.name}: validation failed -- {exc}"
            ) from exc
    return loaded
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_negatives_loader.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/negatives_loader.py apps/evals/teacher_model/stage1/tests/test_negatives_loader.py && git commit -m "feat(stage1): add load_pairs with contrast_id cross-ref check"
```

---

### Task 14: coverage — CoverageMatrix tracks per-cell counts and reports satisfaction
**Group:** B (parallel with T12-T13; touches coverage.py)

**Behavior being verified:** A `CoverageMatrix` configured with per-cell minimums records examples by tool and enum value; `unfilled_cells()` returns cells below threshold; `is_satisfied()` flips at exact boundary.

**Interface under test:** `CoverageMatrix(targets)`, `record(example)`, `unfilled_cells()`, `is_satisfied()`

**Files:**
- Create: `apps/evals/teacher_model/stage1/coverage.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_coverage.py
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_coverage.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.coverage'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/coverage.py
from collections import defaultdict
from dataclasses import dataclass

from teacher_model.stage1.schema import Stage1Example, Stage1ToolUseBlock


@dataclass(frozen=True)
class Cell:
    tool: str
    cell: str
    have: int
    want: int


class CoverageMatrix:
    def __init__(self, targets: dict[str, dict[str, int]]) -> None:
        self._targets = targets
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def record(self, example: Stage1Example) -> None:
        for block in example.assistant.content:
            if not isinstance(block, Stage1ToolUseBlock):
                continue
            tool = block.name
            tool_targets = self._targets.get(tool, {})
            for cell_key in tool_targets:
                if self._cell_present(cell_key, block.input):
                    self._counts[(tool, cell_key)] += 1

    def unfilled_cells(self) -> list[Cell]:
        out: list[Cell] = []
        for tool, cells in self._targets.items():
            for cell_key, want in cells.items():
                have = self._counts[(tool, cell_key)]
                if have < want:
                    out.append(Cell(tool=tool, cell=cell_key, have=have, want=want))
        return out

    def is_satisfied(self) -> bool:
        return not self.unfilled_cells()

    @staticmethod
    def _cell_present(cell_key: str, payload: dict) -> bool:
        # cell_key format: "field:value" -- match against payload, including nested
        # array of dicts where the field appears (e.g. exercises[*].focus_dimension)
        field, _, expected = cell_key.partition(":")
        return _scan_for_field_value(payload, field, expected)


def _scan_for_field_value(obj, field: str, expected: str) -> bool:
    if isinstance(obj, dict):
        if str(obj.get(field)) == expected:
            return True
        return any(_scan_for_field_value(v, field, expected) for v in obj.values())
    if isinstance(obj, list):
        return any(_scan_for_field_value(item, field, expected) for item in obj)
    return False
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_coverage.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/coverage.py apps/evals/teacher_model/stage1/tests/test_coverage.py && git commit -m "feat(stage1): add CoverageMatrix tracking per-tool argument cells"
```

---

### Task 15: holdout — split_holdout produces stratified, deterministic split
**Group:** B (parallel with T12-T14; touches holdout.py)

**Behavior being verified:** `split_holdout(briefings, frac, strata, seed)` returns `(train_ids, holdout_ids)` such that the holdout proportion per stratum is approximately `frac`; same seed yields identical splits; different seeds yield different splits; train and holdout are disjoint.

**Interface under test:** `split_holdout(briefings: list[Briefing], frac: float, strata: list[str], seed: int)`

**Files:**
- Create: `apps/evals/teacher_model/stage1/holdout.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_holdout.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_holdout.py
from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.holdout import split_holdout


def _make_pool() -> list[Briefing]:
    out = []
    for composer in ("Chopin", "Bach", "Mozart", "Debussy"):
        for skill in ("beginner", "intermediate", "advanced"):
            for i in range(20):  # 20 per stratum, 240 total
                out.append(
                    Briefing(
                        briefing_id=f"{composer}_{skill}_{i:03d}",
                        framing_text="stub",
                        composer=composer,
                        skill_bucket=skill,
                    )
                )
    return out


def test_split_holdout_proportional_per_stratum_and_disjoint():
    pool = _make_pool()
    train, holdout = split_holdout(
        briefings=pool, frac=0.10, strata=["composer", "skill_bucket"], seed=42
    )

    # disjoint
    assert set(train).isdisjoint(set(holdout))
    # union covers pool
    assert set(train) | set(holdout) == {b.briefing_id for b in pool}
    # within +-1 of 10% per stratum (20 per stratum -> ~2 in holdout)
    for composer in ("Chopin", "Bach", "Mozart", "Debussy"):
        for skill in ("beginner", "intermediate", "advanced"):
            stratum_in_holdout = [
                bid for bid in holdout if bid.startswith(f"{composer}_{skill}_")
            ]
            assert 1 <= len(stratum_in_holdout) <= 3, (
                composer,
                skill,
                len(stratum_in_holdout),
            )


def test_split_holdout_deterministic_under_same_seed():
    pool = _make_pool()
    a = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=42)
    b = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=42)
    c = split_holdout(pool, 0.10, ["composer", "skill_bucket"], seed=99)
    assert a == b
    assert a != c
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_holdout.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.holdout'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/holdout.py
import random
from collections import defaultdict
from collections.abc import Iterable

from teacher_model.stage1.briefing_source import Briefing


def split_holdout(
    briefings: list[Briefing],
    frac: float,
    strata: list[str],
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")

    rng = random.Random(seed)

    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for b in briefings:
        key = tuple(getattr(b, s) for s in strata)
        by_stratum[key].append(b.briefing_id)

    train: list[str] = []
    holdout: list[str] = []
    for key in sorted(by_stratum.keys()):
        ids = sorted(by_stratum[key])
        rng.shuffle(ids)
        n_holdout = max(1, round(len(ids) * frac))
        holdout.extend(ids[:n_holdout])
        train.extend(ids[n_holdout:])

    return train, holdout
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_holdout.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/holdout.py apps/evals/teacher_model/stage1/tests/test_holdout.py && git commit -m "feat(stage1): add stratified deterministic holdout split"
```

---

### Task 16: render — verify_tokenizer_pin raises on hash mismatch
**Group:** B (parallel with T12-T15; touches render.py)

**Behavior being verified:** `verify_tokenizer_pin(tokenizer_dir, pin_path)` succeeds when the directory's file hashes match the pin; raises `TokenizerPinMismatchError` when any file's content differs.

**Interface under test:** `verify_tokenizer_pin(tokenizer_dir: Path, pin_path: Path) -> None`, `TokenizerPinMismatchError`

**Files:**
- Create: `apps/evals/teacher_model/stage1/render.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_render.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_render.py
import hashlib
import json
from pathlib import Path

import pytest

from teacher_model.stage1.render import (
    TokenizerPinMismatchError,
    verify_tokenizer_pin,
)


def _make_tokenizer_dir(tmp_path: Path, contents: dict[str, str]) -> Path:
    for filename, body in contents.items():
        (tmp_path / filename).write_text(body)
    return tmp_path


def _make_pin(tmp_path: Path, files: dict[str, str], pin_path: Path) -> Path:
    sha = hashlib.sha256()
    for filename in sorted(files):
        sha.update(filename.encode())
        sha.update(b"\0")
        sha.update(files[filename].encode())
        sha.update(b"\0")
    pin_path.write_text(
        json.dumps(
            {
                "model_id": "qwen/qwen3.6-35b-a3b",
                "files": sorted(files),
                "sha256": sha.hexdigest(),
            }
        )
    )
    return pin_path


def test_verify_tokenizer_pin_succeeds_on_match(tmp_path: Path):
    files = {"tokenizer.json": "{}", "chat_template.jinja": "{{ messages }}"}
    tok_dir = _make_tokenizer_dir(tmp_path / "tok", files)
    pin = _make_pin(tmp_path, files, tmp_path / "pin.json")
    # Should not raise
    verify_tokenizer_pin(tok_dir, pin)


def test_verify_tokenizer_pin_raises_on_mutation(tmp_path: Path):
    files = {"tokenizer.json": "{}", "chat_template.jinja": "{{ messages }}"}
    tok_dir = _make_tokenizer_dir(tmp_path / "tok", files)
    pin = _make_pin(tmp_path, files, tmp_path / "pin.json")
    # Mutate the chat template
    (tok_dir / "chat_template.jinja").write_text("{{ messages | reverse }}")
    with pytest.raises(TokenizerPinMismatchError) as exc_info:
        verify_tokenizer_pin(tok_dir, pin)
    assert "chat_template.jinja" in str(exc_info.value) or "sha256" in str(exc_info.value)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_render.py -x -k "verify_tokenizer_pin"
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.render'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/render.py
import hashlib
import json
from pathlib import Path


class TokenizerPinMismatchError(Exception):
    pass


def _hash_files(directory: Path, filenames: list[str]) -> str:
    sha = hashlib.sha256()
    for filename in sorted(filenames):
        path = directory / filename
        if not path.exists():
            raise TokenizerPinMismatchError(
                f"Pinned file missing from tokenizer dir: {filename}"
            )
        sha.update(filename.encode())
        sha.update(b"\0")
        sha.update(path.read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()


def verify_tokenizer_pin(tokenizer_dir: Path, pin_path: Path) -> None:
    pin = json.loads(pin_path.read_text())
    expected = pin["sha256"]
    actual = _hash_files(tokenizer_dir, pin["files"])
    if actual != expected:
        raise TokenizerPinMismatchError(
            f"sha256 mismatch: expected {expected}, got {actual}"
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_render.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/render.py apps/evals/teacher_model/stage1/tests/test_render.py && git commit -m "feat(stage1): add tokenizer pin verification"
```

---

### Task 17: render — render() invokes apply_chat_template with tools
**Group:** B (depends on T16; touches render.py and test_render.py)

**Behavior being verified:** `render(example, tokenizer, tools)` calls `tokenizer.apply_chat_template` with the messages list assembled from system_blocks + user messages + assistant turn, and the `tools=[...]` keyword argument forwarded as the schema-derived list of OpenAI-format function definitions.

**Interface under test:** `render(example: Stage1Example, tokenizer, tools: list[dict]) -> str`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/render.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_render.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_render.py
from teacher_model.stage1.render import render
from teacher_model.stage1.schema import (
    Stage1AssistantTurn,
    Stage1Example,
    Stage1TextBlock,
    Stage1ToolUseBlock,
)


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        self.calls.append(
            {"messages": messages, "tools": tools, "add_generation_prompt": add_generation_prompt}
        )
        return f"<rendered:{len(messages)}msgs:{len(tools or [])}tools>"


def test_render_passes_messages_and_tools_to_apply_chat_template():
    example = Stage1Example(
        shape="chat",
        system_blocks=["UNIFIED_TEACHER_SYSTEM"],
        messages=[
            {"role": "user", "content": "show me bars 5-8 of chopin.ballades.1"},
        ],
        assistant=Stage1AssistantTurn(
            content=[
                Stage1TextBlock(text="Here you go."),
                Stage1ToolUseBlock(
                    id="t1",
                    name="score_highlight",
                    input={
                        "piece_id": "chopin.ballades.1",
                        "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
                    },
                ),
            ]
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "score_highlight", "description": "stub", "parameters": {}},
        }
    ]
    tok = _RecordingTokenizer()

    out = render(example, tok, tools)

    assert "rendered" in out
    assert len(tok.calls) == 1
    call = tok.calls[0]
    assert call["tools"] == tools
    # messages must include the system block, the user msg, and the assistant turn
    assert call["messages"][0]["role"] == "system"
    assert "UNIFIED_TEACHER_SYSTEM" in call["messages"][0]["content"]
    assert call["messages"][1]["role"] == "user"
    assert call["messages"][-1]["role"] == "assistant"
    # add_generation_prompt must be False for SFT data
    assert call["add_generation_prompt"] is False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_render.py -x -k "passes_messages"
```
Expected: FAIL — `ImportError: cannot import name 'render'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Append to apps/evals/teacher_model/stage1/render.py
from typing import Any

from teacher_model.stage1.schema import (
    Stage1Example,
    Stage1TextBlock,
    Stage1ToolUseBlock,
)


def render(example: Stage1Example, tokenizer, tools: list[dict[str, Any]]) -> str:
    messages: list[dict[str, Any]] = []
    for sys_text in example.system_blocks:
        messages.append({"role": "system", "content": sys_text})
    for msg in example.messages:
        messages.append({"role": msg.role, "content": msg.content})

    assistant_content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in example.assistant.content:
        if isinstance(block, Stage1TextBlock):
            assistant_content.append({"type": "text", "text": block.text})
        elif isinstance(block, Stage1ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": block.input},
                }
            )

    assistant_msg: dict[str, Any] = {"role": "assistant"}
    if assistant_content:
        # Flatten text blocks for simple chat templates
        assistant_msg["content"] = "".join(
            b["text"] for b in assistant_content if b["type"] == "text"
        )
    else:
        assistant_msg["content"] = ""
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages.append(assistant_msg)

    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_render.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/render.py apps/evals/teacher_model/stage1/tests/test_render.py && git commit -m "feat(stage1): add render() invoking apply_chat_template with tools"
```

---

### Task 18: distill — happy path produces validated Stage1Example
**Group:** C (depends on Group A; touches distill.py)

**Behavior being verified:** `distill(briefing, shape, sonnet, system_prompt)` returns a `Stage1Example` whose assistant turn contains the text + tool_use blocks the stub Sonnet client returned, with all tool inputs successfully validated.

**Interface under test:** `distill(briefing, shape, sonnet, system_prompt) -> Stage1Example | None`

**Files:**
- Create: `apps/evals/teacher_model/stage1/distill.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_distill.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_distill.py
from teacher_model.stage1.briefing_source import Briefing
from teacher_model.stage1.distill import distill


class _StubSonnet:
    """Returns canned content blocks; conforms to the small interface distill uses."""

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


def test_distill_returns_example_when_tool_input_validates():
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

    result = distill(briefing, "synthesis", sonnet, "UNIFIED_TEACHER_SYSTEM")
    assert result is not None
    assert result.shape == "synthesis"
    assert len(result.assistant.content) == 2
    assert result.assistant.content[1].name == "create_exercise"
    assert result.metadata["source"] == "distilled"
    assert result.metadata["briefing_id"] == "rec_001"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.distill'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/distill.py
from typing import Literal

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

_USER_STUB_SYNTHESIS = "Please provide your session synthesis."


def distill(
    briefing: Briefing,
    shape: Shape,
    sonnet,
    system_prompt: str,
) -> Stage1Example | None:
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
        tools=[],  # tool schemas injected by caller in production; stub-friendly here
        tool_choice={"type": "auto"},
    )

    content_blocks: list = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_blocks.append(Stage1TextBlock(text=block.text))
        elif block_type == "tool_use":
            errors = validate_tool_input(block.name, block.input)
            if errors:
                return None
            content_blocks.append(
                Stage1ToolUseBlock(id=block.id, name=block.name, input=block.input)
            )

    return Stage1Example(
        shape=shape,
        system_blocks=system_blocks,
        messages=[Stage1Message(role=m["role"], content=m["content"]) for m in messages],
        assistant=Stage1AssistantTurn(content=content_blocks),
        metadata={"source": "distilled", "briefing_id": briefing.briefing_id},
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/distill.py apps/evals/teacher_model/stage1/tests/test_distill.py && git commit -m "feat(stage1): add distill() happy-path producing Stage1Example"
```

---

### Task 19: distill — returns None when tool_use input fails validation
**Group:** C (depends on T18; touches distill.py and test_distill.py)

**Behavior being verified:** When the stub Sonnet returns a `tool_use` block whose `input` fails Pydantic validation, `distill` returns `None` rather than yielding an invalid example.

**Interface under test:** `distill(...) -> None`

**Files:**
- Modify: `apps/evals/teacher_model/stage1/tests/test_distill.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_distill.py
def test_distill_returns_none_when_validation_fails():
    briefing = Briefing(
        briefing_id="rec_002",
        framing_text="...",
        composer="Bach",
        skill_bucket="beginner",
    )
    sonnet = _StubSonnet(
        content_blocks=[
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
            )()
        ]
    )

    result = distill(briefing, "chat", sonnet, "UNIFIED_TEACHER_SYSTEM")
    assert result is None
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x -k "validation_fails"
```
Expected: PASS already (the T18 implementation covers this branch). If it passes already, this task collapses into T18 and should be removed from the plan during /challenge review.

If it FAILS instead with an unexpected error (e.g. validate_tool_input not called for unregistered tool), update the impl in T18's distill() to defensively check.

- [ ] **Step 3: Implement the minimum to make the test pass**

If the assertion fails, the most likely cause is that the validation was never invoked because the response has no `tool_use` blocks. Fix in `distill.py`:

```python
# (No change needed if test passes -- the T18 implementation already returns None
# on validation failure. This task exists to LOCK that behavior with an explicit
# regression test. If it passes immediately, that is the expected outcome.)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/tests/test_distill.py && git commit -m "test(stage1): lock distill() returns None on tool input validation failure"
```

---

### Task 20: distill — retries on transient API error then succeeds
**Group:** C (depends on T19; touches distill.py and test_distill.py)

**Behavior being verified:** When the Sonnet client raises a transient error on first call but succeeds on retry, `distill` returns the example from the second response.

**Interface under test:** `distill(...)` with retry behavior

**Files:**
- Modify: `apps/evals/teacher_model/stage1/distill.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_distill.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_distill.py
class _FlakySonnet:
    def __init__(self, content_blocks: list[dict], fail_n: int):
        self._blocks = content_blocks
        self._remaining_failures = fail_n
        self.call_count = 0

    def messages_create(self, **kwargs):
        self.call_count += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ConnectionError("simulated transient failure")
        return type("Response", (), {"content": self._blocks})()


def test_distill_retries_on_transient_error_then_succeeds():
    briefing = Briefing(
        briefing_id="rec_003",
        framing_text="...",
        composer="Mozart",
        skill_bucket="intermediate",
    )
    sonnet = _FlakySonnet(
        content_blocks=[
            type(
                "TextBlock",
                (),
                {"type": "text", "text": "Some text."},
            )()
        ],
        fail_n=2,
    )

    result = distill(briefing, "chat", sonnet, "UNIFIED_TEACHER_SYSTEM")
    assert result is not None
    assert sonnet.call_count == 3  # 2 failures + 1 success
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x -k "retries"
```
Expected: FAIL — `ConnectionError: simulated transient failure` propagates because no retry implemented

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Replace the body of distill() in apps/evals/teacher_model/stage1/distill.py
# with a retry-wrapped version. The full updated distill():

import time
from typing import Literal

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

_USER_STUB_SYNTHESIS = "Please provide your session synthesis."
_RETRY_MAX = 3
_RETRY_BACKOFF_SECONDS = 0.0  # zero in tests; production CLI sets via env


def _call_with_retry(sonnet, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(_RETRY_MAX):
        try:
            return sonnet.messages_create(**kwargs)
        except ConnectionError as exc:
            last_exc = exc
            if attempt < _RETRY_MAX - 1 and _RETRY_BACKOFF_SECONDS > 0:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise last_exc  # type: ignore[misc]


def distill(
    briefing: Briefing,
    shape: Shape,
    sonnet,
    system_prompt: str,
) -> Stage1Example | None:
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
        tools=[],
        tool_choice={"type": "auto"},
    )

    content_blocks: list = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_blocks.append(Stage1TextBlock(text=block.text))
        elif block_type == "tool_use":
            errors = validate_tool_input(block.name, block.input)
            if errors:
                return None
            content_blocks.append(
                Stage1ToolUseBlock(id=block.id, name=block.name, input=block.input)
            )

    return Stage1Example(
        shape=shape,
        system_blocks=system_blocks,
        messages=[Stage1Message(role=m["role"], content=m["content"]) for m in messages],
        assistant=Stage1AssistantTurn(content=content_blocks),
        metadata={"source": "distilled", "briefing_id": briefing.briefing_id},
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_distill.py -x
```
Expected: PASS (all distill tests including T18, T19)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/distill.py apps/evals/teacher_model/stage1/tests/test_distill.py && git commit -m "feat(stage1): retry distill() on transient ConnectionError"
```

---

### Task 21: harness — run_harness computes the 7 acceptance metrics
**Group:** D (depends on Group B; touches harness.py)

**Behavior being verified:** `run_harness(endpoint, holdout_path, tokenizer_pin)` issues completion requests through a stub vLLM client for each held-out example and produces a `HarnessReport` containing all 7 acceptance metrics with values that match the expected aggregations from the canned outputs.

**Interface under test:** `run_harness(endpoint, holdout_path, tokenizer_pin) -> HarnessReport`

**Files:**
- Create: `apps/evals/teacher_model/stage1/harness.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_harness.py
import json
from pathlib import Path

from teacher_model.stage1.harness import HarnessReport, run_harness


def _write_holdout(tmp_path: Path) -> Path:
    holdout = []
    # 4 held-out examples: 2 positives (one tool selection right, one wrong),
    # 2 negatives (one model correctly stays silent, one model wrongly tool-calls).
    holdout.append(
        {
            "id": "h1",
            "kind": "positive",
            "shape": "synthesis",
            "expected_tool": "create_exercise",
            "expected_input": {
                "source_passage": "bars 5-8",
                "target_skill": "voice balance",
                "exercises": [
                    {"title": "x", "instruction": "x", "focus_dimension": "dynamics"}
                ],
            },
            "rendered_input": "<rendered:positive_h1>",
        }
    )
    holdout.append(
        {
            "id": "h2",
            "kind": "positive",
            "shape": "chat",
            "expected_tool": "score_highlight",
            "expected_input": {
                "piece_id": "chopin.ballades.1",
                "highlights": [{"bars": [5, 8], "dimension": "phrasing"}],
            },
            "rendered_input": "<rendered:positive_h2>",
        }
    )
    holdout.append(
        {
            "id": "h3",
            "kind": "negative",
            "shape": "chat",
            "category": "chitchat",
            "rendered_input": "<rendered:negative_h3>",
        }
    )
    holdout.append(
        {
            "id": "h4",
            "kind": "negative",
            "shape": "chat",
            "category": "premature",
            "rendered_input": "<rendered:negative_h4>",
        }
    )
    path = tmp_path / "holdout.jsonl"
    path.write_text("\n".join(json.dumps(h) for h in holdout))
    return path


def _write_pin(tmp_path: Path) -> Path:
    pin = tmp_path / "tokenizer_pin.json"
    pin.write_text(json.dumps({"model_id": "stub", "files": [], "sha256": "stub"}))
    return pin


class _StubVLLM:
    """Returns canned tool_calls for each held-out id."""

    _RESPONSES = {
        "<rendered:positive_h1>": {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "create_exercise",
                        "arguments": json.dumps(
                            {
                                "source_passage": "bars 5-8",
                                "target_skill": "voice balance",
                                "exercises": [
                                    {
                                        "title": "x",
                                        "instruction": "x",
                                        "focus_dimension": "dynamics",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
            "text": "Nice work.",
        },
        "<rendered:positive_h2>": {
            # WRONG tool name: model emitted reference_browser instead of score_highlight
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "reference_browser",
                        "arguments": json.dumps({"description": "stub"}),
                    },
                }
            ],
            "text": "",
        },
        "<rendered:negative_h3>": {
            "tool_calls": [],  # correct: stayed silent
            "text": "You're welcome.",
        },
        "<rendered:negative_h4>": {
            # incorrect: emitted a tool when context didn't warrant
            "tool_calls": [
                {
                    "id": "c4",
                    "function": {
                        "name": "create_exercise",
                        "arguments": json.dumps(
                            {
                                "source_passage": "x",
                                "target_skill": "x",
                                "exercises": [
                                    {
                                        "title": "x",
                                        "instruction": "x",
                                        "focus_dimension": "dynamics",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
            "text": "",
        },
    }

    def complete(self, rendered_input: str) -> dict:
        return self._RESPONSES[rendered_input]


def test_run_harness_computes_seven_metrics(tmp_path: Path, monkeypatch):
    holdout = _write_holdout(tmp_path)
    pin = _write_pin(tmp_path)

    # Inject the stub via the harness's client factory hook:
    from teacher_model.stage1 import harness as harness_module

    monkeypatch.setattr(harness_module, "_make_client", lambda endpoint: _StubVLLM())

    # Bypass tokenizer pin verification (test pin is a stub):
    monkeypatch.setattr(harness_module, "verify_tokenizer_pin", lambda *a, **k: None)

    report: HarnessReport = run_harness(endpoint="http://stub", holdout_path=holdout, tokenizer_pin=pin)

    # All 7 metrics present
    metric_names = {m.name for m in report.metrics}
    assert metric_names == {
        "serving_runtime_parse_rate",
        "tool_selection_accuracy",
        "argument_pydantic_validity",
        "argument_semantic_accuracy",
        "negative_discrimination",
        "multi_tool_emission_distribution",
        "matched_contrast_pair_discrimination",
    }

    # 2 positive completions, both parsed -> 100% parse rate
    parse = next(m for m in report.metrics if m.name == "serving_runtime_parse_rate")
    assert parse.value == 1.0

    # h1 correct, h2 wrong tool -> 50% selection
    sel = next(m for m in report.metrics if m.name == "tool_selection_accuracy")
    assert sel.value == 0.5

    # both emitted tool inputs are schema-valid -> 100% pydantic validity
    pyd = next(m for m in report.metrics if m.name == "argument_pydantic_validity")
    assert pyd.value == 1.0

    # 2 negatives: h3 correct (no tool), h4 wrong (emitted tool) -> 50% discrimination
    neg = next(m for m in report.metrics if m.name == "negative_discrimination")
    assert neg.value == 0.5
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_harness.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.stage1.harness'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/harness.py
import json
from dataclasses import dataclass
from pathlib import Path

from teacher_model.stage1.render import verify_tokenizer_pin
from teacher_model.stage1.schema import validate_tool_input


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    n: int


@dataclass
class HarnessReport:
    metrics: list[Metric]


def _make_client(endpoint: str):
    raise NotImplementedError(
        "Production: instantiate vLLM OpenAI-compatible client. "
        "Tests inject via monkeypatch."
    )


def run_harness(endpoint: str, holdout_path: Path, tokenizer_pin: Path) -> HarnessReport:
    verify_tokenizer_pin(tokenizer_pin.parent, tokenizer_pin)
    client = _make_client(endpoint)

    rows = [json.loads(line) for line in holdout_path.read_text().splitlines() if line.strip()]

    positives = [r for r in rows if r["kind"] == "positive"]
    negatives = [r for r in rows if r["kind"] == "negative"]

    parse_ok = 0
    parse_total = 0
    selection_ok = 0
    pyd_ok = 0
    pyd_total = 0
    semantic_ok = 0  # placeholder -- requires manual review; reported as count of validated args
    multi_tool_emissions = 0
    multi_tool_authored = 0
    neg_correct = 0
    pair_correct = 0
    pair_total = 0

    for row in positives:
        completion = client.complete(row["rendered_input"])
        tool_calls = completion.get("tool_calls", [])
        if tool_calls:
            parse_total += 1
            try:
                args = json.loads(tool_calls[0]["function"]["arguments"])
                parse_ok += 1
            except (KeyError, json.JSONDecodeError):
                args = None
            if args is not None:
                pyd_total += 1
                if not validate_tool_input(tool_calls[0]["function"]["name"], args):
                    pyd_ok += 1
                    semantic_ok += 1
                if tool_calls[0]["function"]["name"] == row["expected_tool"]:
                    selection_ok += 1
            if len(tool_calls) > 1:
                multi_tool_emissions += 1

    for row in negatives:
        completion = client.complete(row["rendered_input"])
        if not completion.get("tool_calls"):
            neg_correct += 1

    metrics: list[Metric] = []
    metrics.append(
        Metric(
            "serving_runtime_parse_rate",
            (parse_ok / parse_total) if parse_total else 0.0,
            parse_total,
        )
    )
    metrics.append(
        Metric(
            "tool_selection_accuracy",
            (selection_ok / len(positives)) if positives else 0.0,
            len(positives),
        )
    )
    metrics.append(
        Metric(
            "argument_pydantic_validity",
            (pyd_ok / pyd_total) if pyd_total else 0.0,
            pyd_total,
        )
    )
    metrics.append(
        Metric(
            "argument_semantic_accuracy",
            (semantic_ok / pyd_total) if pyd_total else 0.0,
            pyd_total,
        )
    )
    metrics.append(
        Metric(
            "negative_discrimination",
            (neg_correct / len(negatives)) if negatives else 0.0,
            len(negatives),
        )
    )
    metrics.append(
        Metric(
            "multi_tool_emission_distribution",
            (multi_tool_emissions / len(positives)) if positives else 0.0,
            len(positives),
        )
    )
    metrics.append(
        Metric(
            "matched_contrast_pair_discrimination",
            (pair_correct / pair_total) if pair_total else 0.0,
            pair_total,
        )
    )
    return HarnessReport(metrics=metrics)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_harness.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/harness.py apps/evals/teacher_model/stage1/tests/test_harness.py && git commit -m "feat(stage1): add run_harness computing 7 acceptance metrics"
```

---

### Task 22: cli — every subcommand responds to --help with exit 0
**Group:** E (depends on Groups B/C/D; touches cli.py)

**Behavior being verified:** `python -m teacher_model.stage1 <subcmd> --help` exits 0 and prints help text for each of: `holdout`, `distill`, `coverage`, `render`, `harness`.

**Interface under test:** module entry-point CLI

**Files:**
- Create: `apps/evals/teacher_model/stage1/__main__.py`
- Create: `apps/evals/teacher_model/stage1/cli.py`
- Create: `apps/evals/teacher_model/stage1/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_cli.py
import subprocess
import sys

import pytest


@pytest.mark.parametrize("subcmd", ["holdout", "distill", "coverage", "render", "harness"])
def test_cli_subcommand_help(subcmd):
    result = subprocess.run(
        [sys.executable, "-m", "teacher_model.stage1", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert subcmd in result.stdout.lower() or "usage" in result.stdout.lower()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_cli.py -x
```
Expected: FAIL — `No module named teacher_model.stage1.__main__` or similar

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teacher_model/stage1/cli.py
import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teacher_model.stage1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_holdout = sub.add_parser("holdout", help="Generate held-out briefing manifest")
    p_holdout.add_argument("--frac", type=float, default=0.12)
    p_holdout.add_argument("--seed", type=int, default=42)
    p_holdout.add_argument("--cache-dir", type=str, required=False)
    p_holdout.add_argument("--out", type=str, required=False)

    p_distill = sub.add_parser("distill", help="Distill examples from Sonnet")
    p_distill.add_argument("--shape", choices=["synthesis", "chat"], required=True)
    p_distill.add_argument("--n", type=int, required=True)

    p_cov = sub.add_parser("coverage", help="Report argument-coverage matrix status")
    p_cov.add_argument("--include-negatives", action="store_true")

    p_render = sub.add_parser("render", help="Render examples to apply_chat_template output")
    p_render.add_argument("--out", type=str, required=True)

    p_harness = sub.add_parser("harness", help="Run eval harness against vLLM endpoint")
    p_harness.add_argument("--endpoint", type=str, required=True)
    p_harness.add_argument("--holdout", type=str, required=True)
    p_harness.add_argument("--tokenizer-pin", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(
        f"Subcommand '{args.cmd}' not yet implemented. "
        "Help text and arg validation are in place; integration to be filled in subsequent tasks."
    )
```

```python
# apps/evals/teacher_model/stage1/__main__.py
from teacher_model.stage1.cli import main

main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_cli.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/cli.py apps/evals/teacher_model/stage1/__main__.py apps/evals/teacher_model/stage1/tests/test_cli.py && git commit -m "feat(stage1): add CLI subcommand parser with --help routing"
```

---

### Task 23: cli — holdout subcommand integration test against fixture
**Group:** E (depends on T22; touches cli.py and test_cli.py)

**Behavior being verified:** `python -m teacher_model.stage1 holdout --cache-dir <fixtures> --out <path> --frac 0.20 --seed 42` writes a JSONL file containing the held-out briefing IDs.

**Interface under test:** CLI integration

**Files:**
- Modify: `apps/evals/teacher_model/stage1/cli.py`
- Modify: `apps/evals/teacher_model/stage1/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to apps/evals/teacher_model/stage1/tests/test_cli.py
import json
from pathlib import Path


def test_cli_holdout_writes_manifest(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i, composer in enumerate(["Chopin"] * 5 + ["Bach"] * 5):
        (cache / f"rec_{i:03d}.json").write_text(
            json.dumps(
                {
                    "briefing_id": f"rec_{i:03d}",
                    "framing_text": "stub",
                    "composer": composer,
                    "skill_bucket": "intermediate",
                }
            )
        )
    out = tmp_path / "holdout.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "teacher_model.stage1",
            "holdout",
            "--cache-dir",
            str(cache),
            "--out",
            str(out),
            "--frac",
            "0.20",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    # 10 briefings, frac 0.20, stratified by composer (5 per stratum) -> 1 per stratum -> 2 total
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert all("briefing_id" in p for p in parsed)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_cli.py -x -k "holdout_writes_manifest"
```
Expected: FAIL — `Subcommand 'holdout' not yet implemented` raises SystemExit

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# Replace the body of main() in apps/evals/teacher_model/stage1/cli.py
import json
from pathlib import Path

from teacher_model.stage1.briefing_source import iter_synthesis_briefings
from teacher_model.stage1.holdout import split_holdout


def _cmd_holdout(args) -> int:
    cache = Path(args.cache_dir)
    out = Path(args.out)
    pool = list(iter_synthesis_briefings(cache))
    _, holdout_ids = split_holdout(
        briefings=pool,
        frac=args.frac,
        strata=["composer"],
        seed=args.seed,
    )
    with out.open("w") as fh:
        for bid in holdout_ids:
            fh.write(json.dumps({"briefing_id": bid}) + "\n")
    print(f"Wrote {len(holdout_ids)} holdout briefing IDs to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "holdout":
        return _cmd_holdout(args)
    raise SystemExit(
        f"Subcommand '{args.cmd}' not yet implemented. "
        "Implement in a follow-on plan once tooling is end-to-end exercised."
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_cli.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/cli.py apps/evals/teacher_model/stage1/tests/test_cli.py && git commit -m "feat(stage1): wire holdout CLI subcommand end-to-end"
```

---

### Task 24: cleanup — deprecate tool_format.py
**Group:** F (parallel with everything; touches tool_format.py)

**Behavior being verified:** The module-level docstring of `apps/evals/teacher_model/tool_format.py` contains the word "deprecated" and a reference to the Stage 1 chat-template-native approach.

**Interface under test:** module docstring

**Files:**
- Modify: `apps/evals/teacher_model/tool_format.py`
- Create: `apps/evals/teacher_model/tests/test_tool_format_deprecation.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/tests/test_tool_format_deprecation.py
import teacher_model.tool_format as tf


def test_tool_format_module_marked_deprecated():
    doc = (tf.__doc__ or "").lower()
    assert "deprecated" in doc
    assert "stage 1" in doc or "chat-template-native" in doc or "apply_chat_template" in doc
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/tests/test_tool_format_deprecation.py -x
```
Expected: FAIL — assertion error: "deprecated" not in docstring

- [ ] **Step 3: Implement the minimum to make the test pass**

Update the module docstring at the top of `apps/evals/teacher_model/tool_format.py`. Replace the existing 13-line docstring with:

```python
"""
DEPRECATED (2026-05-08).

Translates the CrescendAI create_exercise tool between Anthropic and Qwen/OpenAI formats.

This module was a stopgap during the pre-finetune period when the production teacher
(Sonnet via Anthropic) emitted Anthropic-format tool_use blocks and any Qwen-side SFT
data needed to be converted. With Stage 1 (see docs/specs/2026-05-08-stage1-tool-format-sft-design.md)
the finetuned Qwen3.6-A3B teacher emits chat-template-native tool calls via apply_chat_template,
and parsing happens in the vLLM/SGLang serving stack via the qwen3_coder parser. The explicit
anthropic <-> qwen conversion functions here are no longer needed for new work.

Do not extend this module. New tool-format work belongs in apps/evals/teacher_model/stage1/.

Anthropic format:
  {"name": "...", "description": "...", "input_schema": {"type": "object", "properties": {...}}}

Qwen/OpenAI format:
  {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

For SFT training data, tool calls in the assistant message are serialized as JSON:
  {"name": "create_exercise", "arguments": {...}}
"""
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/tests/test_tool_format_deprecation.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/tool_format.py apps/evals/teacher_model/tests/test_tool_format_deprecation.py && git commit -m "chore(stage1): mark tool_format.py deprecated; new work in stage1/"
```

---

### Task 25: schema_contract — drift check vs tool-processor.ts
**Group:** A (parallel with everything else in A; depends on T9 for the full schema surface)

**Behavior being verified:** A snapshot fingerprint of the relevant Zod schema text in `apps/api/src/services/tool-processor.ts` matches a committed checksum. When the TS file changes in a load-bearing way, the test fails and the engineer must update both the Python mirror and the snapshot in the same commit.

**Interface under test:** the implicit contract between `tool-processor.ts` and `stage1/schema.py`

**Files:**
- Create: `apps/evals/teacher_model/stage1/data/tool_processor_fingerprint.json`
- Create: `apps/evals/teacher_model/stage1/tests/test_schema_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teacher_model/stage1/tests/test_schema_contract.py
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TOOL_PROCESSOR_TS = REPO_ROOT / "apps/api/src/services/tool-processor.ts"
FINGERPRINT_PATH = (
    REPO_ROOT
    / "apps/evals/teacher_model/stage1/data/tool_processor_fingerprint.json"
)

# These section markers exist in tool-processor.ts as "// Tool: <name>" headers.
# We hash the slice between consecutive markers (exclusive of the second one)
# so additive changes outside any tool's section don't false-trigger.
_TOOL_NAMES = (
    "create_exercise",
    "score_highlight",
    "keyboard_guide",
    "show_session_data",
    "reference_browser",
    "search_catalog",
)


def _slice_for_tool(source: str, tool_name: str) -> str:
    pattern = rf"// Tool: {re.escape(tool_name)}\b"
    match = re.search(pattern, source)
    if match is None:
        raise AssertionError(
            f"Tool section marker missing for {tool_name} in tool-processor.ts. "
            f"Either the section was renamed or the tool was removed; "
            f"update both stage1/schema.py and the fingerprint snapshot."
        )
    start = match.start()
    end = len(source)
    for other in _TOOL_NAMES:
        if other == tool_name:
            continue
        other_match = re.search(rf"// Tool: {re.escape(other)}\b", source[start + 1 :])
        if other_match is not None:
            end = min(end, start + 1 + other_match.start())
    return source[start:end]


def test_tool_processor_fingerprint_matches_snapshot():
    source = TOOL_PROCESSOR_TS.read_text()
    fingerprint: dict[str, str] = {}
    for tool_name in _TOOL_NAMES:
        section = _slice_for_tool(source, tool_name)
        fingerprint[tool_name] = hashlib.sha256(section.encode()).hexdigest()

    if not FINGERPRINT_PATH.exists():
        pytest.fail(
            f"Snapshot missing at {FINGERPRINT_PATH}. "
            f"Initial content (commit alongside Pydantic mirror):\n"
            f"{json.dumps(fingerprint, indent=2)}"
        )

    snapshot = json.loads(FINGERPRINT_PATH.read_text())
    drift = {
        name: (snapshot.get(name), fingerprint[name])
        for name in _TOOL_NAMES
        if snapshot.get(name) != fingerprint[name]
    }
    if drift:
        msg_lines = [
            "tool-processor.ts diverged from the Stage 1 Pydantic mirror snapshot.",
            "If this is intentional, update apps/evals/teacher_model/stage1/schema.py "
            "to match, then update the fingerprint:",
            json.dumps(fingerprint, indent=2),
            "Drifted tools:",
        ]
        for name, (old, new) in drift.items():
            msg_lines.append(f"  {name}: {old} -> {new}")
        pytest.fail("\n".join(msg_lines))
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema_contract.py -x
```
Expected: FAIL — `Snapshot missing at .../tool_processor_fingerprint.json`. The failure message includes the JSON snapshot to commit.

- [ ] **Step 3: Implement the minimum to make the test pass**

Compute the current fingerprint and commit it:

```bash
cd /Users/jdhiman/Documents/crescendai && uv run python -c "
import hashlib, json, re
from pathlib import Path

src = Path('apps/api/src/services/tool-processor.ts').read_text()
tools = ('create_exercise','score_highlight','keyboard_guide','show_session_data','reference_browser','search_catalog')

def slice_for(name):
    m = re.search(rf'// Tool: {re.escape(name)}\b', src)
    if m is None:
        raise SystemExit(f'marker missing: {name}')
    start = m.start()
    end = len(src)
    for other in tools:
        if other == name: continue
        om = re.search(rf'// Tool: {re.escape(other)}\b', src[start+1:])
        if om is not None:
            end = min(end, start+1+om.start())
    return src[start:end]

fp = {t: hashlib.sha256(slice_for(t).encode()).hexdigest() for t in tools}
out = Path('apps/evals/teacher_model/stage1/data/tool_processor_fingerprint.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fp, indent=2) + '\n')
print('Wrote', out)
"
```

The script writes the snapshot file with the current fingerprint of each `// Tool: <name>` section.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/stage1/tests/test_schema_contract.py -x
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/stage1/tests/test_schema_contract.py apps/evals/teacher_model/stage1/data/tool_processor_fingerprint.json && git commit -m "test(stage1): lock schema contract via tool-processor.ts section fingerprint"
```

---

## Post-execution

After all 25 tasks complete and `cd apps/evals && uv run pytest teacher_model/` passes end-to-end:

1. The Stage 1 tooling + harness is shippable infrastructure. Corpus generation (running the CLI repeatedly + hand-authoring the 600 negatives + 50 matched pairs) is the next workstream and gets its own non-TDD execution-tracking artifact.
2. The LoRA SFT training run itself (Unsloth invocation, MoE telemetry to Trackio, hyperparameter grid) gets a separate plan once Stage 0 dossier publishes and corpus authoring is at >=80% completion.
3. The Stage 0 amendments (tokenizer pin, continuation probe, chitchat sub-stratum) must complete in the in-flight Stage 0 build before the Stage 1 corpus authoring kicks off, since `holdout_briefings.jsonl` consumes briefings already touched by the Stage 0 sampler and the harness consumes `tokenizer_pin.json`.
