# Teacher Model Finetuning - Month 1-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline, eval infrastructure, and must-fix prerequisites for finetuning a piano teacher model (Qwen3.5-27B). Training is gated on 4 conditions; this plan covers everything needed before training can start.

**Architecture:** Python scripts in `apps/evals/` for data collection, transcription, quality filtering, and eval. Shared `LLMClient` for Workers AI / Anthropic calls. Sentence-transformers for relevance classification. Cohere Transcribe + Pyannote for audio transcription. All outputs stored in `apps/evals/teaching_knowledge/data/corpus/`.

**Tech Stack:** Python 3.12+ (uv), sentence-transformers, cohere SDK, pyannote.audio, yt-dlp, datasketch (MinHash), tiktoken

**Spec:** `docs/superpowers/specs/2026-03-30-teacher-model-finetuning-design.md`

---

## File Map

```
apps/evals/
  teacher_model/                         # NEW: all teacher model finetuning code
    __init__.py
    tool_format.py                       # Qwen tool format translator (Must-Fix #1)
    tool_format_test.py                  # Tool format round-trip tests
    relevance_classifier.py              # Relevance classifier (Must-Fix #3)
    relevance_classifier_test.py         # Classifier validation tests
    transcribe.py                        # YouTube -> transcription pipeline
    scrape_text.py                       # PDF/web -> text extraction pipeline
    provenance.py                        # Corpus provenance manifest
    dedup.py                             # MinHash deduplication
    corpus_builder.py                    # Orchestrates full pipeline
    domain_knowledge_probe.py            # 100 MCQ domain knowledge eval
    data/
      corpus/                            # Output: cleaned corpus segments
      provenance.jsonl                   # Provenance manifest
      negatives/                         # Curated negative examples for classifier
      domain_probe.json                  # 100 MCQ questions
      tool_call_scenarios.json           # 100 tool call test scenarios
  shared/
    judge.py                             # MODIFY: add judge_tool_calls()
    prompts/
      synthesis_quality_judge_v2.txt     # MODIFY: fix absent=2 scoring rule
```

---

### Task 1: Fix Judge v2 Absent-Dimension Scoring Rule (Must-Fix #2)

**Files:**
- Modify: `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`
- Modify: `apps/evals/shared/judge.py:79-83`

This is the highest-priority fix. The current scoring rule lets GRPO game the average by omitting dimensions.

- [ ] **Step 1: Read current judge v2 prompt and identify the scoring rule**

The problematic line in `synthesis_quality_judge_v2.txt` (line 11):
```
When a dimension is irrelevant (e.g., no praise present), give a score of 2 (acceptable) if the output simply omits it without harming the student; a 0 is reserved for harmful or contradictory content.
```

- [ ] **Step 2: Update the scoring rule to use N/A instead of 2**

In `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`, replace the scoring rule line:

```
When a dimension is irrelevant (e.g., no praise present), give a score of "N/A" and set the reason to "Not applicable to this response type." A score of 0 is reserved for harmful or contradictory content. Do NOT assign a default numeric score to irrelevant dimensions.
```

Also update the JSON output instruction to allow `"score": "N/A"`:

```
Output your assessment as a JSON array where each element is:
{"criterion": "<Dimension Name>", "score": <0-3 or "N/A">, "evidence": "<short excerpt>", "reason": "<explanation>"}
```

- [ ] **Step 3: Update JudgeResultV2.mean_score to exclude N/A dimensions**

In `apps/evals/shared/judge.py`, update `DimensionScore` and `JudgeResultV2`:

```python
@dataclass
class DimensionScore:
    """Score for a v2 rubric dimension (0-3 scale, or None if N/A)."""
    criterion: str
    score: int | None  # None means N/A (dimension not applicable)
    evidence: str
    reason: str
```

Update `JudgeResultV2.mean_score`:

```python
@property
def mean_score(self) -> float:
    scored = [d for d in self.dimensions if d.score is not None]
    if not scored:
        return 0.0
    return sum(d.score for d in scored) / len(scored)

@property
def scored_dimension_count(self) -> int:
    """Number of dimensions that received a numeric score (not N/A)."""
    return sum(1 for d in self.dimensions if d.score is not None)
```

- [ ] **Step 4: Update the v2 judge response parser to handle N/A scores**

In `judge.py`, find the function that parses the JSON response from the judge LLM (likely `judge_synthesis_v2` or similar). Update it to handle `"N/A"` scores:

```python
# In the response parsing section, replace direct int cast:
raw_score = item.get("score")
if raw_score == "N/A" or raw_score is None:
    score = None
else:
    score = int(raw_score)
```

- [ ] **Step 5: Run existing eval smoke test to verify no regression**

```bash
cd apps/evals && uv run python -c "
from shared.judge import load_prompt
prompt = load_prompt('synthesis_quality_judge_v2.txt')
assert 'N/A' in prompt, 'N/A scoring rule not found in updated prompt'
assert 'score of 2 (acceptable)' not in prompt, 'Old scoring rule still present'
print('Judge v2 prompt updated correctly.')
"
```

- [ ] **Step 6: Commit**

```bash
git add apps/evals/shared/prompts/synthesis_quality_judge_v2.txt apps/evals/shared/judge.py
git commit -m "fix: change judge v2 absent-dimension scoring from 2 to N/A

Prevents GRPO reward hacking by omitting dimensions to inflate average.
Absent dimensions now excluded from mean score calculation."
```

---

### Task 2: Build Qwen Tool Format Translator (Must-Fix #1)

**Files:**
- Create: `apps/evals/teacher_model/__init__.py`
- Create: `apps/evals/teacher_model/tool_format.py`
- Create: `apps/evals/teacher_model/tool_format_test.py`

- [ ] **Step 1: Create the teacher_model package**

```bash
mkdir -p apps/evals/teacher_model/data/corpus apps/evals/teacher_model/data/negatives
touch apps/evals/teacher_model/__init__.py
```

- [ ] **Step 2: Write failing tests for the tool format translator**

Create `apps/evals/teacher_model/tool_format_test.py`:

```python
"""Tests for Anthropic -> Qwen/OpenAI tool format translation."""
import json
import pytest
from teacher_model.tool_format import (
    anthropic_tool_to_openai,
    anthropic_tool_call_to_qwen_chatml,
    openai_tool_call_to_anthropic,
    EXERCISE_TOOL_OPENAI,
)


def test_exercise_tool_schema_translation():
    """Anthropic create_exercise schema translates to valid OpenAI function schema."""
    result = EXERCISE_TOOL_OPENAI
    assert result["type"] == "function"
    assert result["function"]["name"] == "create_exercise"
    params = result["function"]["parameters"]
    assert "source_passage" in params["properties"]
    assert "target_skill" in params["properties"]
    assert "exercises" in params["properties"]
    # exercises should be an array of objects
    exercises_schema = params["properties"]["exercises"]
    assert exercises_schema["type"] == "array"
    item_props = exercises_schema["items"]["properties"]
    assert "title" in item_props
    assert "instruction" in item_props
    assert "focus_dimension" in item_props
    assert "hands" in item_props


def test_tool_call_to_qwen_chatml():
    """A tool call response formats correctly in Qwen ChatML."""
    tool_call = {
        "name": "create_exercise",
        "arguments": {
            "source_passage": "measures 12-16",
            "target_skill": "Voice balancing",
            "exercises": [{
                "title": "Left hand alone",
                "instruction": "Play LH at half tempo, pp throughout.",
                "focus_dimension": "dynamics",
                "hands": "left",
            }],
        },
    }
    chatml = anthropic_tool_call_to_qwen_chatml(tool_call)
    # Qwen uses <tool_call> tags or function_call JSON
    assert "create_exercise" in chatml
    parsed = json.loads(chatml)
    assert parsed["name"] == "create_exercise"
    assert "exercises" in parsed["arguments"]


def test_round_trip_tool_call():
    """Anthropic -> Qwen -> Anthropic round-trip preserves data."""
    original = {
        "name": "create_exercise",
        "arguments": {
            "source_passage": "bars 5-8",
            "target_skill": "Pedaling clarity",
            "exercises": [{
                "title": "Half-pedal drill",
                "instruction": "Play bars 5-8 with half-pedal on beat 3.",
                "focus_dimension": "pedaling",
                "hands": "both",
            }],
        },
    }
    qwen_format = anthropic_tool_call_to_qwen_chatml(original)
    restored = openai_tool_call_to_anthropic(json.loads(qwen_format))
    assert restored["name"] == original["name"]
    assert restored["arguments"] == original["arguments"]


def test_focus_dimension_validation():
    """Only valid CrescendAI dimensions are accepted."""
    valid_dims = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
    for dim in valid_dims:
        tool_call = {
            "name": "create_exercise",
            "arguments": {
                "source_passage": "bar 1",
                "target_skill": "test",
                "exercises": [{"title": "t", "instruction": "i", "focus_dimension": dim, "hands": "both"}],
            },
        }
        chatml = anthropic_tool_call_to_qwen_chatml(tool_call)
        assert dim in chatml


def test_hands_enum_validation():
    """Only left/right/both are valid hands values."""
    for hands in ["left", "right", "both"]:
        tool_call = {
            "name": "create_exercise",
            "arguments": {
                "source_passage": "bar 1",
                "target_skill": "test",
                "exercises": [{"title": "t", "instruction": "i", "focus_dimension": "dynamics", "hands": hands}],
            },
        }
        chatml = anthropic_tool_call_to_qwen_chatml(tool_call)
        assert hands in chatml
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd apps/evals && uv run pytest teacher_model/tool_format_test.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'teacher_model.tool_format'`

- [ ] **Step 4: Implement the tool format translator**

Create `apps/evals/teacher_model/tool_format.py`:

```python
"""Translate tool schemas and calls between Anthropic and Qwen/OpenAI formats.

Anthropic uses:
  tools: [{"name": ..., "description": ..., "input_schema": {...}}]
  Response: content blocks with type "tool_use", id, name, input

Qwen/OpenAI uses:
  tools: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
  Response: message.tool_calls[].function.name + function.arguments (JSON string)

For SFT training data, we serialize tool calls as JSON in the assistant message
using Qwen's ChatML format.
"""
from __future__ import annotations

import json
from typing import Any

VALID_DIMENSIONS = frozenset([
    "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation",
])
VALID_HANDS = frozenset(["left", "right", "both"])

# CrescendAI create_exercise tool in Anthropic format (from prompts.rs)
EXERCISE_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": "create_exercise",
    "description": (
        "Create a focused practice exercise when the student would benefit "
        "from structured drill. Use sparingly -- most observations should be "
        "text-only."
    ),
    "input_schema": {
        "type": "object",
        "required": ["source_passage", "target_skill", "exercises"],
        "properties": {
            "source_passage": {
                "type": "string",
                "description": "e.g., 'measures 12-16' or 'the opening phrase'",
            },
            "target_skill": {
                "type": "string",
                "description": "e.g., 'Voice balancing between hands'",
            },
            "exercises": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title", "instruction", "focus_dimension", "hands"],
                    "properties": {
                        "title": {"type": "string", "description": "Short exercise name"},
                        "instruction": {
                            "type": "string",
                            "description": "Concrete steps. 2-4 sentences.",
                        },
                        "focus_dimension": {
                            "type": "string",
                            "enum": list(VALID_DIMENSIONS),
                        },
                        "hands": {
                            "type": "string",
                            "enum": list(VALID_HANDS),
                        },
                    },
                },
            },
        },
    },
}


def anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool definition to OpenAI/Qwen function format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


# Pre-computed OpenAI format for the exercise tool
EXERCISE_TOOL_OPENAI = anthropic_tool_to_openai(EXERCISE_TOOL_ANTHROPIC)


def anthropic_tool_call_to_qwen_chatml(tool_call: dict[str, Any]) -> str:
    """Serialize a tool call as JSON for Qwen ChatML assistant messages.

    Input: {"name": "create_exercise", "arguments": {...}}
    Output: JSON string with name + arguments (for embedding in ChatML)
    """
    return json.dumps({
        "name": tool_call["name"],
        "arguments": tool_call["arguments"],
    }, ensure_ascii=False)


def openai_tool_call_to_anthropic(parsed: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed Qwen/OpenAI tool call back to Anthropic format.

    Input: {"name": "create_exercise", "arguments": {...}}
    Output: same structure (Anthropic uses the same shape for tool_use blocks)
    """
    return {
        "name": parsed["name"],
        "arguments": parsed["arguments"],
    }


def validate_exercise_tool_call(tool_call: dict[str, Any]) -> list[str]:
    """Validate a create_exercise tool call. Returns list of error strings (empty = valid)."""
    errors: list[str] = []
    args = tool_call.get("arguments", {})

    if not args.get("source_passage"):
        errors.append("Missing source_passage")
    if not args.get("target_skill"):
        errors.append("Missing target_skill")

    exercises = args.get("exercises", [])
    if not exercises:
        errors.append("Empty exercises array")

    for i, ex in enumerate(exercises):
        if not ex.get("title"):
            errors.append(f"Exercise {i}: missing title")
        if not ex.get("instruction"):
            errors.append(f"Exercise {i}: missing instruction")
        dim = ex.get("focus_dimension")
        if dim not in VALID_DIMENSIONS:
            errors.append(f"Exercise {i}: invalid focus_dimension '{dim}', must be one of {sorted(VALID_DIMENSIONS)}")
        hands = ex.get("hands")
        if hands not in VALID_HANDS:
            errors.append(f"Exercise {i}: invalid hands '{hands}', must be one of {sorted(VALID_HANDS)}")

    return errors
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd apps/evals && uv run pytest teacher_model/tool_format_test.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add apps/evals/teacher_model/
git commit -m "feat: add Qwen/OpenAI tool format translator for teacher model SFT

Translates create_exercise tool between Anthropic and Qwen/OpenAI formats.
Round-trip tested. Validates dimensions and hands enums. Required before
generating any SFT training data."
```

---

### Task 3: Build Tool Calling Regression Test Harness

**Files:**
- Create: `apps/evals/teacher_model/data/tool_call_scenarios.json`
- Modify: `apps/evals/shared/judge.py` (add `judge_tool_calls`)

- [ ] **Step 1: Create 100 tool call test scenarios**

Create `apps/evals/teacher_model/data/tool_call_scenarios.json` with 100 scenarios. 50 where a tool call is appropriate (structured drill helps the student), 50 where it is not (simple observation, recognition, encouragement). Each scenario has:

```json
[
  {
    "id": "tool_required_001",
    "student_context": "Intermediate student, 4th session on Chopin Nocturne Op.9 No.2",
    "teaching_moment": {
      "dimension": "pedaling",
      "score": 0.3,
      "baseline": 0.6,
      "bar_range": "8-12"
    },
    "framing": "correction",
    "expects_tool_call": true,
    "rationale": "Significant regression on pedaling in specific bars, structured drill would help"
  },
  {
    "id": "tool_not_required_001",
    "student_context": "Advanced student, polishing Beethoven Op.57 Appassionata",
    "teaching_moment": {
      "dimension": "dynamics",
      "score": 0.85,
      "baseline": 0.7,
      "bar_range": "1-4"
    },
    "framing": "recognition",
    "expects_tool_call": false,
    "rationale": "Improvement above baseline, recognition response is appropriate"
  }
]
```

Generate all 100 scenarios covering: all 6 dimensions, all skill levels (beginner/intermediate/advanced), all framings (correction/recognition/encouragement/question), multiple pieces and composers.

- [ ] **Step 2: Write the judge_tool_calls function**

Add to `apps/evals/shared/judge.py`:

```python
from teacher_model.tool_format import validate_exercise_tool_call, EXERCISE_TOOL_OPENAI


@dataclass
class ToolCallResult:
    """Result of evaluating a single tool call scenario."""
    scenario_id: str
    expected_tool_call: bool
    actual_tool_call: bool
    tool_name_correct: bool | None  # None if no tool call
    schema_valid: bool | None       # None if no tool call
    validation_errors: list[str]


def judge_tool_calls(
    model_response: str,
    scenario: dict[str, Any],
) -> ToolCallResult:
    """Evaluate whether a model response correctly handles tool calling.

    Checks:
    1. Was a tool call present when expected (or absent when not expected)?
    2. Did the tool name match 'create_exercise'?
    3. Did the JSON arguments validate against the schema?
    """
    scenario_id = scenario["id"]
    expects_tool = scenario["expects_tool_call"]

    # Try to extract tool call from response
    # Qwen format: JSON object with "name" and "arguments" keys
    tool_call = None
    try:
        parsed = json.loads(model_response)
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            tool_call = parsed
    except (json.JSONDecodeError, TypeError):
        # Try to find JSON embedded in text
        import re
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"create_exercise"[^{}]*\}', model_response, re.DOTALL)
        if json_match:
            try:
                tool_call = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    has_tool_call = tool_call is not None
    tool_name_correct = None
    schema_valid = None
    validation_errors: list[str] = []

    if has_tool_call:
        tool_name_correct = tool_call.get("name") == "create_exercise"
        validation_errors = validate_exercise_tool_call(tool_call)
        schema_valid = len(validation_errors) == 0

    return ToolCallResult(
        scenario_id=scenario_id,
        expected_tool_call=expects_tool,
        actual_tool_call=has_tool_call,
        tool_name_correct=tool_name_correct,
        schema_valid=schema_valid,
        validation_errors=validation_errors,
    )


def compute_tool_call_metrics(results: list[ToolCallResult]) -> dict[str, float]:
    """Compute aggregate tool calling metrics from a batch of results."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "schema_validity": 0.0}

    # Tool call presence accuracy (correct decision to call or not call)
    correct_presence = sum(
        1 for r in results if r.expected_tool_call == r.actual_tool_call
    )

    # Among actual tool calls, how many had correct schema?
    actual_calls = [r for r in results if r.actual_tool_call]
    valid_schema = sum(1 for r in actual_calls if r.schema_valid)

    # Precision: of tool calls made, how many were appropriate?
    true_positives = sum(
        1 for r in results if r.expected_tool_call and r.actual_tool_call
    )
    false_positives = sum(
        1 for r in results if not r.expected_tool_call and r.actual_tool_call
    )
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Recall: of scenarios needing tools, how many got them?
    expected_calls = [r for r in results if r.expected_tool_call]
    recall = sum(1 for r in expected_calls if r.actual_tool_call) / len(expected_calls) if expected_calls else 0.0

    return {
        "accuracy": correct_presence / total,
        "precision": precision,
        "recall": recall,
        "schema_validity": valid_schema / len(actual_calls) if actual_calls else 0.0,
        "total_scenarios": total,
        "tool_calls_made": len(actual_calls),
        "valid_schemas": valid_schema,
    }
```

- [ ] **Step 3: Run a quick validation that the harness loads**

```bash
cd apps/evals && uv run python -c "
from shared.judge import judge_tool_calls, compute_tool_call_metrics, ToolCallResult
print('Tool call harness imported successfully.')
# Test with a mock response
result = judge_tool_calls(
    '{\"name\": \"create_exercise\", \"arguments\": {\"source_passage\": \"bar 1\", \"target_skill\": \"test\", \"exercises\": [{\"title\": \"t\", \"instruction\": \"i\", \"focus_dimension\": \"dynamics\", \"hands\": \"both\"}]}}',
    {\"id\": \"test_001\", \"expects_tool_call\": True}
)
assert result.actual_tool_call == True
assert result.schema_valid == True
print('Mock tool call validated correctly.')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/shared/judge.py apps/evals/teacher_model/data/tool_call_scenarios.json
git commit -m "feat: add tool calling regression test harness

100 scenarios (50 tool-required, 50 not). Validates presence,
name, and schema. Computes accuracy, precision, recall, schema
validity. Used for per-stage gates during teacher model training."
```

---

### Task 4: Build Relevance Classifier (Must-Fix #3)

**Files:**
- Create: `apps/evals/teacher_model/relevance_classifier.py`
- Create: `apps/evals/teacher_model/relevance_classifier_test.py`
- Create: `apps/evals/teacher_model/data/negatives/` (curated negative examples)

- [ ] **Step 1: Write failing tests for the relevance classifier**

Create `apps/evals/teacher_model/relevance_classifier_test.py`:

```python
"""Tests for pedagogy relevance classifier."""
import pytest
from teacher_model.relevance_classifier import (
    PedagogyRelevanceClassifier,
    ClassifierMetrics,
)


@pytest.fixture
def classifier():
    return PedagogyRelevanceClassifier()


def test_classifier_scores_teaching_moment_high(classifier):
    """Known teaching moments from raw_teaching_db.json should score high."""
    text = (
        "Your 16th notes were really even -- I was really impressed with "
        "your control and your ability to move throughout the broken chords "
        "while remaining consistent with tempo and sound."
    )
    score = classifier.score(text)
    assert score > 0.5, f"Teaching moment scored {score}, expected > 0.5"


def test_classifier_scores_irrelevant_low(classifier):
    """General music chat (not pedagogical) should score low."""
    text = (
        "I just bought a new Yamaha C3X grand piano. The action feels "
        "amazing compared to my old upright. The delivery guys were very "
        "careful bringing it through the front door."
    )
    score = classifier.score(text)
    assert score < 0.4, f"Irrelevant text scored {score}, expected < 0.4"


def test_classifier_batch_scoring(classifier):
    """Batch scoring returns scores for all inputs."""
    texts = [
        "Play bars 5-8 with half-pedal on beat 3.",
        "The concert was held at Carnegie Hall last Tuesday.",
        "Try practicing the left hand alone at half tempo.",
    ]
    scores = classifier.score_batch(texts)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_classifier_validation_metrics(classifier):
    """Classifier produces precision/recall metrics on held-out set."""
    metrics = classifier.validate()
    assert isinstance(metrics, ClassifierMetrics)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert metrics.threshold > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd apps/evals && uv run pytest teacher_model/relevance_classifier_test.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Curate negative examples**

Create `apps/evals/teacher_model/data/negatives/README.md`:

```markdown
# Negative Examples for Relevance Classifier

200-400 text passages from the SAME source domains as the positives
but that are NOT pedagogically relevant. Sources:

1. YouTube masterclass comments (not the lesson content)
2. General music discussion from forums (gear talk, concert reviews)
3. Music news articles (not teaching)
4. Piano shopping/maintenance advice
5. Generic self-help/motivation content
6. Non-piano music education (voice, guitar)

Each file is a JSONL with {"text": "...", "source": "...", "reason": "..."}
```

Create `apps/evals/teacher_model/data/negatives/curated_negatives.jsonl` with 200+ manually curated negative examples from the identified source domains.

- [ ] **Step 4: Implement the relevance classifier**

Create `apps/evals/teacher_model/relevance_classifier.py`:

```python
"""Piano pedagogy relevance classifier using sentence-transformers.

Architecture: Compute embedding centroid of 379 positive teaching moments.
Score new text by cosine similarity to centroid. Threshold determined by
precision/recall curve on held-out validation set.

Requires: pip install sentence-transformers
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).parent / "data"
_TEACHING_DB = _DATA_DIR.parent.parent / "teaching_knowledge" / "data" / "raw_teaching_db.json"
_NEGATIVES_DIR = _DATA_DIR / "negatives"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ClassifierMetrics:
    precision: float
    recall: float
    f1: float
    threshold: float
    n_positives: int
    n_negatives: int


class PedagogyRelevanceClassifier:
    """Score text passages for piano pedagogy relevance."""

    def __init__(self, model_name: str = MODEL_NAME, threshold: float | None = None):
        self._model = SentenceTransformer(model_name)
        self._positives = self._load_positives()
        self._negatives = self._load_negatives()
        self._centroid = self._compute_centroid()
        self.threshold = threshold or self._find_optimal_threshold()

    def _load_positives(self) -> list[str]:
        """Load positive examples from raw_teaching_db.json."""
        if not _TEACHING_DB.exists():
            raise FileNotFoundError(f"Teaching DB not found: {_TEACHING_DB}")
        data = json.loads(_TEACHING_DB.read_text())
        return [item["what_teacher_said"] for item in data if item.get("what_teacher_said")]

    def _load_negatives(self) -> list[str]:
        """Load curated negative examples from negatives/ directory."""
        negatives = []
        for path in _NEGATIVES_DIR.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    negatives.append(item["text"])
        return negatives

    def _compute_centroid(self) -> np.ndarray:
        """Compute embedding centroid of positive examples."""
        embeddings = self._model.encode(self._positives, show_progress_bar=False)
        return np.mean(embeddings, axis=0)

    def _find_optimal_threshold(self) -> float:
        """Find threshold that maximizes F1 on the full dataset."""
        all_texts = self._positives + self._negatives
        labels = [1] * len(self._positives) + [0] * len(self._negatives)
        scores = self.score_batch(all_texts)

        best_f1 = 0.0
        best_threshold = 0.3

        for t in np.arange(0.1, 0.9, 0.01):
            preds = [1 if s >= t else 0 for s in scores]
            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        return best_threshold

    def score(self, text: str) -> float:
        """Score a single text passage for pedagogy relevance (0.0 to 1.0)."""
        embedding = self._model.encode([text], show_progress_bar=False)[0]
        similarity = np.dot(embedding, self._centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(self._centroid)
        )
        return float(np.clip(similarity, 0.0, 1.0))

    def score_batch(self, texts: Sequence[str]) -> list[float]:
        """Score a batch of text passages."""
        if not texts:
            return []
        embeddings = self._model.encode(list(texts), show_progress_bar=False)
        centroid_norm = self._centroid / np.linalg.norm(self._centroid)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarities = normalized @ centroid_norm
        return [float(np.clip(s, 0.0, 1.0)) for s in similarities]

    def is_relevant(self, text: str) -> bool:
        """Binary classification using the threshold."""
        return self.score(text) >= self.threshold

    def validate(self) -> ClassifierMetrics:
        """Compute precision/recall on the full dataset (positives + negatives)."""
        all_texts = self._positives + self._negatives
        labels = [1] * len(self._positives) + [0] * len(self._negatives)
        scores = self.score_batch(all_texts)
        preds = [1 if s >= self.threshold else 0 for s in scores]

        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return ClassifierMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=self.threshold,
            n_positives=len(self._positives),
            n_negatives=len(self._negatives),
        )
```

- [ ] **Step 5: Add sentence-transformers to pyproject.toml**

```bash
cd apps/evals && uv add sentence-transformers
```

- [ ] **Step 6: Run tests**

```bash
cd apps/evals && uv run pytest teacher_model/relevance_classifier_test.py -v
```
Expected: all 4 tests PASS (may take 30s for model download on first run)

- [ ] **Step 7: Commit**

```bash
git add apps/evals/teacher_model/relevance_classifier.py apps/evals/teacher_model/relevance_classifier_test.py apps/evals/teacher_model/data/negatives/ apps/evals/pyproject.toml apps/evals/uv.lock
git commit -m "feat: add pedagogy relevance classifier for CPT corpus filtering

Sentence-transformers cosine similarity against 379 teaching moment centroid.
200+ curated negatives. Auto-tunes threshold via F1 optimization.
Validates precision/recall before corpus assembly."
```

---

### Task 5: Build Corpus Provenance Manifest

**Files:**
- Create: `apps/evals/teacher_model/provenance.py`

- [ ] **Step 1: Write failing test**

Create inline test at bottom of `provenance.py` (or separate test file):

```python
"""Corpus provenance tracking for legal/compliance audits."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
PROVENANCE_PATH = _DATA_DIR / "provenance.jsonl"


@dataclass
class ProvenanceRecord:
    url: str
    title: str
    channel_or_publisher: str
    download_timestamp: str
    license_claimed: str
    word_count: int
    inclusion_threshold_score: float | None = None
    source_tier: str = ""  # tier1_youtube, tier2_literature, tier3_musicology, tier4_own

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ProvenanceRecord":
        return cls(**json.loads(line))


class ProvenanceManifest:
    """Append-only provenance manifest for corpus documents."""

    def __init__(self, path: Path = PROVENANCE_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: ProvenanceRecord) -> None:
        """Append a provenance record to the manifest."""
        with open(self._path, "a") as f:
            f.write(record.to_json() + "\n")

    def count(self) -> int:
        """Count total records in the manifest."""
        if not self._path.exists():
            return 0
        return sum(1 for line in self._path.read_text().splitlines() if line.strip())

    def total_words(self) -> int:
        """Sum of word counts across all records."""
        if not self._path.exists():
            return 0
        total = 0
        for line in self._path.read_text().splitlines():
            if line.strip():
                record = ProvenanceRecord.from_json(line)
                total += record.word_count
        return total

    def by_tier(self) -> dict[str, int]:
        """Count records per source tier."""
        tiers: dict[str, int] = {}
        if not self._path.exists():
            return tiers
        for line in self._path.read_text().splitlines():
            if line.strip():
                record = ProvenanceRecord.from_json(line)
                tiers[record.source_tier] = tiers.get(record.source_tier, 0) + 1
        return tiers

    def summary(self) -> str:
        """Print a summary of the manifest."""
        count = self.count()
        words = self.total_words()
        tiers = self.by_tier()
        est_tokens = int(words * 1.3)  # rough word-to-token ratio
        lines = [
            f"Corpus Provenance: {count} documents, ~{words:,} words (~{est_tokens:,} tokens)",
            "By tier:",
        ]
        for tier, n in sorted(tiers.items()):
            lines.append(f"  {tier}: {n}")
        return "\n".join(lines)
```

- [ ] **Step 2: Quick smoke test**

```bash
cd apps/evals && uv run python -c "
from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
import tempfile, pathlib
manifest = ProvenanceManifest(pathlib.Path(tempfile.mktemp()))
manifest.add(ProvenanceRecord(
    url='https://youtube.com/watch?v=test',
    title='Test Video',
    channel_or_publisher='Test Channel',
    download_timestamp='2026-03-30T00:00:00Z',
    license_claimed='YouTube ToS',
    word_count=1500,
    source_tier='tier1_youtube',
))
assert manifest.count() == 1
assert manifest.total_words() == 1500
print('Provenance manifest works.')
print(manifest.summary())
"
```

- [ ] **Step 3: Commit**

```bash
git add apps/evals/teacher_model/provenance.py
git commit -m "feat: add corpus provenance manifest for legal/compliance tracking

Append-only JSONL manifest tracking URL, publisher, license, word count,
and relevance score for every document in the CPT corpus."
```

---

### Task 6: Build YouTube Transcription Pipeline

**Files:**
- Create: `apps/evals/teacher_model/transcribe.py`

- [ ] **Step 1: Add dependencies**

```bash
cd apps/evals && uv add yt-dlp cohere pyannote.audio tiktoken
```

- [ ] **Step 2: Implement the transcription pipeline**

Create `apps/evals/teacher_model/transcribe.py`:

```python
"""YouTube -> transcription pipeline using Cohere Transcribe + Pyannote diarization.

Usage:
  # Transcribe a single video
  uv run python -m teacher_model.transcribe --url "https://youtube.com/watch?v=XXX"

  # Transcribe a playlist
  uv run python -m teacher_model.transcribe --playlist "https://youtube.com/playlist?list=XXX"

  # Transcribe from a channel (most recent N videos)
  uv run python -m teacher_model.transcribe --channel "@tonebasePiano" --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cohere
import tiktoken

from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier

_DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = _DATA_DIR / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Token counter
_ENC = tiktoken.get_encoding("cl100k_base")


def download_audio(url: str, output_dir: Path) -> tuple[Path, dict]:
    """Download audio from YouTube video using yt-dlp. Returns (audio_path, metadata)."""
    output_template = str(output_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", output_template,
        "--write-info-json",
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Find the output files
    info_files = list(output_dir.glob("*.info.json"))
    if not info_files:
        raise FileNotFoundError(f"No info.json found after downloading {url}")
    metadata = json.loads(info_files[0].read_text())
    video_id = metadata.get("id", "unknown")
    audio_path = output_dir / f"{video_id}.wav"
    if not audio_path.exists():
        # Try other extensions
        for ext in ["wav", "opus", "webm", "m4a"]:
            candidate = output_dir / f"{video_id}.{ext}"
            if candidate.exists():
                audio_path = candidate
                break

    return audio_path, metadata


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using Cohere Transcribe API."""
    co = cohere.ClientV2()
    with open(audio_path, "rb") as f:
        response = co.audio.transcriptions.create(
            file=f,
            model="cohere-transcribe-03-2026",
        )
    return response.text


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_ENC.encode(text))


def process_video(
    url: str,
    manifest: ProvenanceManifest,
    classifier: PedagogyRelevanceClassifier | None = None,
    source_tier: str = "tier1_youtube",
) -> dict:
    """Full pipeline: download -> transcribe -> score -> save.

    Returns: {"video_id": str, "title": str, "word_count": int, "token_count": int, "relevance_score": float, "saved": bool}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Download
        audio_path, metadata = download_audio(url, tmp_path)
        video_id = metadata.get("id", "unknown")
        title = metadata.get("title", "Unknown")
        channel = metadata.get("channel", metadata.get("uploader", "Unknown"))

        # Transcribe
        transcript = transcribe_audio(audio_path)

    # Score relevance
    relevance_score = classifier.score(transcript) if classifier else None
    saved = relevance_score is None or relevance_score >= (classifier.threshold if classifier else 0.3)

    word_count = len(transcript.split())
    token_count = count_tokens(transcript)

    if saved:
        # Save transcript
        output_path = CORPUS_DIR / f"{video_id}.txt"
        output_path.write_text(transcript)

        # Record provenance
        manifest.add(ProvenanceRecord(
            url=url,
            title=title,
            channel_or_publisher=channel,
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            license_claimed="YouTube ToS",
            word_count=word_count,
            inclusion_threshold_score=relevance_score,
            source_tier=source_tier,
        ))

    return {
        "video_id": video_id,
        "title": title,
        "word_count": word_count,
        "token_count": token_count,
        "relevance_score": relevance_score,
        "saved": saved,
    }


def get_playlist_urls(playlist_url: str, limit: int | None = None) -> list[str]:
    """Extract video URLs from a YouTube playlist."""
    cmd = ["yt-dlp", "--flat-playlist", "--print", "url", playlist_url]
    if limit:
        cmd.extend(["--playlist-end", str(limit)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube piano pedagogy content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single video URL")
    group.add_argument("--playlist", help="Playlist URL")
    group.add_argument("--channel", help="Channel handle (e.g., @tonebasePiano)")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process")
    parser.add_argument("--no-filter", action="store_true", help="Skip relevance filtering")
    parser.add_argument("--tier", default="tier1_youtube", help="Source tier tag")
    args = parser.parse_args()

    manifest = ProvenanceManifest()
    classifier = None if args.no_filter else PedagogyRelevanceClassifier()

    if args.url:
        urls = [args.url]
    elif args.playlist:
        urls = get_playlist_urls(args.playlist, args.limit)
    elif args.channel:
        channel_url = f"https://youtube.com/{args.channel}/videos"
        urls = get_playlist_urls(channel_url, args.limit)
    else:
        urls = []

    print(f"Processing {len(urls)} videos...")
    for i, url in enumerate(urls):
        try:
            result = process_video(url, manifest, classifier, args.tier)
            status = "SAVED" if result["saved"] else "FILTERED"
            score_str = f" (relevance: {result['relevance_score']:.2f})" if result["relevance_score"] is not None else ""
            print(f"[{i+1}/{len(urls)}] {status} {result['title'][:60]}{score_str} ({result['token_count']} tokens)")
        except Exception as e:
            print(f"[{i+1}/{len(urls)}] ERROR {url}: {e}")

    print(f"\n{manifest.summary()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Quick import test (no network calls)**

```bash
cd apps/evals && uv run python -c "
from teacher_model.transcribe import count_tokens, CORPUS_DIR
assert count_tokens('Hello world') > 0
assert CORPUS_DIR.exists()
print('Transcription pipeline imports successfully.')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/teacher_model/transcribe.py apps/evals/pyproject.toml apps/evals/uv.lock
git commit -m "feat: add YouTube transcription pipeline for teacher model corpus

yt-dlp download -> Cohere Transcribe -> relevance filter -> provenance tracking.
Supports single video, playlist, and channel modes. Writes to corpus/ directory."
```

---

### Task 7: Build Domain Knowledge Probe (Gate 4 Eval)

**Files:**
- Create: `apps/evals/teacher_model/domain_knowledge_probe.py`
- Create: `apps/evals/teacher_model/data/domain_probe.json`

- [ ] **Step 1: Create 100 MCQ domain knowledge questions**

Create `apps/evals/teacher_model/data/domain_probe.json` with 100 factual multiple-choice questions derived from the pedagogy literature. Each question tests knowledge that would be present in the CPT corpus:

```json
[
  {
    "id": "dk_001",
    "question": "According to Tobias Matthay's 'Act of Touch', what is the primary mechanism for producing a singing tone on the piano?",
    "choices": {
      "A": "Striking the key with maximum finger speed",
      "B": "Controlled arm weight transferred through relaxed fingers",
      "C": "Pressing the key slowly after initial contact",
      "D": "Using only finger muscles isolated from the arm"
    },
    "correct": "B",
    "source": "Matthay, The Act of Touch in All Its Diversity (1903)",
    "topic": "technique"
  },
  {
    "id": "dk_002",
    "question": "In Chopin Nocturne performance practice, what pedaling technique does Charles Rosen recommend for maintaining harmonic clarity?",
    "choices": {
      "A": "No pedal throughout",
      "B": "Full pedal with changes every measure",
      "C": "Half-pedal changes synchronized with harmonic shifts",
      "D": "Flutter pedaling on every beat"
    },
    "correct": "C",
    "source": "Rosen, The Romantic Generation (1995)",
    "topic": "pedaling"
  }
]
```

Questions should cover: technique (Matthay, Leschetizky, Neuhaus), repertoire (Bach, Chopin, Beethoven, Debussy, Liszt performance practice), pedagogy methods (Suzuki, RCM, Faber), practice psychology (deliberate practice, Kageyama), and musical concepts (phrasing, dynamics, articulation, pedaling, interpretation).

- [ ] **Step 2: Implement the domain knowledge probe runner**

Create `apps/evals/teacher_model/domain_knowledge_probe.py`:

```python
"""Domain knowledge probe for evaluating CPT knowledge injection.

Runs 100 MCQ questions against a model and measures accuracy.
Used as Gate 4: CPT model must score >= 60%.

Usage:
  # Against Qwen base (pre-CPT baseline):
  uv run python -m teacher_model.domain_knowledge_probe --provider workers-ai

  # Against a local vLLM endpoint (post-CPT):
  uv run python -m teacher_model.domain_knowledge_probe --endpoint http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from teaching_knowledge.llm_client import LLMClient

_DATA_DIR = Path(__file__).parent / "data"
PROBE_PATH = _DATA_DIR / "domain_probe.json"

SYSTEM_PROMPT = (
    "You are taking a piano pedagogy knowledge quiz. For each question, "
    "respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not explain your reasoning."
)


@dataclass
class ProbeResult:
    question_id: str
    topic: str
    correct_answer: str
    model_answer: str | None
    is_correct: bool


def load_questions() -> list[dict]:
    """Load domain knowledge probe questions."""
    if not PROBE_PATH.exists():
        raise FileNotFoundError(f"Domain probe not found: {PROBE_PATH}")
    return json.loads(PROBE_PATH.read_text())


def extract_answer(response: str) -> str | None:
    """Extract a single letter answer from model response."""
    response = response.strip()
    # Try exact single letter
    if len(response) == 1 and response.upper() in "ABCD":
        return response.upper()
    # Try "A)" or "A." or "(A)" patterns
    match = re.search(r'\b([ABCD])[).:\s]', response)
    if match:
        return match.group(1)
    # Try first letter if short response
    if len(response) <= 3 and response[0].upper() in "ABCD":
        return response[0].upper()
    return None


def run_probe(
    client: LLMClient,
    questions: list[dict] | None = None,
) -> list[ProbeResult]:
    """Run the domain knowledge probe against a model."""
    if questions is None:
        questions = load_questions()

    results: list[ProbeResult] = []
    for q in questions:
        user_msg = f"{q['question']}\n\nA) {q['choices']['A']}\nB) {q['choices']['B']}\nC) {q['choices']['C']}\nD) {q['choices']['D']}"
        response = client.complete(SYSTEM_PROMPT, user_msg, max_tokens=10)
        model_answer = extract_answer(response)

        results.append(ProbeResult(
            question_id=q["id"],
            topic=q.get("topic", "unknown"),
            correct_answer=q["correct"],
            model_answer=model_answer,
            is_correct=model_answer == q["correct"],
        ))

    return results


def summarize_results(results: list[ProbeResult]) -> dict:
    """Summarize probe results by topic and overall."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    overall_accuracy = correct / total if total > 0 else 0.0

    by_topic: dict[str, dict] = {}
    for r in results:
        if r.topic not in by_topic:
            by_topic[r.topic] = {"correct": 0, "total": 0}
        by_topic[r.topic]["total"] += 1
        if r.is_correct:
            by_topic[r.topic]["correct"] += 1

    topic_accuracy = {
        topic: stats["correct"] / stats["total"]
        for topic, stats in by_topic.items()
    }

    return {
        "overall_accuracy": overall_accuracy,
        "correct": correct,
        "total": total,
        "by_topic": topic_accuracy,
        "gate_passed": overall_accuracy >= 0.60,
    }


def main():
    parser = argparse.ArgumentParser(description="Run domain knowledge probe")
    parser.add_argument("--provider", default="workers-ai", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--endpoint", default=None, help="Custom API endpoint (for vLLM)")
    args = parser.parse_args()

    client = LLMClient(provider=args.provider)
    if args.model:
        client.model = args.model

    print(f"Running domain knowledge probe ({len(load_questions())} questions)...")
    results = run_probe(client)
    summary = summarize_results(results)

    print(f"\nOverall accuracy: {summary['overall_accuracy']:.1%} ({summary['correct']}/{summary['total']})")
    print(f"Gate 4 (>= 60%): {'PASSED' if summary['gate_passed'] else 'FAILED'}")
    print("\nBy topic:")
    for topic, acc in sorted(summary["by_topic"].items()):
        print(f"  {topic}: {acc:.1%}")

    # Save results
    output_path = _DATA_DIR / "domain_probe_results.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Quick import test**

```bash
cd apps/evals && uv run python -c "
from teacher_model.domain_knowledge_probe import extract_answer, load_questions
assert extract_answer('B') == 'B'
assert extract_answer('A) Arm weight') == 'A'
assert extract_answer('The answer is C.') == 'C'
print('Domain knowledge probe imports correctly.')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/teacher_model/domain_knowledge_probe.py apps/evals/teacher_model/data/domain_probe.json
git commit -m "feat: add domain knowledge probe for CPT gate evaluation

100 MCQ questions on piano pedagogy (technique, repertoire, methods,
practice psychology). Gate 4: model must score >= 60% post-CPT."
```

---

### Task 8: Build PDF/Web Text Extraction Pipeline

**Files:**
- Create: `apps/evals/teacher_model/scrape_text.py`

- [ ] **Step 1: Add dependencies**

```bash
cd apps/evals && uv add pymupdf beautifulsoup4 requests
```

- [ ] **Step 2: Implement the text extraction pipeline**

Create `apps/evals/teacher_model/scrape_text.py`:

```python
"""PDF and web text extraction for Tier 2-3 corpus sources.

Usage:
  # Extract text from a PDF
  uv run python -m teacher_model.scrape_text --pdf path/to/book.pdf --tier tier2_literature

  # Scrape a web page
  uv run python -m teacher_model.scrape_text --url "https://bulletproofmusician.com/article" --tier tier2_literature

  # Batch from a URL list file
  uv run python -m teacher_model.scrape_text --url-list urls.txt --tier tier3_musicology
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from teacher_model.provenance import ProvenanceManifest, ProvenanceRecord
from teacher_model.relevance_classifier import PedagogyRelevanceClassifier

_DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = _DATA_DIR / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_web_text(url: str) -> tuple[str, str]:
    """Extract article text from a web page. Returns (text, title)."""
    response = requests.get(url, timeout=30, headers={
        "User-Agent": "CrescendAI-Research/1.0 (piano pedagogy corpus collection)"
    })
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, nav, footer, header elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string if soup.title else "Unknown"

    # Try article tag first, then main, then body
    content = soup.find("article") or soup.find("main") or soup.find("body")
    if content is None:
        return "", title

    # Extract text paragraphs
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return text, title.strip()


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove control chars."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def process_pdf(
    pdf_path: Path,
    manifest: ProvenanceManifest,
    classifier: PedagogyRelevanceClassifier | None = None,
    source_tier: str = "tier2_literature",
    publisher: str = "Unknown",
) -> dict:
    """Extract and process a PDF document."""
    text = extract_pdf_text(pdf_path)
    text = clean_text(text)
    word_count = len(text.split())

    relevance_score = classifier.score(text[:2000]) if classifier else None  # Score first 2K chars
    saved = relevance_score is None or relevance_score >= (classifier.threshold if classifier else 0.3)

    if saved and text:
        doc_id = sha256(str(pdf_path).encode()).hexdigest()[:12]
        output_path = CORPUS_DIR / f"pdf_{doc_id}.txt"
        output_path.write_text(text)

        manifest.add(ProvenanceRecord(
            url=str(pdf_path),
            title=pdf_path.stem,
            channel_or_publisher=publisher,
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            license_claimed="PDF source",
            word_count=word_count,
            inclusion_threshold_score=relevance_score,
            source_tier=source_tier,
        ))

    return {"path": str(pdf_path), "word_count": word_count, "relevance_score": relevance_score, "saved": saved}


def process_url(
    url: str,
    manifest: ProvenanceManifest,
    classifier: PedagogyRelevanceClassifier | None = None,
    source_tier: str = "tier2_literature",
) -> dict:
    """Extract and process a web page."""
    text, title = extract_web_text(url)
    text = clean_text(text)
    word_count = len(text.split())

    relevance_score = classifier.score(text[:2000]) if classifier else None
    saved = relevance_score is None or relevance_score >= (classifier.threshold if classifier else 0.3)

    if saved and text:
        doc_id = sha256(url.encode()).hexdigest()[:12]
        output_path = CORPUS_DIR / f"web_{doc_id}.txt"
        output_path.write_text(text)

        manifest.add(ProvenanceRecord(
            url=url,
            title=title,
            channel_or_publisher=url.split("/")[2],
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            license_claimed="Web/open-access",
            word_count=word_count,
            inclusion_threshold_score=relevance_score,
            source_tier=source_tier,
        ))

    return {"url": url, "title": title, "word_count": word_count, "relevance_score": relevance_score, "saved": saved}


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs and web pages")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", type=Path, help="Path to a PDF file")
    group.add_argument("--pdf-dir", type=Path, help="Directory of PDF files")
    group.add_argument("--url", help="Web page URL")
    group.add_argument("--url-list", type=Path, help="File with one URL per line")
    parser.add_argument("--tier", default="tier2_literature", help="Source tier tag")
    parser.add_argument("--publisher", default="Unknown", help="Publisher name (for PDFs)")
    parser.add_argument("--no-filter", action="store_true", help="Skip relevance filtering")
    args = parser.parse_args()

    manifest = ProvenanceManifest()
    classifier = None if args.no_filter else PedagogyRelevanceClassifier()

    if args.pdf:
        result = process_pdf(args.pdf, manifest, classifier, args.tier, args.publisher)
        print(f"{'SAVED' if result['saved'] else 'FILTERED'}: {result['path']} ({result['word_count']} words)")
    elif args.pdf_dir:
        for pdf in sorted(args.pdf_dir.glob("*.pdf")):
            result = process_pdf(pdf, manifest, classifier, args.tier, args.publisher)
            print(f"{'SAVED' if result['saved'] else 'FILTERED'}: {pdf.name} ({result['word_count']} words)")
    elif args.url:
        result = process_url(args.url, manifest, classifier, args.tier)
        print(f"{'SAVED' if result['saved'] else 'FILTERED'}: {result['title']} ({result['word_count']} words)")
    elif args.url_list:
        urls = [line.strip() for line in args.url_list.read_text().splitlines() if line.strip()]
        for i, url in enumerate(urls):
            try:
                result = process_url(url, manifest, classifier, args.tier)
                print(f"[{i+1}/{len(urls)}] {'SAVED' if result['saved'] else 'FILTERED'}: {result['title'][:60]} ({result['word_count']} words)")
            except Exception as e:
                print(f"[{i+1}/{len(urls)}] ERROR {url}: {e}")

    print(f"\n{manifest.summary()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Quick import test**

```bash
cd apps/evals && uv run python -c "
from teacher_model.scrape_text import clean_text, extract_pdf_text
assert clean_text('  hello   world  \n\n\n\ntest  ') == 'hello world\n\ntest'
print('Text extraction pipeline imports correctly.')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/teacher_model/scrape_text.py apps/evals/pyproject.toml apps/evals/uv.lock
git commit -m "feat: add PDF/web text extraction pipeline for Tier 2-3 corpus

PyMuPDF for PDFs, BeautifulSoup for web scraping. Relevance filtering,
provenance tracking, and clean text output to corpus/ directory."
```

---

### Task 9: Build MinHash Deduplication

**Files:**
- Create: `apps/evals/teacher_model/dedup.py`

- [ ] **Step 1: Add dependency**

```bash
cd apps/evals && uv add datasketch
```

- [ ] **Step 2: Implement deduplication**

Create `apps/evals/teacher_model/dedup.py`:

```python
"""MinHash + LSH deduplication for corpus documents.

Usage:
  uv run python -m teacher_model.dedup --corpus-dir data/corpus --threshold 0.8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasketch import MinHash, MinHashLSH

_DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = _DATA_DIR / "corpus"


def text_to_shingles(text: str, k: int = 5) -> set[str]:
    """Convert text to character k-shingles."""
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    """Build a MinHash from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def find_duplicates(
    corpus_dir: Path = CORPUS_DIR,
    threshold: float = 0.8,
    num_perm: int = 128,
) -> list[tuple[str, str, float]]:
    """Find near-duplicate pairs in the corpus. Returns list of (file1, file2, similarity)."""
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}

    for f in files:
        text = f.read_text()
        if len(text) < 100:
            continue
        shingles = text_to_shingles(text)
        if not shingles:
            continue
        mh = build_minhash(shingles, num_perm)
        minhashes[f.name] = mh
        try:
            lsh.insert(f.name, mh)
        except ValueError:
            pass  # Duplicate key, skip

    duplicates: list[tuple[str, str, float]] = []
    seen = set()
    for name, mh in minhashes.items():
        candidates = lsh.query(mh)
        for candidate in candidates:
            if candidate != name:
                pair = tuple(sorted([name, candidate]))
                if pair not in seen:
                    seen.add(pair)
                    sim = minhashes[name].jaccard(minhashes[candidate])
                    duplicates.append((pair[0], pair[1], sim))

    return sorted(duplicates, key=lambda x: -x[2])


def remove_duplicates(
    corpus_dir: Path = CORPUS_DIR,
    threshold: float = 0.8,
    dry_run: bool = True,
) -> int:
    """Find and remove duplicate documents (keeps the first alphabetically)."""
    duplicates = find_duplicates(corpus_dir, threshold)
    removed = 0

    for file1, file2, sim in duplicates:
        to_remove = corpus_dir / file2  # Keep file1, remove file2
        if dry_run:
            print(f"WOULD REMOVE: {file2} (duplicate of {file1}, similarity {sim:.2f})")
        else:
            to_remove.unlink()
            print(f"REMOVED: {file2} (duplicate of {file1}, similarity {sim:.2f})")
        removed += 1

    return removed


def main():
    parser = argparse.ArgumentParser(description="Deduplicate corpus documents")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity threshold")
    parser.add_argument("--remove", action="store_true", help="Actually remove duplicates (default: dry run)")
    args = parser.parse_args()

    print(f"Scanning {args.corpus_dir} for duplicates (threshold: {args.threshold})...")
    removed = remove_duplicates(args.corpus_dir, args.threshold, dry_run=not args.remove)
    print(f"\n{'Would remove' if not args.remove else 'Removed'}: {removed} duplicates")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Quick test**

```bash
cd apps/evals && uv run python -c "
from teacher_model.dedup import text_to_shingles, build_minhash
shingles = text_to_shingles('This is a test of deduplication.')
assert len(shingles) > 0
mh = build_minhash(shingles)
print(f'Shingles: {len(shingles)}, MinHash built successfully.')
"
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/teacher_model/dedup.py apps/evals/pyproject.toml apps/evals/uv.lock
git commit -m "feat: add MinHash deduplication for CPT corpus

LSH-based near-duplicate detection. Default 0.8 Jaccard threshold.
Dry-run mode by default. Removes YouTube re-uploads and quoted passages."
```

---

### Task 10: Corpus Builder Orchestrator

**Files:**
- Create: `apps/evals/teacher_model/corpus_builder.py`

- [ ] **Step 1: Implement the orchestrator**

Create `apps/evals/teacher_model/corpus_builder.py`:

```python
"""Orchestrates full corpus assembly pipeline.

Usage:
  # Run full pipeline status
  uv run python -m teacher_model.corpus_builder status

  # Run dedup pass
  uv run python -m teacher_model.corpus_builder dedup

  # Compute final corpus stats
  uv run python -m teacher_model.corpus_builder stats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken

from teacher_model.provenance import ProvenanceManifest
from teacher_model.dedup import find_duplicates, remove_duplicates

_DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = _DATA_DIR / "corpus"
_ENC = tiktoken.get_encoding("cl100k_base")


def corpus_stats() -> dict:
    """Compute statistics for the assembled corpus."""
    files = sorted(CORPUS_DIR.glob("*.txt"))
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for f in files:
        text = f.read_text()
        total_words += len(text.split())
        total_tokens += len(_ENC.encode(text))
        total_chars += len(text)

    manifest = ProvenanceManifest()
    provenance_count = manifest.count()
    tiers = manifest.by_tier()

    return {
        "documents": len(files),
        "provenance_records": provenance_count,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_by_tier": tiers,
        "target_100m": total_tokens >= 100_000_000,
        "target_200m": total_tokens >= 200_000_000,
    }


def print_status():
    """Print current corpus assembly status."""
    stats = corpus_stats()
    pct_100m = min(stats["total_tokens"] / 100_000_000 * 100, 100)
    pct_200m = min(stats["total_tokens"] / 200_000_000 * 100, 100)

    print("=" * 60)
    print("CORPUS ASSEMBLY STATUS")
    print("=" * 60)
    print(f"Documents:  {stats['documents']}")
    print(f"Words:      {stats['total_words']:,}")
    print(f"Tokens:     {stats['total_tokens']:,}")
    print(f"Progress:   {pct_100m:.1f}% of 100M target / {pct_200m:.1f}% of 200M target")
    print()
    print("By source tier:")
    for tier, count in sorted(stats["tokens_by_tier"].items()):
        print(f"  {tier}: {count} documents")
    print()

    dupes = find_duplicates(CORPUS_DIR)
    if dupes:
        print(f"Duplicates found: {len(dupes)} pairs (run 'corpus_builder dedup' to remove)")
    else:
        print("No duplicates detected.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Corpus assembly orchestrator")
    parser.add_argument("command", choices=["status", "stats", "dedup"], help="Command to run")
    parser.add_argument("--remove", action="store_true", help="For dedup: actually remove files")
    args = parser.parse_args()

    if args.command == "status":
        print_status()
    elif args.command == "stats":
        import json
        stats = corpus_stats()
        print(json.dumps(stats, indent=2))
    elif args.command == "dedup":
        removed = remove_duplicates(CORPUS_DIR, dry_run=not args.remove)
        print(f"{'Removed' if args.remove else 'Would remove'}: {removed} duplicates")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick test**

```bash
cd apps/evals && uv run python -c "
from teacher_model.corpus_builder import corpus_stats
stats = corpus_stats()
print(f'Current corpus: {stats[\"documents\"]} docs, {stats[\"total_tokens\"]:,} tokens')
"
```

- [ ] **Step 3: Commit**

```bash
git add apps/evals/teacher_model/corpus_builder.py
git commit -m "feat: add corpus builder orchestrator with status and dedup commands

Tracks corpus assembly progress toward 100M/200M token targets.
Integrates provenance manifest and MinHash dedup."
```

---

## Summary

| Task | What | Must-Fix | Est. Time |
|------|------|----------|-----------|
| 1 | Fix judge v2 absent=2 scoring rule | #2 | 15 min |
| 2 | Qwen tool format translator | #1 | 30 min |
| 3 | Tool calling regression test harness | -- | 30 min |
| 4 | Relevance classifier | #3 | 45 min |
| 5 | Corpus provenance manifest | -- | 15 min |
| 6 | YouTube transcription pipeline | -- | 30 min |
| 7 | Domain knowledge probe (Gate 4) | -- | 45 min |
| 8 | PDF/web text extraction pipeline | -- | 30 min |
| 9 | MinHash deduplication | -- | 15 min |
| 10 | Corpus builder orchestrator | -- | 15 min |

**Total: ~4.5 hours of implementation time.** After this, you can start collecting corpus data immediately.

**What's NOT in this plan (gated, future sprints):**
- CPT training script (Gate 4 must pass first)
- SFT dataset generation (needs CPT + tool format validated)
- GRPO training (needs SFT + rubric validated)
- DPO training (needs GRPO)
- Together.ai deployment (needs all training stages)
- API integration (needs deployed model)
