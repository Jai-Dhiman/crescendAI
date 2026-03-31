"""Tests for the Qwen/OpenAI tool format translator."""

import json
import pytest

from teacher_model.tool_format import (
    EXERCISE_TOOL_ANTHROPIC,
    EXERCISE_TOOL_OPENAI,
    anthropic_tool_call_to_qwen_chatml,
    anthropic_tool_to_openai,
    openai_tool_call_to_anthropic,
    validate_exercise_tool_call,
    VALID_FOCUS_DIMENSIONS,
    VALID_HANDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_tool_call(
    focus_dimension: str = "dynamics",
    hands: str = "both",
) -> dict:
    """Return a minimal valid Anthropic tool_use block."""
    return {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "create_exercise",
        "input": {
            "source_passage": "measures 1-4",
            "target_skill": "Even tone across registers",
            "exercises": [
                {
                    "title": "Slow practice",
                    "instruction": "Play measures 1-4 at half tempo. Focus on evenness.",
                    "focus_dimension": focus_dimension,
                    "hands": hands,
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# 1. Schema translation
# ---------------------------------------------------------------------------

def test_exercise_tool_schema_translation():
    """Anthropic schema translates to a valid OpenAI function schema."""
    result = anthropic_tool_to_openai(EXERCISE_TOOL_ANTHROPIC)

    assert result["type"] == "function"
    fn = result["function"]
    assert fn["name"] == "create_exercise"
    assert fn["description"] == EXERCISE_TOOL_ANTHROPIC["description"]

    # Anthropic uses "input_schema"; OpenAI uses "parameters"
    assert "parameters" in fn
    assert "input_schema" not in fn
    assert fn["parameters"] == EXERCISE_TOOL_ANTHROPIC["input_schema"]

    # Required top-level fields are preserved inside parameters
    params = fn["parameters"]
    assert set(params["required"]) == {"source_passage", "target_skill", "exercises"}

    # Exercise item schema is intact
    exercise_props = params["properties"]["exercises"]["items"]["properties"]
    assert "focus_dimension" in exercise_props
    assert exercise_props["focus_dimension"]["enum"] == [
        "dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"
    ]


def test_exercise_tool_openai_constant_matches_function():
    """Pre-computed EXERCISE_TOOL_OPENAI must equal anthropic_tool_to_openai(EXERCISE_TOOL_ANTHROPIC)."""
    assert EXERCISE_TOOL_OPENAI == anthropic_tool_to_openai(EXERCISE_TOOL_ANTHROPIC)


# ---------------------------------------------------------------------------
# 2. Tool call -> Qwen ChatML serialization
# ---------------------------------------------------------------------------

def test_tool_call_to_qwen_chatml():
    """Tool call serializes correctly as JSON with name + arguments keys."""
    tc = _minimal_tool_call()
    serialized = anthropic_tool_call_to_qwen_chatml(tc)

    parsed = json.loads(serialized)
    assert parsed["name"] == "create_exercise"
    assert parsed["arguments"] == tc["input"]
    assert "type" not in parsed
    assert "id" not in parsed


def test_tool_call_to_qwen_chatml_is_valid_json():
    """The serialized output must be valid JSON."""
    tc = _minimal_tool_call()
    serialized = anthropic_tool_call_to_qwen_chatml(tc)
    # json.loads raises if invalid
    json.loads(serialized)


# ---------------------------------------------------------------------------
# 3. Round-trip: Anthropic -> Qwen -> Anthropic
# ---------------------------------------------------------------------------

def test_round_trip_tool_call():
    """Anthropic -> Qwen serialization -> Anthropic preserves name and input data."""
    original = _minimal_tool_call()

    # Step 1: Anthropic -> Qwen ChatML string
    serialized = anthropic_tool_call_to_qwen_chatml(original)

    # Step 2: Parse the string (as Qwen would produce it)
    parsed = json.loads(serialized)

    # Step 3: Convert back to Anthropic format
    recovered = openai_tool_call_to_anthropic(parsed)

    assert recovered["type"] == "tool_use"
    assert recovered["name"] == original["name"]
    assert recovered["input"] == original["input"]


def test_round_trip_preserves_nested_exercise_data():
    """Nested exercise array is fully preserved through the round-trip."""
    original = _minimal_tool_call()
    original["input"]["exercises"].append({
        "title": "Hands separate",
        "instruction": "Isolate the left hand. Play slowly.",
        "focus_dimension": "timing",
        "hands": "left",
    })

    serialized = anthropic_tool_call_to_qwen_chatml(original)
    parsed = json.loads(serialized)
    recovered = openai_tool_call_to_anthropic(parsed)

    assert len(recovered["input"]["exercises"]) == 2
    assert recovered["input"]["exercises"][1]["hands"] == "left"


# ---------------------------------------------------------------------------
# 4. focus_dimension validation
# ---------------------------------------------------------------------------

def test_focus_dimension_validation_all_valid():
    """All 6 valid focus_dimensions are accepted without errors."""
    for dim in VALID_FOCUS_DIMENSIONS:
        tc = _minimal_tool_call(focus_dimension=dim)
        errors = validate_exercise_tool_call(tc)
        assert errors == [], f"Expected no errors for focus_dimension={dim!r}, got: {errors}"


def test_validate_catches_invalid_dimension():
    """Invalid focus_dimension produces an error."""
    tc = _minimal_tool_call(focus_dimension="intonation")
    errors = validate_exercise_tool_call(tc)
    assert len(errors) == 1
    assert "focus_dimension" in errors[0]
    assert "intonation" in errors[0]


def test_validate_catches_empty_string_dimension():
    """Empty string is not a valid focus_dimension."""
    tc = _minimal_tool_call(focus_dimension="")
    errors = validate_exercise_tool_call(tc)
    assert any("focus_dimension" in e for e in errors)


# ---------------------------------------------------------------------------
# 5. hands enum validation
# ---------------------------------------------------------------------------

def test_hands_enum_validation_all_valid():
    """Only left/right/both are accepted for hands."""
    for hands_val in VALID_HANDS:
        tc = _minimal_tool_call(hands=hands_val)
        errors = validate_exercise_tool_call(tc)
        assert errors == [], f"Expected no errors for hands={hands_val!r}, got: {errors}"


def test_validate_catches_invalid_hands():
    """Invalid hands value produces an error."""
    tc = _minimal_tool_call(hands="together")
    errors = validate_exercise_tool_call(tc)
    assert len(errors) == 1
    assert "hands" in errors[0]
    assert "together" in errors[0]


# ---------------------------------------------------------------------------
# 6. Missing required fields
# ---------------------------------------------------------------------------

def test_validate_catches_missing_top_level_fields():
    """Missing source_passage, target_skill, or exercises each produce an error."""
    base = _minimal_tool_call()["input"]

    for field in ("source_passage", "target_skill", "exercises"):
        payload = {k: v for k, v in base.items() if k != field}
        tc = {"type": "tool_use", "name": "create_exercise", "input": payload}
        errors = validate_exercise_tool_call(tc)
        assert any(field in e for e in errors), (
            f"Expected error for missing '{field}', got: {errors}"
        )


def test_validate_catches_missing_exercise_fields():
    """Missing required fields inside an exercise item produce per-item errors."""
    tc = _minimal_tool_call()
    # Remove 'instruction' from the first exercise
    del tc["input"]["exercises"][0]["instruction"]
    errors = validate_exercise_tool_call(tc)
    assert any("instruction" in e for e in errors)


def test_validate_catches_missing_focus_dimension_in_exercise():
    """Missing focus_dimension inside exercise produces error."""
    tc = _minimal_tool_call()
    del tc["input"]["exercises"][0]["focus_dimension"]
    errors = validate_exercise_tool_call(tc)
    assert any("focus_dimension" in e for e in errors)


def test_validate_catches_missing_hands_in_exercise():
    """Missing hands inside exercise produces error."""
    tc = _minimal_tool_call()
    del tc["input"]["exercises"][0]["hands"]
    errors = validate_exercise_tool_call(tc)
    assert any("hands" in e for e in errors)


# ---------------------------------------------------------------------------
# 7. arguments-style input (Qwen/SFT dict)
# ---------------------------------------------------------------------------

def test_validate_accepts_arguments_key():
    """validate_exercise_tool_call also accepts 'arguments' key (Qwen/SFT style)."""
    serialized = anthropic_tool_call_to_qwen_chatml(_minimal_tool_call())
    parsed = json.loads(serialized)
    # parsed has {"name": ..., "arguments": ...}
    errors = validate_exercise_tool_call(parsed)
    assert errors == []


def test_validate_rejects_missing_payload_key():
    """A dict with neither 'input' nor 'arguments' key returns an error."""
    errors = validate_exercise_tool_call({"name": "create_exercise"})
    assert len(errors) == 1
    assert "input" in errors[0] or "arguments" in errors[0]


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

def test_validate_empty_exercises_list():
    """An empty exercises list produces an error."""
    tc = _minimal_tool_call()
    tc["input"]["exercises"] = []
    errors = validate_exercise_tool_call(tc)
    assert any("empty" in e for e in errors)


def test_validate_multiple_exercises_all_valid():
    """Multiple valid exercises produce no errors."""
    tc = _minimal_tool_call()
    tc["input"]["exercises"].append({
        "title": "Pedal control",
        "instruction": "Apply half pedal throughout. Listen for blur.",
        "focus_dimension": "pedaling",
        "hands": "right",
    })
    errors = validate_exercise_tool_call(tc)
    assert errors == []


def test_validate_multiple_exercises_second_invalid():
    """An invalid second exercise is reported with the correct index."""
    tc = _minimal_tool_call()
    tc["input"]["exercises"].append({
        "title": "Bad exercise",
        "instruction": "Do something.",
        "focus_dimension": "intonation",  # invalid
        "hands": "both",
    })
    errors = validate_exercise_tool_call(tc)
    assert any("exercises[1]" in e and "focus_dimension" in e for e in errors)
