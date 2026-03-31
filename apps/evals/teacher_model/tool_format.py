"""
Translates the CrescendAI create_exercise tool between Anthropic and Qwen/OpenAI formats.

SFT training data must use Qwen's native tool format, but the current production system
uses Anthropic's format. This module bridges the two.

Anthropic format:
  {"name": "...", "description": "...", "input_schema": {"type": "object", "properties": {...}}}

Qwen/OpenAI format:
  {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

For SFT training data, tool calls in the assistant message are serialized as JSON:
  {"name": "create_exercise", "arguments": {...}}
"""

import json
from typing import Any


EXERCISE_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": "create_exercise",
    "description": (
        "Create a focused practice exercise when the student would benefit from structured drill. "
        "Use sparingly -- most observations should be text-only."
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
                        "title": {"type": "string"},
                        "instruction": {
                            "type": "string",
                            "description": "Concrete steps. 2-4 sentences.",
                        },
                        "focus_dimension": {
                            "type": "string",
                            "enum": [
                                "dynamics",
                                "timing",
                                "pedaling",
                                "articulation",
                                "phrasing",
                                "interpretation",
                            ],
                        },
                        "hands": {
                            "type": "string",
                            "enum": ["left", "right", "both"],
                        },
                    },
                },
            },
        },
    },
}

VALID_FOCUS_DIMENSIONS = frozenset(
    ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
)
VALID_HANDS = frozenset(["left", "right", "both"])


def anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool definition to OpenAI function format.

    Anthropic uses `input_schema` as the key for the JSON Schema; OpenAI uses `parameters`.
    The outer wrapper changes from a flat dict to {"type": "function", "function": {...}}.
    """
    function_def: dict[str, Any] = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
    }
    return {"type": "function", "function": function_def}


def anthropic_tool_call_to_qwen_chatml(tool_call: dict[str, Any]) -> str:
    """Serialize an Anthropic tool call as JSON for Qwen ChatML.

    Anthropic tool_use content blocks have the shape:
      {"type": "tool_use", "id": "...", "name": "...", "input": {...}}

    Qwen's ChatML expects the assistant's tool call to be a JSON string with:
      {"name": "...", "arguments": {...}}
    """
    payload = {
        "name": tool_call["name"],
        "arguments": tool_call["input"],
    }
    return json.dumps(payload, ensure_ascii=False)


def openai_tool_call_to_anthropic(parsed: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed Qwen/OpenAI tool call back to Anthropic tool_use format.

    `parsed` is the dict decoded from the JSON string produced by
    `anthropic_tool_call_to_qwen_chatml`, i.e. {"name": "...", "arguments": {...}}.

    Returns an Anthropic tool_use content block (without `id`, which the caller
    must supply when constructing a real API request).
    """
    return {
        "type": "tool_use",
        "name": parsed["name"],
        "input": parsed["arguments"],
    }


def validate_exercise_tool_call(tool_call: dict[str, Any]) -> list[str]:
    """Validate a create_exercise tool call.

    Accepts either an Anthropic tool_use block (with `input` key) or a
    Qwen/SFT-style dict (with `arguments` key). Returns a list of error
    strings. An empty list means the call is valid.
    """
    errors: list[str] = []

    # Resolve the payload regardless of wrapper format.
    if "input" in tool_call:
        payload = tool_call["input"]
    elif "arguments" in tool_call:
        payload = tool_call["arguments"]
    else:
        errors.append("tool_call must contain 'input' or 'arguments' key")
        return errors

    if not isinstance(payload, dict):
        errors.append("tool call payload must be a dict")
        return errors

    # Top-level required fields.
    for field in ("source_passage", "target_skill", "exercises"):
        if field not in payload:
            errors.append(f"missing required field: {field}")

    if "exercises" not in payload:
        return errors

    exercises = payload["exercises"]
    if not isinstance(exercises, list):
        errors.append("exercises must be a list")
        return errors

    if len(exercises) == 0:
        errors.append("exercises list must not be empty")

    for i, exercise in enumerate(exercises):
        if not isinstance(exercise, dict):
            errors.append(f"exercises[{i}] must be a dict")
            continue

        for field in ("title", "instruction", "focus_dimension", "hands"):
            if field not in exercise:
                errors.append(f"exercises[{i}] missing required field: {field}")

        if "focus_dimension" in exercise:
            dim = exercise["focus_dimension"]
            if dim not in VALID_FOCUS_DIMENSIONS:
                errors.append(
                    f"exercises[{i}] invalid focus_dimension '{dim}'; "
                    f"must be one of {sorted(VALID_FOCUS_DIMENSIONS)}"
                )

        if "hands" in exercise:
            hands = exercise["hands"]
            if hands not in VALID_HANDS:
                errors.append(
                    f"exercises[{i}] invalid hands '{hands}'; "
                    f"must be one of {sorted(VALID_HANDS)}"
                )

    return errors


# Pre-computed OpenAI format constant derived from the Anthropic constant above.
EXERCISE_TOOL_OPENAI: dict[str, Any] = anthropic_tool_to_openai(EXERCISE_TOOL_ANTHROPIC)
