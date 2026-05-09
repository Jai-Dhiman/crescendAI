"""Extract and score tool calls from arbitrary-format base-model output.

Tolerated input shapes (in priority order):
  1. Qwen-native:   <tool_call>{...}</tool_call>
  2. Fenced JSON:   ```json\n{...}\n```  (or any fenced block)
  3. Raw JSON:      {"name": "...", "arguments": {...}}

Scoring is split:
  - discipline_correct: did the model make the right call/no-call decision,
    AND if it called, was the tool name correct?
  - format_valid:       given that it called, did `arguments` validate against
    the tool schema? None when no call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from jsonschema import Draft202012Validator, ValidationError

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


@dataclass
class ToolCase:
    case_id: str
    expected_call: bool
    expected_tool: str | None  # None for negative cases
    category: str | None       # for negative cases: chitchat/premature/etc.


@dataclass
class ToolProbeResult:
    case_id: str
    called: bool
    tool_name: str | None
    arguments: dict | None
    discipline_correct: bool
    format_valid: bool | None
    extraction_format: str | None  # "qwen_tag" | "fenced" | "raw_json" | None
    error: str


def _try_parse_call(payload: str) -> tuple[str | None, dict | None]:
    """Parse a JSON blob expected to contain {name, arguments}."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str):
        return None, None
    if args is not None and not isinstance(args, dict):
        return None, None
    return name, args or {}


def _extract_call(raw: str) -> tuple[str | None, dict | None, str | None]:
    """Return (tool_name, arguments, extraction_format) or (None, None, None)."""
    # 1. Qwen-native tag
    m = _TOOL_CALL_TAG_RE.search(raw)
    if m:
        name, args = _try_parse_call(m.group(1).strip())
        if name is not None:
            return name, args, "qwen_tag"

    # 2. Fenced JSON block
    for m in _FENCE_RE.finditer(raw):
        name, args = _try_parse_call(m.group(1).strip())
        if name is not None:
            return name, args, "fenced"

    # 3. Raw JSON object anywhere in the response
    for m in _JSON_OBJECT_RE.finditer(raw):
        name, args = _try_parse_call(m.group(0))
        if name is not None:
            return name, args, "raw_json"

    return None, None, None


def _validate_args(tool_name: str, args: dict, schemas: dict) -> tuple[bool, str]:
    schema = schemas.get(tool_name)
    if schema is None:
        return False, f"unknown tool: {tool_name}"
    try:
        Draft202012Validator(schema).validate(args)
        return True, ""
    except ValidationError as e:
        return False, e.message


def score_response(
    raw: str,
    expected: ToolCase,
    schemas: dict,
) -> ToolProbeResult:
    name, args, fmt = _extract_call(raw or "")
    called = name is not None

    if not called:
        # No call made.
        discipline = expected.expected_call is False
        return ToolProbeResult(
            case_id=expected.case_id,
            called=False,
            tool_name=None,
            arguments=None,
            discipline_correct=discipline,
            format_valid=None,
            extraction_format=None,
            error="",
        )

    # Call made; check tool name discipline + schema.
    if expected.expected_call is False:
        # Negative case but model called: undisciplined regardless of which tool.
        format_valid, err = _validate_args(name, args or {}, schemas) if name in schemas else (False, "unknown tool")
        return ToolProbeResult(
            case_id=expected.case_id,
            called=True,
            tool_name=name,
            arguments=args,
            discipline_correct=False,
            format_valid=format_valid,
            extraction_format=fmt,
            error=err,
        )

    # Positive case: tool name must match expected_tool.
    tool_name_correct = (name == expected.expected_tool)
    if not tool_name_correct:
        return ToolProbeResult(
            case_id=expected.case_id,
            called=True,
            tool_name=name,
            arguments=args,
            discipline_correct=False,
            format_valid=False,
            extraction_format=fmt,
            error=f"expected {expected.expected_tool!r}, got {name!r}",
        )

    format_valid, err = _validate_args(name, args or {}, schemas)
    return ToolProbeResult(
        case_id=expected.case_id,
        called=True,
        tool_name=name,
        arguments=args,
        discipline_correct=True,
        format_valid=format_valid,
        extraction_format=fmt,
        error=err,
    )
