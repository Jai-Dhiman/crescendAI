"""Score post-tool-result assistant continuations against degeneracy categories.

Categories (in priority order):
  empty           -- response shorter than _MIN_CHARS or under _MIN_TOKENS_APPROX words
  refusal         -- explicit refusal phrases ("cannot continue", "unable to help")
  repetition      -- response re-emits the same tool_call as the initial turn
  format_collapse -- response is raw JSON / not natural prose
  clean           -- none of the above
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FIXTURES_PATH = Path(__file__).parent / "data" / "continuation_fixtures.json"

_MIN_CHARS = 30
_MIN_TOKENS_APPROX = 10  # whitespace-split word count

_REFUSAL_PATTERNS = (
    re.compile(r"\bI (cannot|can't|am unable to|won't) (continue|help|proceed|assist)", re.IGNORECASE),
    re.compile(r"\bI (cannot|can't) (provide|do)\b", re.IGNORECASE),
    re.compile(r"\bI'm sorry,? (but )?I (cannot|can't)", re.IGNORECASE),
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RAW_JSON_RE = re.compile(r"^\s*[\{\[]")


Category = Literal["clean", "empty", "refusal", "repetition", "format_collapse"]


@dataclass
class ContinuationResult:
    category: Category
    is_degenerate: bool
    detail: str


def _extract_tool_call_payload(text: str) -> str | None:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()
    return payload


def _is_raw_json_dump(text: str) -> bool:
    s = text.strip()
    if not _RAW_JSON_RE.match(s):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def score_continuation(
    initial_assistant: str,
    tool_result: dict,
    follow_up_response: str,
) -> ContinuationResult:
    """Classify a follow-up assistant response after a tool_result."""
    text = follow_up_response or ""
    stripped = text.strip()

    # 1. repetition: same tool_call payload re-emitted (checked before empty so a
    #    bare tool_call tag isn't misclassified as empty)
    initial_payload = _extract_tool_call_payload(initial_assistant)
    follow_up_payload = _extract_tool_call_payload(stripped)
    if initial_payload and follow_up_payload and initial_payload == follow_up_payload:
        return ContinuationResult(
            category="repetition",
            is_degenerate=True,
            detail="follow-up re-emits identical tool_call",
        )

    # 2. format collapse: raw JSON dump (checked before empty so compact JSON
    #    isn't misclassified as empty due to low word count)
    if _is_raw_json_dump(stripped):
        return ContinuationResult(
            category="format_collapse",
            is_degenerate=True,
            detail="response is parseable JSON, not prose",
        )

    # 3. empty / truncated
    if len(stripped) < _MIN_CHARS or len(stripped.split()) < _MIN_TOKENS_APPROX:
        return ContinuationResult(
            category="empty",
            is_degenerate=True,
            detail=f"len={len(stripped)} chars, words={len(stripped.split())}",
        )

    # 4. refusal
    for pat in _REFUSAL_PATTERNS:
        if pat.search(stripped):
            return ContinuationResult(
                category="refusal",
                is_degenerate=True,
                detail=f"matched: {pat.pattern}",
            )

    return ContinuationResult(category="clean", is_degenerate=False, detail="")


def load_tool_result_fixture(tool_name: str) -> dict:
    """Load the canned tool_result payload for a given tool name."""
    fixtures = json.loads(_FIXTURES_PATH.read_text())
    if tool_name not in fixtures:
        raise KeyError(f"No continuation fixture for tool {tool_name!r}")
    return fixtures[tool_name]
