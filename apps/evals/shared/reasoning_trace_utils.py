"""Utilities for decoding reasoning_trace values from the observations table.

reasoning_trace is a polymorphic column:
  - Prose string (legacy): used verbatim in LLM prompts.
  - Discriminated union JSON: written by persistAccumulatedMoments when
    bar-analysis facts (llmAnalysis) are present.  Shape:
      { "kind": "facts", "tier": int, "bar_range": str|null,
        "selected": {"dimension": str, "analysis": str},
        "correlated": [{"dimension": str, "analysis": str}] }

Use decode_reasoning_trace() before inserting a reasoning_trace value into any
LLM prompt so that structured facts are presented as readable prose rather than
raw JSON.
"""

from __future__ import annotations

import json


def _facts_to_prose(facts: dict) -> str:
    """Render a BarAnalysisFacts dict to a short, readable prose summary."""
    selected = facts.get("selected", {})
    dimension = selected.get("dimension", "unknown")
    analysis = selected.get("analysis", "")
    bar_range = facts.get("bar_range")
    tier = facts.get("tier")
    correlated: list[dict] = facts.get("correlated", [])

    parts: list[str] = []

    location = f"bars {bar_range}" if bar_range else "this passage"
    parts.append(f"Selected dimension '{dimension}' at {location}: {analysis}.")

    if correlated:
        corr_summaries = [
            f"{c.get('dimension', '?')} ({c.get('analysis', '')})"
            for c in correlated
        ]
        parts.append(f"Also notable: {'; '.join(corr_summaries)}.")

    if tier is not None:
        parts.append(f"(Analysis tier {tier}.)")

    return " ".join(parts)


def decode_reasoning_trace(trace: str) -> str:
    """Return a prompt-safe string for the given reasoning_trace value.

    - If trace is empty or not valid JSON, return as-is (prose path).
    - If trace parses as JSON with kind == "facts", render to prose.
    - If trace parses as JSON without kind (legacy raw JSON), return as-is.
    """
    if not trace:
        return trace

    try:
        parsed = json.loads(trace)
    except (json.JSONDecodeError, TypeError):
        return trace

    if not isinstance(parsed, dict):
        return trace

    if parsed.get("kind") == "facts":
        return _facts_to_prose(parsed)

    # Legacy or unrecognized JSON — return verbatim so we don't silently mangle it.
    return trace
