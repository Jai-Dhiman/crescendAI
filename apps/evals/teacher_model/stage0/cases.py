"""Tool-probe case loader.

The case file is one JSON object per line:
{
  "case_id": "p_search_01",
  "expected_call": true,
  "expected_tool": "search_catalog",
  "category": null,            // for positives
  "briefing": { ... full briefing object the model is shown ... }
}

For negatives, `expected_call` is false, `expected_tool` is null, and
`category` is one of: chitchat / premature / ambiguous / already_recommended
/ out_of_scope / borderline_wrong_tool.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolCase:
    case_id: str
    expected_call: bool
    expected_tool: str | None
    category: str | None
    briefing: dict


def load_cases(path: Path) -> list[ToolCase]:
    """Read tool_probe_cases.jsonl into ToolCase records."""
    out: list[ToolCase] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(
            ToolCase(
                case_id=row["case_id"],
                expected_call=bool(row["expected_call"]),
                expected_tool=row.get("expected_tool"),
                category=row.get("category"),
                briefing=row["briefing"],
            )
        )
    return out
