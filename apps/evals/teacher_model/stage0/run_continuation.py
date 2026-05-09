"""Pipeline B+: replay positive successful tool calls with a synthetic tool_result.

For each row in tool_runs.jsonl where expected_call is True AND called is True
AND discipline_correct is True, build a follow-up turn where:
  1. The original assistant turn (with tool_call) is included.
  2. A synthetic tool_result message is appended (from continuation_fixtures.json).
  3. The model is asked to continue.

The follow-up response is classified by score_continuation; one row per replay
is written to out_path with {case_id, category, is_degenerate, detail}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from teacher_model.stage0.cases import load_cases
from teacher_model.stage0.continuation_probe import (
    load_tool_result_fixture,
    score_continuation,
)


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


@dataclass
class RunStats:
    processed: int
    errors: int


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(
    tool_runs_path: Path,
    cases_path: Path,
    teacher_client: _Client,
    out_path: Path,
    max_tokens: int = 600,
) -> RunStats:
    tool_runs = _read_jsonl(tool_runs_path)
    cases = {c.case_id: c for c in load_cases(cases_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    with out_path.open("w") as fout:
        for tr in tool_runs:
            if not (tr.get("expected_call") and tr.get("called") and tr.get("discipline_correct")):
                continue
            case = cases.get(tr["case_id"])
            if case is None:
                continue
            tool_name = tr.get("tool_name") or ""
            try:
                fixture = load_tool_result_fixture(tool_name)
            except KeyError:
                continue

            initial_assistant = tr.get("raw_response", "")
            user = (
                f"<case_id>{case.case_id}</case_id>\n"
                f"Original briefing:\n{json.dumps(case.briefing, indent=2)}\n\n"
                f"You called: {initial_assistant}\n\n"
                f"<tool_result tool=\"{tool_name}\">{json.dumps(fixture)}</tool_result>\n\n"
                f"Continue with your teacher response, integrating the tool result for the student."
            )
            row: dict = {
                "case_id": case.case_id,
                "tool_name": tool_name,
                "category": "",
                "is_degenerate": False,
                "detail": "",
                "follow_up_response": "",
                "model_id": getattr(teacher_client, "model", ""),
                "error": "",
            }
            try:
                follow_up = teacher_client.complete(user=user, system="", max_tokens=max_tokens)
                row["follow_up_response"] = follow_up
                result = score_continuation(initial_assistant, fixture, follow_up)
                row["category"] = result.category
                row["is_degenerate"] = result.is_degenerate
                row["detail"] = result.detail
                processed += 1
            except Exception as exc:
                row["error"] = str(exc)[:500]
                errors += 1
            fout.write(json.dumps(row) + "\n")

    return RunStats(processed=processed, errors=errors)
