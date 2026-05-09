"""Pipeline B: run base teacher LLM on tool-probe cases, score each response."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from teacher_model.stage0.cases import load_cases
from teacher_model.stage0.tool_scorer import ToolCase as ScorerCase, score_response


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


@dataclass
class RunStats:
    processed: int
    errors: int
    skipped: int


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)  # JSONDecodeError propagates; corrupt file is fatal
        rid = row.get("case_id")
        if rid:
            done.add(rid)
    return done


def run(
    cases_path: Path,
    system_prompt_path: Path,
    schemas: dict,
    teacher_client: _Client,
    out_path: Path,
    max_tokens: int = 800,
) -> RunStats:
    cases = load_cases(cases_path)
    system_prompt = system_prompt_path.read_text()
    completed = _load_completed(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    skipped = 0

    with out_path.open("a") as fout:
        for c in cases:
            if c.case_id in completed:
                skipped += 1
                continue
            user = (
                f"<case_id>{c.case_id}</case_id>\n"
                f"Briefing:\n{json.dumps(c.briefing, indent=2)}"
            )
            row: dict = {
                "case_id": c.case_id,
                "expected_call": c.expected_call,
                "expected_tool": c.expected_tool,
                "category": c.category,
                "called": False,
                "tool_name": None,
                "arguments": None,
                "discipline_correct": False,
                "format_valid": None,
                "extraction_format": None,
                "raw_response": "",
                "model_id": getattr(teacher_client, "model", ""),
                "latency_ms": 0,
                "error": "",
            }
            try:
                t0 = time.monotonic()
                raw = teacher_client.complete(user=user, system=system_prompt, max_tokens=max_tokens)
                row["latency_ms"] = round((time.monotonic() - t0) * 1000)
                row["raw_response"] = raw
                result = score_response(
                    raw,
                    ScorerCase(
                        case_id=c.case_id,
                        expected_call=c.expected_call,
                        expected_tool=c.expected_tool,
                        category=c.category,
                    ),
                    schemas,
                )
                row.update({
                    "called": result.called,
                    "tool_name": result.tool_name,
                    "arguments": result.arguments,
                    "discipline_correct": result.discipline_correct,
                    "format_valid": result.format_valid,
                    "extraction_format": result.extraction_format,
                })
                processed += 1
            except Exception as exc:
                row["error"] = str(exc)[:500]
                errors += 1
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    return RunStats(processed=processed, errors=errors, skipped=skipped)
