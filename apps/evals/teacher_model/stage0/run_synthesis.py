"""Pipeline A: run base teacher LLM on the n=100 holdout, judge each response.

Resumable: rows already present in out_path (by recording_id) are skipped on
subsequent invocations -- including errored rows, to prevent duplicate appends.
To retry errored rows, the caller must explicitly remove them from the file.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from teaching_knowledge.run_eval import (
    aggregate_muq,
    build_synthesis_user_msg,
    extract_teacher_response,
)


class _Client(Protocol):
    model: str

    def complete(self, user: str, system: str, max_tokens: int) -> str: ...


JudgeFn = Callable[..., Any]  # (synthesis_text, context, **kwargs) -> JudgeResultV2Extended


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
        rid = row.get("recording_id")
        if rid:
            done.add(rid)
    return done


def run(
    holdout_path: Path,
    out_path: Path,
    teacher_client: _Client,
    judge_fn: JudgeFn,
    *,
    system_prompt: str,
    judge_provider: str = "workers-ai",
    judge_model: str | None = "@cf/google/gemma-4-26b-a4b-it",
    max_tokens: int = 1024,
    limit: int | None = None,
) -> RunStats:
    holdout = [
        json.loads(line) for line in holdout_path.read_text().splitlines() if line.strip()
    ]
    if limit is not None:
        holdout = holdout[:limit]
    completed = _load_completed(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    skipped = 0

    with out_path.open("a") as fout:
        for entry in holdout:
            rid = entry["recording_id"]
            if rid in completed:
                skipped += 1
                continue

            briefing = json.loads(Path(entry["briefing_path"]).read_text())
            chunks = briefing.get("chunks", [])
            muq_means = aggregate_muq(chunks) if chunks else {}
            duration = float(briefing.get("total_duration_seconds", 0.0))
            meta = {
                "title": entry.get("title", ""),
                "composer": entry.get("composer", ""),
                "skill_bucket": int(entry.get("skill_bucket", 3)),
            }

            row: dict = {
                "recording_id": rid,
                "model_id": getattr(teacher_client, "model", ""),
                "stratum": entry.get("stratum"),
                "era": entry.get("era"),
                "skill_bucket": entry.get("skill_bucket"),
                "synthesis_text": "",
                "judge_dimensions": [],
                "judge_model": "",
                "synthesis_latency_ms": 0,
                "judge_latency_ms": 0,
                "routed_provider": "",
                "error": "",
            }

            try:
                user_msg = build_synthesis_user_msg(muq_means, duration, meta)
                t0 = time.monotonic()
                raw = teacher_client.complete(user=user_msg, system=system_prompt, max_tokens=max_tokens)
                row["synthesis_latency_ms"] = round((time.monotonic() - t0) * 1000)
                row["synthesis_text"] = extract_teacher_response(raw)
                row["routed_provider"] = getattr(teacher_client, "last_routed_provider", "") or ""

                judge_ctx = {
                    "piece_name": meta["title"],
                    "composer": meta["composer"],
                    "skill_level": meta["skill_bucket"],
                }
                jres = judge_fn(
                    row["synthesis_text"],
                    judge_ctx,
                    provider=judge_provider,
                    model=judge_model,
                )
                row["judge_dimensions"] = [
                    {
                        "criterion": d.criterion,
                        "process": getattr(d, "process", None),
                        "outcome": getattr(d, "outcome", None),
                        "score": getattr(d, "score", None),
                        "evidence": getattr(d, "evidence", ""),
                        "reason": getattr(d, "reason", ""),
                    }
                    for d in jres.dimensions
                ]
                row["judge_model"] = getattr(jres, "model", "")
                row["judge_latency_ms"] = round(getattr(jres, "latency_ms", 0.0))
                processed += 1
            except Exception as exc:
                row["error"] = str(exc)[:500]
                errors += 1

            fout.write(json.dumps(row) + "\n")
            fout.flush()

    return RunStats(processed=processed, errors=errors, skipped=skipped)
