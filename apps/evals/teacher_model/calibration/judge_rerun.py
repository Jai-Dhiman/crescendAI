"""Re-run the v2 judge on a fixed set of anchor syntheses at a labeled time point.

Used to compute judge-vs-judge kappa (day1 vs day30) for the protocol's drift gate.
The judge_callable parameter is injectable so tests can stub the network call;
in production, callers pass shared.judge.judge_synthesis_v2 (wrapped) here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _build_baseline_index(baseline_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with baseline_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            synth_id = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[synth_id] = row
    return index


def rerun_anchors(
    anchor_synth_ids: list[str],
    baseline_path: Path,
    output_path: Path,
    run_label: str,
    judge_callable: Callable[[str, dict], dict] | None = None,
) -> None:
    if judge_callable is None:
        raise ValueError(
            "judge_callable must be provided. In production pass a wrapper around "
            "shared.judge.judge_synthesis_v2 that returns a dict; tests pass a stub."
        )

    index = _build_baseline_index(baseline_path)

    with output_path.open("w") as out:
        for synth_id in anchor_synth_ids:
            if synth_id not in index:
                raise KeyError(f"synth_id not found in baseline: {synth_id}")
            row = index[synth_id]
            context = {
                "piece_name": row.get("title", "Unknown"),
                "composer": row.get("composer", "Unknown"),
                "skill_level": row.get("skill_bucket", "Unknown"),
            }
            result = judge_callable(row["synthesis_text"], context)
            record = {
                "synth_id": synth_id,
                "run_label": run_label,
                "dimensions": result["dimensions"],
                "model": result.get("model", ""),
                "prompt_version": result.get("prompt_version", ""),
                "latency_ms": result.get("latency_ms", 0.0),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record) + "\n")
