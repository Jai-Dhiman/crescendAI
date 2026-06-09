# apps/evals/teaching_knowledge/ablation/run_ablation.py
"""4-condition signal ablation orchestrator (V6 DO path, #28).

Replays each recording's chunks through the real SessionBrain V6 DO under four
conditions -- real / shuffle / marginal / flip -- corrupting the per-chunk MuQ
predictions the DO ingests (not a Python-framed prompt). If the V6 output is
invariant to corrupted signals, the teacher is hallucinating rather than reading
the data; analyze.py turns the real-vs-corrupted judge deltas into a verdict.

The DO driver is injected so the orchestration is unit-testable without a live
wrangler dev; the live verdict needs Anthropic credits.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from shared.provenance import make_run_provenance
from teaching_knowledge.ablation.corrupt_signals import corrupt_chunks
from teaching_knowledge.run_eval import build_do_row

CONDITIONS = ("real", "shuffle", "marginal", "flip")


def _default_driver(wrangler_url, recording_cache, student_id, piece_query):
    """Default DO-replay driver: real run_recording over wrangler dev."""
    from shared.pipeline_client import run_recording

    return asyncio.run(
        run_recording(
            wrangler_url,
            recording_cache,
            student_id=student_id,
            piece_query=piece_query,
        )
    )


def _load_completed(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done.add((row["recording_id"], row["condition"]))
    return done


def run_ablation(
    *,
    recordings: list[dict[str, Any]],
    out_path: Path,
    wrangler_url: str = "http://localhost:8787",
    driver=_default_driver,
    judge_fn=None,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
    seed: int = 42,
    skip_judge: bool = False,
) -> None:
    """Drive each recording through the DO under all four conditions.

    `recordings` items: {recording_id, chunks: [{chunk_index, predictions, ...}], meta}.
    `driver(wrangler_url, recording_cache, student_id, piece_query) -> SessionResult`
    is injected (default: shared.pipeline_client.run_recording).
    """
    if not skip_judge and judge_fn is None:
        raise ValueError("judge_fn is required unless skip_judge=True")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(out_path)
    provenance = make_run_provenance()
    all_chunks = [rec["chunks"] for rec in recordings]

    with out_path.open("a") as fout:
        for idx, rec in enumerate(recordings):
            real_chunks = rec["chunks"]
            for condition in CONDITIONS:
                key = (rec["recording_id"], condition)
                if key in completed:
                    continue

                if condition == "real":
                    used_chunks = real_chunks
                else:
                    used_chunks = corrupt_chunks(
                        real_chunks,
                        mode=condition,
                        seed=seed + idx,
                        all_chunks=all_chunks,
                    )

                recording_cache = {
                    "recording_id": rec["recording_id"],
                    "chunks": used_chunks,
                }
                # Fresh per-condition eval identity -> cold-start fidelity and no
                # cross-condition longitudinal bleed in sessionHistory/pastDiagnoses.
                student_id = f"eval-ablation-{rec['recording_id']}-{condition}"
                meta = rec.get("meta", {})

                session_result = driver(
                    wrangler_url,
                    recording_cache,
                    student_id,
                    meta.get("piece_slug") or None,
                )

                # Reuse the baseline row shaping: judges the FULL V6 artifact (#28),
                # not the headline alone. skip_judge -> dry_run (judge not called).
                row = build_do_row(
                    session_result,
                    meta,
                    judge_fn,
                    provenance,
                    dry_run=skip_judge,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
                row["condition"] = condition
                row["judge_skipped"] = skip_judge

                fout.write(json.dumps(row) + "\n")
                fout.flush()
