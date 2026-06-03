"""
Orchestrator: inference cache -> Sonnet synthesis -> Gemma judge -> JSONL results.

Usage:
    cd apps/evals
    uv run python -m teaching_knowledge.run_eval
    uv run python -m teaching_knowledge.run_eval --limit 50
    uv run python -m teaching_knowledge.run_eval --out results/my_run.jsonl
    uv run python -m teaching_knowledge.run_eval --dry-run   # synthesis only, skip judge

Models:
    Synthesis: claude-sonnet-4-6 (Anthropic, prod model)
    Judge:     @cf/google/gemma-4-26b-a4b-it (Workers AI)

Resume-safe: any recording_id already written to the output JSONL is skipped.
Errors per recording are written to the JSONL and the run continues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import yaml

from shared.provenance import RunProvenance, make_run_provenance
from shared.judge_compatibility import assert_judge_compatible
from shared.judge_atomic import judge_atomic_matrix

# Root paths
EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parents[1]
CACHE_DIR = REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
SKILL_EVAL_DIR = REPO_ROOT / "model" / "data" / "evals" / "skill_eval"
RESULTS_DIR = EVALS_ROOT / "results"
SYNTHESIS_SYSTEM_PATH = REPO_ROOT / "apps" / "shared" / "teacher-style" / "synthesis_system.txt"

# MuQ global mean from Rust STOP classifier (synthesis.ts SCALER_MEAN)
SCALER_MEAN: dict[str, float] = {
    "dynamics": 0.545,
    "timing": 0.4848,
    "pedaling": 0.4594,
    "articulation": 0.5369,
    "phrasing": 0.5188,
    "interpretation": 0.5064,
}
DIMS = list(SCALER_MEAN.keys())

# Loaded from apps/shared/teacher-style/synthesis_system.txt (keep in sync with prompts.ts SESSION_SYNTHESIS_SYSTEM)
SESSION_SYNTHESIS_SYSTEM = SYNTHESIS_SYSTEM_PATH.read_text().strip()


def _assert_models_compatible(teacher_model: str, judge_model: str) -> None:
    assert_judge_compatible(teacher_model, judge_model)


def _judge_provider_for(judge_model: str) -> str:
    """Autodetect the judge provider from the model name.

    @cf/* => workers-ai; vendor/model => openrouter (bare names unreachable
    here -- the family guard blocks them upstream).
    """
    return "openrouter" if "/" in judge_model and not judge_model.startswith("@cf/") else "workers-ai"


def _maybe_atomic_judge(
    *, synthesis_text: str, context: dict, judge_dimensions: list[dict],
    threshold: float, client,
) -> dict | None:
    scores = [
        float(d["score"]) for d in judge_dimensions
        if isinstance(d.get("score"), (int, float))
    ]
    if not scores:
        return None
    mean_score = sum(scores) / len(scores)
    if mean_score >= threshold:
        return None
    result = judge_atomic_matrix(synthesis_text=synthesis_text, context=context, client=client)
    return {
        "moves": [
            {"move_id": m.move_id, "attempted": m.attempted, "criteria": m.criteria}
            for m in result.moves
        ],
        "threshold": threshold,
    }


def load_manifests() -> dict[str, dict[str, Any]]:
    """Build a video_id -> metadata lookup from all skill_eval manifests."""
    lookup: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(SKILL_EVAL_DIR.glob("*/manifest.yaml")):
        data = yaml.safe_load(manifest_path.read_text())
        piece_slug = data.get("piece", manifest_path.parent.name)
        title = data.get("title", piece_slug)
        composer = data.get("composer", "Unknown")
        for rec in data.get("recordings", []):
            video_id = rec.get("video_id")
            if video_id:
                lookup[video_id] = {
                    "piece_slug": piece_slug,
                    "title": title,
                    "composer": composer,
                    "skill_bucket": rec.get("skill_bucket", 3),
                }
    return lookup


def aggregate_muq(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Compute per-dimension mean MuQ scores across all chunks."""
    accum: dict[str, list[float]] = {dim: [] for dim in DIMS}
    for chunk in chunks:
        preds = chunk.get("predictions", {})
        for dim in DIMS:
            if dim in preds and preds[dim] is not None:
                accum[dim].append(float(preds[dim]))
    return {
        dim: round(sum(vals) / len(vals), 4)
        for dim, vals in accum.items()
        if vals
    }


def load_completed_ids(out_path: Path) -> set[str]:
    """Return recording_ids already written to the output JSONL."""
    if not out_path.exists():
        return set()
    completed: set[str] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            rid = row.get("recording_id")
            if rid and not row.get("error") and row.get("synthesis_text"):
                completed.add(rid)
        except json.JSONDecodeError:
            pass
    return completed


def _filter_cache_files_by_split(
    cache_files: list[Path],
    split_path: Path | None,
    which: str,
) -> list[Path]:
    """Filter cache files by split membership.

    When split_path is None, returns the full list unchanged.
    When which == "all", returns only files whose stem is in (train + holdout).
    """
    if split_path is None:
        return cache_files
    from teaching_knowledge.scripts.split import load_split

    allowed = load_split(split_path, which=which)
    return [f for f in cache_files if f.stem in allowed]


def _build_row(
    recording_id: str,
    meta: dict,
    muq_means: dict[str, float],
    synthesis_text: str,
    synthesis_latency_ms: int,
    judge_dimensions: list[dict],
    judge_model: str,
    judge_latency_ms: int,
    error: str,
    provenance: RunProvenance,
) -> dict:
    """Build a single output-JSONL row with provenance stamped in."""
    return {
        "recording_id": recording_id,
        "run_id": provenance.run_id,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "piece_slug": meta["piece_slug"],
        "title": meta["title"],
        "composer": meta["composer"],
        "skill_bucket": meta["skill_bucket"],
        "muq_means": muq_means,
        "synthesis_text": synthesis_text,
        "synthesis_latency_ms": synthesis_latency_ms,
        "judge_dimensions": judge_dimensions,
        "judge_model": judge_model,
        "judge_latency_ms": judge_latency_ms,
        "error": error,
    }


def build_do_row(
    session_result,
    meta: dict[str, Any],
    judge_fn,
    provenance: RunProvenance,
    *,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
) -> dict:
    """Build one aggregator-schema JSONL row from a DO SessionResult.

    No thin-framing fallback: a DO/WS failure (or missing synthesis) records the
    error verbatim and skips the judge. Piece resolution is reported via
    `piece_resolved` (False -> the DO ran Tier 2/3 with bar_range null).
    """
    recording_id = getattr(session_result, "recording_id", meta.get("recording_id", ""))
    piece_resolved = getattr(session_result, "piece_identification", None) is not None

    base = {
        "recording_id": recording_id,
        "run_id": provenance.run_id,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "piece_slug": meta.get("piece_slug", ""),
        "title": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_bucket": meta.get("skill_bucket", 3),
        "piece_resolved": piece_resolved,
        "synthesis_text": "",
        "synthesis_latency_ms": int(getattr(session_result, "synthesis_latency_ms", 0) or 0),
        "judge_dimensions": [],
        "judge_model": "",
        "judge_latency_ms": 0,
        "error": "",
    }

    synthesis = getattr(session_result, "synthesis", None)
    errors = getattr(session_result, "errors", []) or []
    if synthesis is None or not getattr(synthesis, "text", ""):
        base["error"] = "; ".join(errors) if errors else "DO returned no synthesis text"
        return base

    base["synthesis_text"] = synthesis.text

    if dry_run:
        base["judge_model"] = "dry_run"
        return base

    judge_ctx = {
        "piece_name": meta.get("title", ""),
        "composer": meta.get("composer", ""),
        "skill_level": meta.get("skill_bucket", 3),
    }
    jr = judge_fn(
        synthesis.text,
        judge_ctx,
        provider=judge_provider,
        model=judge_model,
    )
    base["judge_dimensions"] = [
        {
            "criterion": d.criterion,
            "process": getattr(d, "process", None),
            "outcome": getattr(d, "outcome", None),
            "score": getattr(d, "score", None),
            "evidence": getattr(d, "evidence", ""),
            "reason": getattr(d, "reason", ""),
        }
        for d in jr.dimensions
    ]
    base["judge_model"] = getattr(jr, "model", "")
    base["judge_latency_ms"] = round(getattr(jr, "latency_ms", 0.0))
    return base


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


def run_do_baseline(
    holdout_path: Path,
    out_path: Path,
    wrangler_url: str,
    judge_fn,
    *,
    driver=_default_driver,
    limit: int | None = None,
    dry_run: bool = False,
    judge_provider: str = "workers-ai",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
    student_id: str = "eval-student-001",
) -> None:
    """Drive holdout recordings through the real SessionBrain DO and write JSONL.

    `driver(wrangler_url, recording_cache, student_id, piece_query) -> SessionResult`
    is injected so the orchestration is unit-testable without a live DO.
    The default driver calls shared.pipeline_client.run_recording.
    """
    holdout = [
        json.loads(line)
        for line in holdout_path.read_text().splitlines()
        if line.strip()
    ]
    if limit is not None:
        holdout = holdout[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(out_path)
    provenance = make_run_provenance()
    print(f"run_id: {provenance.run_id} | DO path | wrangler={wrangler_url}")

    processed = 0
    errors = 0

    with out_path.open("a") as fout:
        for entry in holdout:
            recording_id = entry["recording_id"]
            if recording_id in completed:
                continue

            meta = {
                "recording_id": recording_id,
                "piece_slug": entry.get("piece_slug", ""),
                "title": entry.get("title", ""),
                "composer": entry.get("composer", ""),
                "skill_bucket": int(entry.get("skill_bucket", 3)),
            }

            briefing = json.loads(Path(entry["briefing_path"]).read_text())
            recording_cache = {
                "recording_id": briefing.get("recording_id", recording_id),
                "chunks": briefing.get("chunks", []),
            }

            # Each recording gets a fresh eval identity so sessionHistory /
            # pastDiagnoses queries in the DO see an empty longitudinal record.
            # The outer `student_id` param is ignored here intentionally.
            per_recording_student_id = f"eval-{recording_id}"

            try:
                session_result = driver(
                    wrangler_url,
                    recording_cache,
                    per_recording_student_id,
                    meta["piece_slug"] or None,
                )
                row = build_do_row(
                    session_result,
                    meta,
                    judge_fn,
                    provenance,
                    dry_run=dry_run,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            except Exception as exc:  # driver/judge hard failure -> error row, never thin fallback
                row = {
                    "recording_id": recording_id,
                    "run_id": provenance.run_id,
                    "git_sha": provenance.git_sha,
                    "git_dirty": provenance.git_dirty,
                    "piece_slug": meta["piece_slug"],
                    "title": meta["title"],
                    "composer": meta["composer"],
                    "skill_bucket": meta["skill_bucket"],
                    "piece_resolved": False,
                    "synthesis_text": "",
                    "synthesis_latency_ms": 0,
                    "judge_dimensions": [],
                    "judge_model": "",
                    "judge_latency_ms": 0,
                    "error": str(exc)[:500],
                }

            if row.get("error"):
                errors += 1
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1
            print(f"[{processed}] {recording_id} | {meta['piece_slug']}"
                  + (f" | ERROR: {row['error'][:80]}" if row.get("error") else ""))

    print(f"Done. processed={processed} errors={errors} -> {out_path}")


def run(
    limit: int | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
    split: str = "all",
    split_path: Path | None = None,
    teacher_model: str = "claude-sonnet-4-6",
    judge_model: str = "@cf/google/gemma-4-26b-a4b-it",
    atomic_threshold: float | None = 2.0,
    system_prompt_path: Path | None = None,
) -> None:
    from teaching_knowledge.llm_client import LLMClient
    from shared.judge import judge_synthesis_v2

    if not dry_run:
        _assert_models_compatible(teacher_model, judge_model)

    system_prompt = (
        system_prompt_path.read_text().strip()
        if system_prompt_path is not None
        else SESSION_SYNTHESIS_SYSTEM
    )

    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "baseline_v1.jsonl"

    print(f"Output: {out_path}")
    print(f"Cache:  {CACHE_DIR}")

    manifest_lookup = load_manifests()
    print(f"Manifest entries loaded: {len(manifest_lookup)}")

    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Cache directory not found: {CACHE_DIR}")

    cache_files = [
        f for f in sorted(CACHE_DIR.glob("*.json"))
        if f.name != "_fingerprint.json"
    ]
    print(f"Cache files: {len(cache_files)}")
    cache_files = _filter_cache_files_by_split(cache_files, split_path, which=split)
    print(f"After split filter ({split}): {len(cache_files)}")

    completed_ids = load_completed_ids(out_path)
    if completed_ids:
        print(f"Resuming -- {len(completed_ids)} already completed")

    synthesis_client = LLMClient(provider="anthropic", model=teacher_model)
    print(f"Synthesis: {synthesis_client.model}")
    if not dry_run:
        print(f"Judge:     {judge_model}")

    provenance = make_run_provenance()
    print(f"run_id: {provenance.run_id}")
    print(f"git_sha: {provenance.git_sha}{' (dirty)' if provenance.git_dirty else ''}")

    processed = 0
    skipped_no_manifest = 0
    errors = 0

    with out_path.open("a") as fout:
        for cache_path in cache_files:
            if limit is not None and processed >= limit:
                break

            data = json.loads(cache_path.read_text())
            recording_id = data.get("recording_id", cache_path.stem)

            if recording_id in completed_ids:
                continue

            meta = manifest_lookup.get(recording_id)
            if meta is None:
                skipped_no_manifest += 1
                continue

            chunks = data.get("chunks", [])
            if not chunks:
                skipped_no_manifest += 1
                continue

            muq_means = aggregate_muq(chunks)
            duration_seconds = data.get("total_duration_seconds", 0.0)

            try:
                raise NotImplementedError(
                    "Thin-framing synthesis path removed (issue #22). "
                    "Use run_do_baseline via --do-path."
                )

            except Exception as exc:
                errors += 1
                result = _build_row(
                    recording_id=recording_id,
                    meta=meta,
                    muq_means=muq_means,
                    synthesis_text="",
                    synthesis_latency_ms=0,
                    judge_dimensions=[],
                    judge_model="",
                    judge_latency_ms=0,
                    error=str(exc),
                    provenance=provenance,
                )

            fout.write(json.dumps(result) + "\n")
            fout.flush()
            processed += 1

            # Progress line
            mean_score_str = ""
            if result.get("judge_dimensions"):
                scores = [
                    d["score"]
                    for d in result["judge_dimensions"]
                    if isinstance(d.get("score"), (int, float))
                ]
                if scores:
                    mean_score_str = f" | mean={round(sum(scores)/len(scores), 2)}"

            status = (
                f"[{processed}] {recording_id} | {meta['piece_slug']} "
                f"sk={meta['skill_bucket']}{mean_score_str}"
            )
            if result["error"]:
                status += f" | ERROR: {result['error'][:80]}"
            print(status)

    print(
        f"\nDone. processed={processed} errors={errors} "
        f"skipped_no_manifest={skipped_no_manifest}"
    )
    print(f"Results: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Teaching quality eval runner")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max recordings to process (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/baseline_v1.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthesis only, skip judge (faster for prompt tuning checks)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "holdout", "all"],
        default="all",
        help="Filter recordings by split membership (default: all)",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to splits.json (default: data/splits.json if present)",
    )
    parser.add_argument(
        "--teacher-model",
        default="claude-sonnet-4-6",
        help="Teacher model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--judge-model",
        default="@cf/google/gemma-4-26b-a4b-it",
        help="Judge model name (default: @cf/google/gemma-4-26b-a4b-it)",
    )
    parser.add_argument(
        "--atomic-threshold",
        type=float,
        default=2.0,
        help="Run atomic-matrix judge when mean judge score is below this threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="Path to a synthesis system prompt .txt file (default: apps/shared/teacher-style/synthesis_system.txt).",
    )
    parser.add_argument(
        "--do-path",
        action="store_true",
        help="Drive synthesis through the real SessionBrain DO (requires wrangler dev).",
    )
    parser.add_argument(
        "--wrangler-url",
        default="http://localhost:8787",
        help="wrangler dev base URL for --do-path (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.do_path:
        from shared.judge import judge_synthesis_v2

        judge_provider = _judge_provider_for(args.judge_model)
        holdout = EVALS_ROOT / "teacher_model" / "stage0" / "data" / "stage0_holdout.jsonl"
        out = args.out if args.out is not None else (RESULTS_DIR / "baseline_v2_do.jsonl")
        run_do_baseline(
            holdout_path=holdout,
            out_path=out,
            wrangler_url=args.wrangler_url,
            judge_fn=judge_synthesis_v2,
            limit=args.limit,
            dry_run=args.dry_run,
            judge_provider=judge_provider,
            judge_model=args.judge_model,
        )
        return

    default_split_file = EVALS_ROOT / "teaching_knowledge" / "data" / "splits.json"
    split_path = args.split_file
    if split_path is None and args.split != "all" and default_split_file.exists():
        split_path = default_split_file
    if args.split != "all" and split_path is None:
        raise FileNotFoundError(
            f"--split {args.split} requires a splits.json file. "
            f"Expected at {default_split_file}, or pass --split-file explicitly."
        )
    run(
        limit=args.limit,
        out_path=args.out,
        dry_run=args.dry_run,
        split=args.split,
        split_path=split_path,
        teacher_model=args.teacher_model,
        judge_model=args.judge_model,
        atomic_threshold=args.atomic_threshold,
        system_prompt_path=args.system_prompt,
    )


if __name__ == "__main__":
    main()
