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
import json
import time
from pathlib import Path
from typing import Any

import yaml

from shared.style_rules import get_style_guidance

# Root paths
EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parents[1]
CACHE_DIR = REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
SKILL_EVAL_DIR = REPO_ROOT / "model" / "data" / "evals" / "skill_eval"
RESULTS_DIR = EVALS_ROOT / "results"

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

# Copied from apps/api/src/services/prompts.ts (SESSION_SYNTHESIS_SYSTEM)
SESSION_SYNTHESIS_SYSTEM = """You are a warm, perceptive piano teacher reviewing a practice session. You watched the entire session and now give your student one cohesive, encouraging response.

## What you receive

A JSON object with the full session context: duration, practice pattern (modes and transitions), top teaching moments (dimensions with scores and deviations from baseline), drilling progress, and student memory.

## How to respond

1. Start with what went well -- acknowledge effort and specific improvements.
2. Identify the 1-2 most important things to work on, grounded in the session data.
3. If drilling occurred, comment on the progression (first vs final scores).
4. Frame suggestions as actionable practice strategies, not abstract criticism.
5. Keep it conversational -- 3-6 sentences. You are talking TO the student.
6. Reference specific musical details (bars, sections, dimensions) when the data supports it.
7. Do NOT mention scores, numbers, or model outputs directly. Translate them into musical language.
8. Do NOT list all dimensions. Focus on what matters most for THIS session.

## Calibration

The MuQ audio model has R2~0.5 and 80% pairwise accuracy. Scores are directional signals, not precise measurements. A deviation of 0.1 is noise; 0.2+ is meaningful. Use deviations to identify patterns, not to make absolute claims."""


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


def build_synthesis_user_msg(
    muq_means: dict[str, float],
    duration_seconds: float,
    meta: dict[str, Any],
) -> str:
    """Build the user message for Sonnet, matching prod buildSynthesisFraming()."""
    deviations = {
        dim: muq_means[dim] - SCALER_MEAN[dim]
        for dim in DIMS
        if dim in muq_means
    }

    # Top moments: largest absolute deviation, floor at 0.05 to exclude noise
    top_moments = sorted(
        [
            {
                "dimension": dim,
                "score": muq_means[dim],
                "deviation_from_mean": round(dev, 3),
                "direction": "above_average" if dev > 0 else "below_average",
            }
            for dim, dev in deviations.items()
            if abs(dev) >= 0.05
        ],
        key=lambda x: abs(float(x["deviation_from_mean"])),
        reverse=True,
    )[:4]

    session_data = {
        "duration_minutes": round(duration_seconds / 60, 1),
        "practice_pattern": "continuous_play",
        "top_moments": top_moments,
        "drilling_records": [],
        "piece": {
            "title": meta["title"],
            "composer": meta["composer"],
            "skill_level": meta["skill_bucket"],
        },
    }

    guidance = get_style_guidance(meta.get("composer", ""))

    parts: list[str] = [
        "<session_data>",
        json.dumps(session_data, indent=2),
        "</session_data>",
    ]
    if guidance:
        parts.append("")
        parts.append(guidance)
    parts.append("")
    parts.append(
        "<task>Write <analysis>...</analysis> first as a reasoning scratchpad "
        "(this will be stripped). Then write your teacher response: 3-6 sentences, "
        "conversational, warm, specific. Do not mention scores or numbers. Focus on "
        "what matters most for this session.</task>"
    )
    return "\n".join(parts)


def extract_teacher_response(raw: str) -> str:
    """Strip the <analysis>...</analysis> scratchpad from synthesis output."""
    if "<analysis>" not in raw:
        return raw.strip()
    parts = raw.split("</analysis>", 1)
    return parts[1].strip() if len(parts) == 2 else raw.strip()


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
            if rid:
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


def run(
    limit: int | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
    split: str = "all",
    split_path: Path | None = None,
) -> None:
    from teaching_knowledge.llm_client import LLMClient
    from shared.judge import judge_synthesis_v2

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

    synthesis_client = LLMClient(provider="anthropic", tier="quality")
    print(f"Synthesis: {synthesis_client.model}")
    if not dry_run:
        print("Judge:     @cf/google/gemma-4-26b-a4b-it (workers-ai)")

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
                user_msg = build_synthesis_user_msg(muq_means, duration_seconds, meta)

                t0 = time.monotonic()
                raw = synthesis_client.complete(
                    user=user_msg,
                    system=SESSION_SYNTHESIS_SYSTEM,
                    max_tokens=1024,
                )
                synthesis_latency_ms = (time.monotonic() - t0) * 1000
                synthesis_text = extract_teacher_response(raw)

                if dry_run:
                    result = {
                        "recording_id": recording_id,
                        "piece_slug": meta["piece_slug"],
                        "title": meta["title"],
                        "composer": meta["composer"],
                        "skill_bucket": meta["skill_bucket"],
                        "muq_means": muq_means,
                        "synthesis_text": synthesis_text,
                        "synthesis_latency_ms": round(synthesis_latency_ms),
                        "judge_dimensions": [],
                        "judge_model": "dry_run",
                        "judge_latency_ms": 0,
                        "error": "",
                    }
                else:
                    judge_context = {
                        "piece_name": meta["title"],
                        "composer": meta["composer"],
                        "skill_level": meta["skill_bucket"],
                    }
                    judge_result = judge_synthesis_v2(
                        synthesis_text=synthesis_text,
                        context=judge_context,
                        provider="workers-ai",
                    )
                    result = {
                        "recording_id": recording_id,
                        "piece_slug": meta["piece_slug"],
                        "title": meta["title"],
                        "composer": meta["composer"],
                        "skill_bucket": meta["skill_bucket"],
                        "muq_means": muq_means,
                        "synthesis_text": synthesis_text,
                        "synthesis_latency_ms": round(synthesis_latency_ms),
                        "judge_dimensions": [
                            {
                                "criterion": d.criterion,
                                "score": d.score,
                                "evidence": d.evidence,
                                "reason": d.reason,
                            }
                            for d in judge_result.dimensions
                        ],
                        "judge_model": judge_result.model,
                        "judge_latency_ms": round(judge_result.latency_ms),
                        "error": "",
                    }

            except Exception as exc:
                errors += 1
                result = {
                    "recording_id": recording_id,
                    "piece_slug": meta["piece_slug"],
                    "title": meta["title"],
                    "composer": meta["composer"],
                    "skill_bucket": meta["skill_bucket"],
                    "muq_means": muq_means,
                    "synthesis_text": "",
                    "synthesis_latency_ms": 0,
                    "judge_dimensions": [],
                    "judge_model": "",
                    "judge_latency_ms": 0,
                    "error": str(exc),
                }

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
    args = parser.parse_args()
    default_split_file = EVALS_ROOT / "teaching_knowledge" / "data" / "splits.json"
    split_path = args.split_file
    if split_path is None and args.split != "all" and default_split_file.exists():
        split_path = default_split_file
    run(
        limit=args.limit,
        out_path=args.out,
        dry_run=args.dry_run,
        split=args.split,
        split_path=split_path,
    )


if __name__ == "__main__":
    main()
