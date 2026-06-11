"""Opus proxy rater for rubric calibration Phase 1.

Uses claude-opus-4-7 as a synthetic rater across all 220 calibration items
(200 main + 20 anchors). Runs API calls concurrently (8 at a time), collects
results in memory, then writes sequentially to avoid file-concurrency issues.

After all ratings complete, automatically runs the full analysis pipeline:
  analyze_calibration.calibrate() -> analyze_drift.analyze_drift() -> emit_recipe.emit()

Outputs:
  artifacts/opus_ratings.jsonl      -- append-only rating events
  artifacts/filter_recipe_opus.py   -- Stage 2 SFT filter recipe

Usage (from apps/evals/):
    uv run python -m teacher_model.calibration.run_opus_rating_session
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from shared.gateway import async_anthropic_client

from teacher_model.calibration.analyze_calibration import calibrate
from teacher_model.calibration.analyze_drift import analyze_drift
from teacher_model.calibration.emit_recipe import emit
from teacher_model.calibration.rater_cli import (
    PHASE_1_SUB_SCORES,
    capture_synthesis_ratings,
    redact_for_rater,
)

_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_MANIFEST_PATH = _ARTIFACTS_DIR / "manifest.json"
_RATINGS_PATH = _ARTIFACTS_DIR / "opus_ratings.jsonl"
_RECIPE_PATH = _ARTIFACTS_DIR / "filter_recipe_opus.py"
_BASELINE_PATH = Path(__file__).parent.parent.parent / "results" / "baseline_v1.jsonl"
_RUBRIC_PATH = Path(__file__).parent.parent.parent / "shared" / "prompts" / "rubric_definition.json"

_MODEL = "claude-opus-4-7"
_CONCURRENCY = 8

# Maps (sub_score_id, criterion_slug, leg) in PHASE_1_SUB_SCORES order.
_SUB_SCORE_MAP: list[tuple[str, str, str]] = [
    ("ascf_process",              "ascf",              "process"),
    ("concrete_artifact_process", "concrete_artifact", "process"),
    ("praise_process",            "praise",            "process"),
    ("autonomy_process",          "autonomy",          "process"),
    ("scaffolded_process",        "scaffolded",        "process"),
    ("style_process",             "style",             "process"),
    ("tone_process",              "tone",              "process"),
    ("autonomy_outcome",          "autonomy",          "outcome"),
    ("tone_outcome",              "tone",              "outcome"),
    ("concrete_artifact_outcome", "concrete_artifact", "outcome"),
    ("praise_outcome",            "praise",            "outcome"),
]

# Slug -> display name (matches rubric_definition.json key order).
_CRITERION_NAMES: dict[str, str] = {
    "ascf":              "Audible-Specific Corrective Feedback",
    "concrete_artifact": "Concrete Artifact Provision",
    "praise":            "Specific Positive Praise",
    "autonomy":          "Autonomy-Supporting Motivation",
    "scaffolded":        "Scaffolded Guided Discovery",
    "style":             "Style-Consistent Musical Language",
    "tone":              "Appropriate Tone & Language",
}


def _build_system_prompt(rubric: dict) -> str:
    dims_text = []
    for dim, slug in zip(rubric["dimensions"], _CRITERION_NAMES):
        scale = dim["scale"]
        dims_text.append(
            f"**{dim['name']}** — {dim['description']}\n"
            f"  0: {scale['0']}\n"
            f"  1: {scale['1']}\n"
            f"  2: {scale['2']}\n"
            f"  3: {scale['3']}"
        )
    return (
        "You are an expert evaluator of AI-generated piano teaching feedback.\n\n"
        "Rate each of the 7 rubric dimensions on two sub-scores:\n"
        "- **process** (0–3): did the teacher's response demonstrate this behavior?\n"
        "- **outcome** (0–3): would the student actually benefit from this?\n\n"
        "Scale anchors (apply to both sub-scores):\n"
        "  0 = absent or harmful\n"
        "  1 = attempted but weak or vague\n"
        "  2 = present and adequate\n"
        "  3 = specific, well-executed, exemplary\n\n"
        "Rubric dimensions:\n\n"
        + "\n\n".join(dims_text)
        + "\n\nAlways use the submit_ratings tool to record your assessment."
    )


def _build_rating_tool() -> dict:
    criterion_schema = {
        slug: {
            "type": "object",
            "properties": {
                "process": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "Process sub-score (0–3)",
                },
                "outcome": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "Outcome sub-score (0–3)",
                },
                "evidence": {
                    "type": "string",
                    "description": "Short quote from the synthesis supporting the score",
                },
                "reason": {
                    "type": "string",
                    "description": "One-sentence explanation of the score",
                },
            },
            "required": ["process", "outcome", "evidence", "reason"],
        }
        for slug in _CRITERION_NAMES
    }
    return {
        "name": "submit_ratings",
        "description": "Submit process and outcome scores for all 7 rubric dimensions.",
        "input_schema": {
            "type": "object",
            "properties": criterion_schema,
            "required": list(_CRITERION_NAMES.keys()),
        },
    }


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            sid = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[sid] = row
    return index


def _count_rated(ratings_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not ratings_path.exists():
        return counts
    with ratings_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event_type") == "rating":
                counts[rec["synth_id"]] += 1
    return counts


def _build_sequence(manifest: dict) -> list[dict]:
    items: list[dict] = []
    for entry in manifest["main"]:
        items.append({"type": "main", "synth_id": entry["synth_id"]})
    for anchor in manifest["anchors"]:
        items.append({
            "type": "anchor",
            "synth_id": anchor["synth_id_displayed"],
            "anchor_origin_id": anchor["synth_id"],
        })
    return items


def _extract_ratings(tool_input: dict) -> list[tuple[int, str, str]]:
    """Extract (value, evidence, reason) in PHASE_1_SUB_SCORES order."""
    result: list[tuple[int, str, str]] = []
    for _sub_score, slug, leg in _SUB_SCORE_MAP:
        criterion = tool_input[slug]
        result.append((int(criterion[leg]), criterion["evidence"], criterion["reason"]))
    return result


async def _rate_synthesis(
    client: Any,
    row: dict,
    item: dict,
    system_prompt: str,
    rating_tool: dict,
    semaphore: asyncio.Semaphore,
) -> list[tuple[int, str, str]]:
    user_content = (
        f"Piece: {row.get('title', 'Unknown')} by {row.get('composer', 'Unknown')}\n"
        f"Skill level: bucket {row.get('skill_bucket', '?')} (1=beginner, 5=advanced)\n\n"
        f"SYNTHESIS TO EVALUATE:\n{row['synthesis_text']}"
    )
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model=_MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    tools=[rating_tool],
                    tool_choice={"type": "tool", "name": "submit_ratings"},
                )
                tool_input = response.content[0].input
                return _extract_ratings(tool_input)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Opus rating failed after 3 attempts for {item['synth_id']}: {exc}"
                    ) from exc
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _write_ratings(
    baseline: dict,
    item: dict,
    collected: list[tuple[int, str, str]],
    ratings_path: Path,
) -> None:
    origin_id = item.get("anchor_origin_id", item["synth_id"])
    row = baseline[origin_id]
    redacted = redact_for_rater(row)
    redacted["synth_id"] = item["synth_id"]
    if item["type"] == "anchor":
        redacted["anchor_origin_id"] = item["anchor_origin_id"]

    it = iter(collected)
    capture_synthesis_ratings(
        redacted_row=redacted,
        sub_scores=PHASE_1_SUB_SCORES,
        session_id=str(uuid.uuid4()),
        session_idx_start=1,
        output_path=ratings_path,
        input_provider=lambda _, __: next(it),
    )


async def _run_ratings(
    baseline: dict,
    sequence: list[dict],
    rated_counts: dict[str, int],
    system_prompt: str,
    rating_tool: dict,
    ratings_path: Path,
) -> None:
    client = async_anthropic_client()
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    pending = [
        item for item in sequence
        if rated_counts.get(item["synth_id"], 0) < len(PHASE_1_SUB_SCORES)
    ]

    print(f"Sending {len(pending)} items to {_MODEL} ({_CONCURRENCY} concurrent)...")

    completed = 0

    async def _task(item: dict) -> tuple[dict, list[tuple[int, str, str]]]:
        origin_id = item.get("anchor_origin_id", item["synth_id"])
        row = baseline[origin_id]
        ratings = await _rate_synthesis(client, row, item, system_prompt, rating_tool, semaphore)
        return item, ratings

    tasks = [asyncio.create_task(_task(item)) for item in pending]
    results: list[tuple[dict, list[tuple[int, str, str]]]] = []

    for coro in asyncio.as_completed(tasks):
        item, ratings = await coro
        results.append((item, ratings))
        completed += 1
        if completed % 10 == 0 or completed == len(pending):
            print(f"  {completed}/{len(pending)} rated...")

    # Write sequentially to avoid interleaved appends.
    print("Writing ratings to disk...")
    # Restore original sequence order before writing.
    order = {item["synth_id"]: i for i, item in enumerate(sequence)}
    results.sort(key=lambda r: order[r[0]["synth_id"]])
    for item, ratings in results:
        _write_ratings(baseline, item, ratings, ratings_path)
    print(f"Wrote {len(results)} syntheses ({len(results) * len(PHASE_1_SUB_SCORES)} events) to {ratings_path}")


def _run_analysis(ratings_path: Path, baseline_path: Path) -> None:
    print("\nRunning calibration analysis...")
    cal_report = calibrate(ratings_path=ratings_path, baseline_path=baseline_path)
    drift_report = analyze_drift(ratings_path=ratings_path, judge_runs_path=None)
    emit(calibration_report=cal_report, drift_report=drift_report, output_path=_RECIPE_PATH)

    print(f"\n--- Calibration Report ---")
    print(f"Threshold decision agreement: {cal_report['threshold_decision_agreement']:.3f}")
    print(f"Threshold decision kappa:     {cal_report['threshold_decision_kappa']:.3f}")
    print(f"Trusted sub-scores (Phase 1): {cal_report['n_phase_1_trusted']}/11")
    print(f"Aggregate gate pass:          {cal_report['aggregate_gate_pass']}")
    print(f"\nPer-sub-score kappa:")
    for sub in PHASE_1_SUB_SCORES:
        k = cal_report["per_sub_score_kappa"].get(sub, float("nan"))
        bucket = cal_report["buckets"].get(sub, "?")
        print(f"  {sub:<32} kappa={k:+.3f}  [{bucket}]")
    print(f"\nFilter recipe written to: {_RECIPE_PATH}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_BASELINE_PATH,
        help=f"Path to baseline JSONL (default: {_BASELINE_PATH})",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=_RATINGS_PATH,
        help=f"Path to ratings output JSONL (default: {_RATINGS_PATH})",
    )
    args = parser.parse_args()
    baseline_path = args.baseline
    ratings_path = args.ratings

    for path, name in [(_MANIFEST_PATH, "manifest.json"), (baseline_path, baseline_path.name),
                       (_RUBRIC_PATH, "rubric_definition.json")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}.", file=sys.stderr)
            sys.exit(1)

    manifest = json.loads(_MANIFEST_PATH.read_text())
    rubric = json.loads(_RUBRIC_PATH.read_text())
    baseline = _load_baseline(baseline_path)
    sequence = _build_sequence(manifest)
    rated_counts = _count_rated(ratings_path)

    n_done = sum(
        1 for item in sequence
        if rated_counts.get(item["synth_id"], 0) >= len(PHASE_1_SUB_SCORES)
    )
    total = len(sequence)
    print(f"Total items: {total} | Already rated: {n_done} | Remaining: {total - n_done}")

    if n_done < total:
        system_prompt = _build_system_prompt(rubric)
        rating_tool = _build_rating_tool()
        asyncio.run(_run_ratings(baseline, sequence, rated_counts, system_prompt, rating_tool, ratings_path))
    else:
        print("All items already rated.")

    _run_analysis(ratings_path=ratings_path, baseline_path=baseline_path)


if __name__ == "__main__":
    main()
