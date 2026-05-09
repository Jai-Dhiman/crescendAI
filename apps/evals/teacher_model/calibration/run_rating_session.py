"""Interactive founder rating session for the rubric calibration protocol.

Usage (from apps/evals/):
    uv run python -m teacher_model.calibration.run_rating_session

Presents each synthesis one at a time, collects 11 sub-score ratings (value
0-3, evidence quote, reason), and writes append-only JSONL to artifacts/.

All 11 inputs are collected in memory before writing anything to disk — so
Ctrl-C during a sub-score prompt leaves the file clean and the synthesis will
be re-presented on resume.

Anchor duplicates are silently interspersed at positions 200-219 with
scrambled display IDs; the rater cannot distinguish them from regular items.
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from teacher_model.calibration.rater_cli import (
    PHASE_1_SUB_SCORES,
    capture_synthesis_ratings,
    redact_for_rater,
)

_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_MANIFEST_PATH = _ARTIFACTS_DIR / "manifest.json"
_RATINGS_PATH = _ARTIFACTS_DIR / "ratings.jsonl"
_BASELINE_PATH = Path(__file__).parent.parent.parent / "results" / "baseline_v1.jsonl"

_SUB_SCORE_LABELS: dict[str, str] = {
    "ascf_process":              "Audible-Specific Corrective Feedback — process",
    "concrete_artifact_process": "Concrete Artifact Provision — process",
    "praise_process":            "Specific Positive Praise — process",
    "autonomy_process":          "Autonomy-Supporting Motivation — process",
    "scaffolded_process":        "Scaffolded Guided Discovery — process",
    "style_process":             "Style-Consistent Musical Language — process",
    "tone_process":              "Appropriate Tone & Language — process",
    "autonomy_outcome":          "Autonomy-Supporting Motivation — outcome",
    "tone_outcome":              "Appropriate Tone & Language — outcome",
    "concrete_artifact_outcome": "Concrete Artifact Provision — outcome",
    "praise_outcome":            "Specific Positive Praise — outcome",
}

_CLEAR = "\033[2J\033[H"


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            synth_id = f"{row['piece_slug']}__{row['recording_id']}__{row['skill_bucket']}"
            index[synth_id] = row
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


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Enter a number between {lo} and {hi}.")


def _divider() -> None:
    print("-" * 60)


def _collect_sub_scores() -> list[tuple[int, str, str]]:
    collected: list[tuple[int, str, str]] = []
    for i, sub_score in enumerate(PHASE_1_SUB_SCORES, 1):
        _divider()
        label = _SUB_SCORE_LABELS[sub_score]
        print(f"  [{i}/{len(PHASE_1_SUB_SCORES)}] {label}")
        value = _prompt_int("  Score (0=low  1=developing  2=proficient  3=exemplary): ", 0, 3)
        evidence = input("  Evidence quote (or Enter to skip): ").strip()
        reason = input("  Reason for this score: ").strip()
        collected.append((value, evidence, reason))
        print()
    return collected


def _display_header(row: dict, item: dict, position: int, total: int) -> None:
    print(_CLEAR, end="", flush=True)
    tag = " [ANCHOR]" if item["type"] == "anchor" else ""
    print(f"{'=' * 60}")
    print(f"  SYNTHESIS {position} of {total}{tag}")
    print(f"  Piece:  {row.get('title', '?')}")
    print(f"  By:     {row.get('composer', '?')}   |   Skill bucket: {row.get('skill_bucket', '?')}")
    print(f"  ID:     {item['synth_id']}")
    print(f"{'=' * 60}")
    print()
    print(row.get("synthesis_text", "(no synthesis text)"))
    print()


def run() -> None:
    for path, name in [(_MANIFEST_PATH, "manifest.json"), (_BASELINE_PATH, "baseline_v1.jsonl")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}.", file=sys.stderr)
            sys.exit(1)

    manifest = json.loads(_MANIFEST_PATH.read_text())
    baseline = _load_baseline(_BASELINE_PATH)
    sequence = _build_sequence(manifest)
    total = len(sequence)

    rated_counts = _count_rated(_RATINGS_PATH)
    n_done = sum(
        1 for item in sequence
        if rated_counts.get(item["synth_id"], 0) >= len(PHASE_1_SUB_SCORES)
    )

    print("Calibration rating session")
    print(f"  Total items:  {total} ({len(manifest['main'])} main + {len(manifest['anchors'])} anchors)")
    print(f"  Rated so far: {n_done}")
    print(f"  Remaining:    {total - n_done}")
    print()

    if n_done == total:
        print("All items are rated. Run analyze_calibration.calibrate() to generate the report.")
        return

    first_unrated = next(
        i for i, item in enumerate(sequence)
        if rated_counts.get(item["synth_id"], 0) < len(PHASE_1_SUB_SCORES)
    )
    input(
        f"Starting from item {first_unrated + 1}. "
        "Press Enter to begin, or Ctrl-C to quit and resume later..."
    )

    for i in range(first_unrated, total):
        item = sequence[i]
        if rated_counts.get(item["synth_id"], 0) >= len(PHASE_1_SUB_SCORES):
            continue

        origin_id = item.get("anchor_origin_id", item["synth_id"])
        row = baseline.get(origin_id)
        if row is None:
            print(f"WARNING: {origin_id} not in baseline — skipping.")
            continue

        redacted = redact_for_rater(row)
        redacted["synth_id"] = item["synth_id"]
        if item["type"] == "anchor":
            redacted["anchor_origin_id"] = item["anchor_origin_id"]

        _display_header(row, item, i + 1, total)

        # Collect all inputs before touching the file. Ctrl-C here leaves the file untouched.
        try:
            collected = _collect_sub_scores()
        except KeyboardInterrupt:
            print("\n\nInterrupted before any events were written. Resume anytime.")
            sys.exit(0)

        # All inputs collected — write atomically.
        collected_iter = iter(collected)
        capture_synthesis_ratings(
            redacted_row=redacted,
            sub_scores=PHASE_1_SUB_SCORES,
            session_id=str(uuid.uuid4()),
            session_idx_start=1,
            output_path=_RATINGS_PATH,
            input_provider=lambda _, __: next(collected_iter),
        )

        rated_counts[item["synth_id"]] = len(PHASE_1_SUB_SCORES)
        remaining = total - sum(
            1 for seq_item in sequence
            if rated_counts.get(seq_item["synth_id"], 0) >= len(PHASE_1_SUB_SCORES)
        )
        _divider()
        print(f"Saved. {remaining} items remaining.")

        if remaining > 0:
            ans = input("Continue to next? [Y/n]: ").strip().lower()
            if ans == "n":
                print("Session paused. Run again to resume.")
                return

    print("\nAll items rated. Run analyze_calibration to generate the calibration report.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Run again to resume from where you left off.")
        sys.exit(0)
