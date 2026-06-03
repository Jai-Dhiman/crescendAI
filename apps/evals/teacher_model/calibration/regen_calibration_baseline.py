"""Regenerate synthesis + judge for the 182 calibration manifest recordings only.

Reads the calibration manifest to get the target recording_ids, then runs
Sonnet synthesis (new prompt) + Gemma judge (new rubric) for only those
recordings. Writes to results/calibration_baseline.jsonl (resume-safe).

Usage (from apps/evals/):
    uv run python -m teacher_model.calibration.regen_calibration_baseline
    uv run python -m teacher_model.calibration.regen_calibration_baseline --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EVALS_ROOT.parents[1]

CACHE_DIR = REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
CALIBRATION_MANIFEST = Path(__file__).parent / "artifacts" / "manifest.json"
OUT_PATH = EVALS_ROOT / "results" / "calibration_baseline.jsonl"


def _manifest_recording_ids() -> set[str]:
    data = json.loads(CALIBRATION_MANIFEST.read_text())
    ids: set[str] = set()
    for entry in data["main"]:
        ids.add(entry["synth_id"].split("__")[1])
    for anchor in data["anchors"]:
        ids.add(anchor["synth_id"].split("__")[1])
    return ids


def _load_completed(out_path: Path) -> set[str]:
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


def _default_driver(wrangler_url, recording_cache, student_id, piece_query):
    """Default DO-replay driver: real run_recording over wrangler dev."""
    import asyncio

    from shared.pipeline_client import run_recording

    return asyncio.run(
        run_recording(
            wrangler_url,
            recording_cache,
            student_id=student_id,
            piece_query=piece_query,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip judge, synthesis only")
    parser.add_argument("--wrangler-url", default="http://localhost:8787",
                        help="wrangler dev base URL (DO-path synthesis).")
    args = parser.parse_args()

    # Import run_eval helpers — must run from apps/evals/
    try:
        from teaching_knowledge.run_eval import (
            build_do_row,
            load_manifests,
        )
        from shared.judge import judge_synthesis_v2
        from shared.provenance import make_run_provenance as _mkprov
    except ImportError as e:
        print(f"ERROR: {e}\nRun from apps/evals/ with uv run.", file=sys.stderr)
        sys.exit(1)

    driver = _default_driver

    target_ids = _manifest_recording_ids()
    completed = _load_completed(OUT_PATH)
    remaining = target_ids - completed

    print(f"Calibration manifest recording_ids: {len(target_ids)}")
    print(f"Already done: {len(completed)}")
    print(f"Remaining:    {len(remaining)}")

    if not remaining:
        print("All calibration recordings already synthesized.")
        return

    manifest_lookup = load_manifests()
    cache_files = {f.stem: f for f in CACHE_DIR.glob("*.json") if f.name != "_fingerprint.json"}

    provenance = _mkprov()
    print(f"Synthesis: real SessionBrain DO via {args.wrangler_url} (buildSynthesisFraming)")
    print(f"Judge prompt:     synthesis_quality_judge_v2.txt")
    print(f"run_id: {provenance.run_id}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    with OUT_PATH.open("a") as fout:
        for rid in sorted(remaining):
            cache_path = cache_files.get(rid)
            if cache_path is None:
                print(f"  SKIP {rid}: no cache file")
                continue

            meta = manifest_lookup.get(rid)
            if meta is None:
                print(f"  SKIP {rid}: no skill_eval manifest entry")
                continue

            data = json.loads(cache_path.read_text())
            chunks = data.get("chunks", [])
            if not chunks:
                print(f"  SKIP {rid}: empty chunks")
                continue

            recording_cache = {"recording_id": rid, "chunks": chunks}
            try:
                session_result = driver(
                    args.wrangler_url,
                    recording_cache,
                    "calibration-student-001",
                    meta.get("piece_slug") or None,
                )
                row = build_do_row(
                    session_result,
                    meta,
                    judge_synthesis_v2,
                    provenance,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                print(f"  ERROR {rid}: {exc}")
                continue

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1
            print(f"  [{processed}/{len(remaining)}] {rid} ({meta['title'][:30]})")

    print(f"\nDone. Wrote {processed} rows to {OUT_PATH}")
    print(f"Pass --baseline {OUT_PATH} to run_opus_rating_session to use this file.")


if __name__ == "__main__":
    main()
