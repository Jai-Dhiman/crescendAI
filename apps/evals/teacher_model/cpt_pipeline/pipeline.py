"""CLI driver for the cpt_pipeline.

Usage:
  uv run python -m teacher_model.cpt_pipeline.pipeline run \\
    --corpus-dir <path> --provenance-dir <path> --out-dir <path> \\
    --repo-id <hf_repo_id> [--push-disabled]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teacher_model.cpt_pipeline.dedup import run_dedup
from teacher_model.cpt_pipeline.hf_publish import run_publish
from teacher_model.cpt_pipeline.ingest import run_ingest
from teacher_model.cpt_pipeline.split import run_split
from teacher_model.cpt_pipeline.structural_filter import run_filter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPT corpus preprocessing pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run all pipeline stages.")
    run_p.add_argument("--corpus-dir", type=Path, required=True)
    run_p.add_argument("--provenance-dir", type=Path, required=True)
    run_p.add_argument("--out-dir", type=Path, required=True)
    run_p.add_argument("--repo-id", type=str, required=True)
    run_p.add_argument("--push-disabled", action="store_true",
                       help="Skip stage 5 (HF Hub push). Useful for tests and dry runs.")
    run_p.add_argument("--seed", type=int, default=42)
    return parser


def run_pipeline(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[stage 1/5] ingest: {args.corpus_dir} -> {out}/1_ingest")
        ingest_manifest = run_ingest(args.corpus_dir, args.provenance_dir, out / "1_ingest")
        print(f"[stage 2/5] structural_filter: {ingest_manifest} -> {out}/2_filter")
        filter_manifest = run_filter(ingest_manifest, out / "2_filter")
        print(f"[stage 3/5] dedup: {filter_manifest} -> {out}/3_dedup")
        dedup_manifest = run_dedup(filter_manifest, out / "3_dedup")
        print(f"[stage 4/5] split: {dedup_manifest} -> {out}/4_split")
        train_path, val_path = run_split(dedup_manifest, out / "4_split", seed=args.seed)
    except (OSError, ValueError) as e:
        print(f"[pipeline] stage failed: {e}", file=sys.stderr)
        return 1

    if args.push_disabled:
        print("[stage 5/5] push DISABLED via --push-disabled")
        return 0

    try:
        print(f"[stage 5/5] hf_publish: {args.repo_id} (private)")
        url = run_publish(train_path, val_path, args.repo_id, private=True, card_out_dir=out / "5_publish")
        print(f"published: {url}")
    except (OSError, RuntimeError) as e:
        print(f"[pipeline] publish failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline(sys.argv[1:]))
