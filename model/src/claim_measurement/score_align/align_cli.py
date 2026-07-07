"""FRONT 7b CLI: batch-annotate cached claim bundles with score-aligned onsets.

Usage:
    cd model && uv run python -m claim_measurement.score_align.align_cli [--force]

Idempotent: bundles already carrying score_align schema v1 are skipped unless
--force. Per-clip failures (missing score mapping, parangonar timeout, degenerate
alignment) are recorded and the batch continues; exit code 1 if any clip errored
or timed out. --score-root/--bundle-root exist for worktree runs where the
__file__-anchored defaults point at a tree without data/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chroma_dtw_eval.amt_regen import DEFAULT_SCORE_BY_PIECE
from claim_measurement.extract_cli import DEFAULT_BUNDLE_ROOT, _time_limit
from claim_measurement.score_align.align_notes import (
    SCORE_ALIGN_SCHEMA,
    ScoreAlignError,
    align_bundle_file,
)


def _score_map(score_root: Path | None) -> dict[str, Path]:
    if score_root is None:
        return dict(DEFAULT_SCORE_BY_PIECE)
    return {piece: score_root / p.name for piece, p in DEFAULT_SCORE_BY_PIECE.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--score-root", type=Path, default=None,
                    help="rebase the piece->score map onto this directory")
    ap.add_argument("--only-piece", default=None)
    ap.add_argument("--timeout-sec", type=int, default=300,
                    help="per-clip parangonar guard; <=0 disables")
    ap.add_argument("--force", action="store_true",
                    help="re-align bundles that already carry score_align")
    args = ap.parse_args(argv)

    scores = _score_map(args.score_root)
    bundle_paths = sorted(args.bundle_root.glob("*/*.json"))
    if not bundle_paths:
        raise ScoreAlignError(f"no bundles under {args.bundle_root}")

    results: list[dict] = []
    for bp in bundle_paths:
        piece = bp.parent.name
        if args.only_piece and piece != args.only_piece:
            continue
        score_path = scores.get(piece)
        if score_path is None:
            results.append({"piece": piece, "bundle": bp.name, "status": "no_score"})
            continue
        body = json.loads(bp.read_text())
        if not args.force and body.get("score_align", {}).get("schema") == SCORE_ALIGN_SCHEMA:
            results.append({"piece": piece, "bundle": bp.name, "status": "skip"})
            continue
        try:
            with _time_limit(args.timeout_sec):
                meta = align_bundle_file(bp, score_path)
        except ScoreAlignError as e:
            results.append({"piece": piece, "bundle": bp.name, "status": "error",
                            "error": str(e)})
            continue
        except TimeoutError as e:
            results.append({"piece": piece, "bundle": bp.name, "status": "timeout",
                            "error": str(e)})
            continue
        results.append({
            "piece": piece, "bundle": bp.name, "status": "ok",
            "n_matched": meta["n_matched"], "n_annotated": meta["n_annotated"],
            "n_windows": meta["n_windows"], "n_notes": meta["n_notes"],
            "residual_rms_ms": round(meta["residual_rms_ms"], 1),
            "median_abs_residual_ms": round(meta["median_abs_residual_ms"], 1),
        })

    for r in results:
        print(json.dumps(r))
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"summary": counts}))
    return 1 if any(r["status"] in ("error", "timeout") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
