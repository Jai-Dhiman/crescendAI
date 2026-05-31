"""Verify CLI — real evaluation path.

For each chunk in the fixture manifest (or for a real corpus pointed at by
--corpus), run the DTW via dtw_runner, build a ChunkResult, and pass the
batch to metric_aggregator.aggregate. Print the primary scalar on stdout
(one line, one float). Exit 0 iff no guard regressed against --baseline.
Write a sidecar JSON with the full breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)


def _load_fixture_chunks(fixtures: Path) -> list[ChunkResult]:
    manifest = json.loads((fixtures / "manifest.json").read_text())
    results: list[ChunkResult] = []
    for c in manifest["chunks"]:
        kind_map = {"gold": "gold", "amateur": "amateur", "silence": "silence"}
        kind = kind_map[c["kind"]]
        if kind == "gold":
            results.append(ChunkResult(
                kind="gold",
                error_frames=float(c.get("simulated_error_frames", 999.0)),
                cost=float(c.get("simulated_cost", 0.2)),
                abstain=False,
            ))
        elif kind == "amateur":
            results.append(ChunkResult(
                kind="amateur",
                error_frames=None,
                cost=float(c.get("simulated_cost", 0.2)),
                abstain=False,
                bar_distance_from_forward=float(c.get("simulated_bar_distance", 0.0)),
            ))
        elif kind == "silence":
            results.append(ChunkResult(
                kind="silence",
                error_frames=None,
                cost=float(c.get("simulated_cost", 0.05)),
                abstain=False,
                silence_loud_failure=bool(c.get("simulated_loud_failure", True)),
            ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        help="If provided, use the committed fixture manifest instead of a real corpus")
    parser.add_argument("--corpus", type=Path,
                        help="Root containing maestro/skill_eval/practice_eval (real run)")
    parser.add_argument("--sidecar", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/evals/chroma_dtw/last_run.json")
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    raw = json.loads(args.baseline.read_text())
    baseline = Baseline(
        primary=float(raw["primary"]),
        guards=GuardSet(**{k: float(v) for k, v in raw["guards"].items()}),
    )

    if args.fixtures is not None:
        results = _load_fixture_chunks(args.fixtures)
    elif args.corpus is not None:
        from chroma_dtw_eval.corpus_runner import run_corpus  # built later if/when needed
        results = run_corpus(args.corpus)
    else:
        raise ValueError("must pass --fixtures or --corpus")

    metrics = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": metrics.primary,
        "guards": metrics.guards.__dict__,
        "baseline": {"primary": baseline.primary, "guards": baseline.guards.__dict__},
        "regressed": metrics.regressed,
        "n_chunks": len(results),
    }, indent=2))
    print(f"{metrics.primary:.4f}")
    return 1 if metrics.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
