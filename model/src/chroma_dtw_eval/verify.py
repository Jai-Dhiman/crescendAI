"""Verify CLI entry point.

This is a thin assembly module -- it loads the fixture manifest, baseline file,
emits the primary scalar on stdout, and exits 0 or non-zero by comparing against
the baseline. In Task 0 the per-chunk computation is a stub that returns 100.0
(perfect alignment) so the contract is testable before the deep modules ship.
Later tasks replace the stub with real metric_aggregator calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("model/data/evals/chroma_dtw/last_run.json"),
    )
    args = parser.parse_args(argv)

    manifest_path = args.fixtures / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    baseline = json.loads(args.baseline.read_text())

    n = len(manifest["chunks"])
    primary = 100.0 if n > 0 else 0.0
    guards = {"g1": 0.0, "g2": 1.0, "g3": 100.0, "g4": 100.0, "g5": 0.0}

    regressed: list[str] = []
    if primary + 1e-9 < baseline["primary"]:
        regressed.append("primary")
    if guards["g1"] > baseline["guards"]["g1"] + 1.0:
        regressed.append("g1")
    if guards["g2"] < baseline["guards"]["g2"] - 0.02:
        regressed.append("g2")
    if guards["g3"] < baseline["guards"]["g3"] - 1.0:
        regressed.append("g3")
    if guards["g4"] < baseline["guards"]["g4"] - 1.0:
        regressed.append("g4")
    if guards["g5"] > baseline["guards"]["g5"] + 1.0:
        regressed.append("g5")

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": primary,
        "guards": guards,
        "baseline": baseline,
        "regressed": regressed,
        "n_chunks": n,
    }, indent=2))

    print(f"{primary:.4f}")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
