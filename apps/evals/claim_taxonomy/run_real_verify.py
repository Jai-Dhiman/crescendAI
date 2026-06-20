"""Run verify() on real measurement bundles and emit measured error bars.

This is a reproducible measurement harness, NOT a test. It loads the bundles produced
by model/src/claim_measurement/extract_cli.py (real AMT transcription of cached audio),
runs the shipped verify() over a handful of REAL teacher-prose claims (timing / pedaling
/ dynamics, drawn from audit/baseline_v1_audit.json), and reports the VerdictResults plus
a per-dimension measured error-bar summary.

The verifier is piece-agnostic: it measures the bundle's substrate for the claimed
dimension+location. Pairing real claim TEXT with a real bundle therefore exercises the
measurement path on real audio (the truth label is about the measured performance).

Usage:
    uv run --extra all python -m claim_taxonomy.run_real_verify \
        --bundle-root /path/to/model/data/evals/claim_bundles
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from claim_taxonomy.verifier.orchestrator import verify
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

_HERE = Path(__file__).resolve().parent
TAXONOMY_PATH = _HERE / "claim_taxonomy.json"

# Real teacher-prose propositions (verbatim from audit/baseline_v1_audit.json), one per
# admissible dimension, spanning whole_piece and a bar-range location.
REAL_CLAIMS = [
    {
        "proposition": "The tempo was steady throughout the run",
        "dimension": "timing", "location": "whole_piece", "polarity": "neutral",
        "magnitude": None,
    },
    {
        "proposition": "There was a slight rush in the ornamental passages",
        "dimension": "timing", "location": {"bar_start": 9, "bar_end": 12},
        "polarity": "-", "magnitude": None,
    },
    {
        "proposition": "Your pedaling created exactly the kind of luminous, breathing resonance the Nocturne needs",
        "dimension": "pedaling", "location": "whole_piece", "polarity": "+",
        "magnitude": None,
    },
    {
        "proposition": "The melodic dynamics need to ebb and swell like a voice",
        "dimension": "dynamics", "location": "whole_piece", "polarity": "-",
        "magnitude": None,
    },
]


def _discover_bundles(bundle_root: Path) -> list[Path]:
    return sorted(
        p for p in bundle_root.rglob("*.json")
        if p.name != "_index.json" and not p.name.endswith(".tmp")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_taxonomy.run_real_verify")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the measured error-bar summary JSON.")
    args = parser.parse_args(argv)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    bundles = _discover_bundles(args.bundle_root)
    if not bundles:
        raise FileNotFoundError(f"no bundles under {args.bundle_root}; run extract_cli first")

    rows: list[dict] = []
    for bundle_path in bundles:
        bundle = json.loads(bundle_path.read_text())
        tag = f"{bundle.get('piece_id')}/{bundle.get('video_id')}"
        n_pedals = len(bundle.get("pedal_events", []))
        n_notes = len(bundle.get("notes", []))
        print(f"\n=== bundle {tag}  (notes={n_notes}, pedal_events={n_pedals}) ===")
        for claim in REAL_CLAIMS:
            # Fresh seeded engine per claim -> deterministic error bars.
            result = verify(dict(claim), bundle, taxonomy, SubstrateErrorEngine(seed=42))
            loc = claim["location"]
            loc_s = "whole_piece" if loc == "whole_piece" else f"bars {loc['bar_start']}-{loc['bar_end']}"
            print(
                f"  [{claim['dimension']:9s} {loc_s:14s} pol={claim['polarity']:>7s}] "
                f"-> {result.verdict:12s} reason={str(result.reason_code):16s} "
                f"d={result.measured_value:+.4f}{result.units} tau={result.tau} "
                f"err={result.error_bar:.4f} n={result.event_count}"
            )
            rows.append({
                "bundle": tag, "dimension": claim["dimension"], "location": loc_s,
                "polarity": claim["polarity"], "verdict": result.verdict,
                "reason_code": result.reason_code, "d": result.measured_value,
                "tau": result.tau, "error_bar": result.error_bar,
                "event_count": result.event_count, "units": result.units,
            })

    # Per-dimension measured error-bar summary (only rows where a measurement ran, i.e.
    # event_count > 0 -> error_bar is a real measured value, not a 0.0 short-circuit).
    summary: dict[str, dict] = {}
    for dim in ("timing", "pedaling", "dynamics"):
        errs = [r["error_bar"] for r in rows if r["dimension"] == dim and r["event_count"] > 0]
        ds = [r["d"] for r in rows if r["dimension"] == dim and r["event_count"] > 0]
        if not errs:
            summary[dim] = {"measured": False, "note": "no localizable measurement on these bundles"}
            continue
        summary[dim] = {
            "measured": True,
            "units": next(r["units"] for r in rows if r["dimension"] == dim),
            "n_measurements": len(errs),
            "error_bar_min": min(errs), "error_bar_median": statistics.median(errs),
            "error_bar_max": max(errs),
            "d_min": min(ds), "d_max": max(ds),
        }

    print("\n=== per-dimension measured error-bar summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
