"""GATE 1 analyzer (apps/evals env): turn corrupt bundles + warp maps into a report.

Reads the meta files written by claim_measurement.gate1.build_corrupt_bundles,
loads each (clean_bundle, corrupt_bundle, warp_map) triple, and computes per-bar
localization error against the SHIPPED LocationResolver. Reports resolvable-rate
and within-tolerance accuracy pooled overall and grouped by corruption kind.

Bar-range localization is dimension-independent here; it governs TIMING bar-range
claims. Dynamics/pedaling claims are whole_piece (trivially resolvable from the
anchor span), so their GATE risk is measurement robustness, not localization.

Usage:
    uv run --extra all python -m claim_taxonomy.gate1.analyze \
        --out-root <gate1_root> --report <out.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from claim_taxonomy.gate1.localization import (
    accuracy_at_tolerances,
    bar_localization_deltas,
)

DEFAULT_TOLERANCES_SEC = [0.1, 0.25, 0.5, 1.0, 1.5]


def report_from_loaded(loaded: list[dict], tolerances_sec: list[float]) -> dict:
    """Aggregate already-loaded (clean, corrupt, warp) triples into a GATE report."""
    all_deltas = []
    by_kind: dict[str, list] = {}
    rows = []
    for item in loaded:
        deltas = bar_localization_deltas(
            item["clean_bundle"], item["corrupt_bundle"], item["warp_map"]
        )
        all_deltas.extend(deltas)
        by_kind.setdefault(item["kind"], []).extend(deltas)
        rows.append({
            "piece": item["piece"], "video": item["video"],
            "spec_id": item["spec_id"], "kind": item["kind"],
            **accuracy_at_tolerances(deltas, tolerances_sec),
        })
    return {
        "tolerances_sec": tolerances_sec,
        "overall": accuracy_at_tolerances(all_deltas, tolerances_sec),
        "by_kind": {k: accuracy_at_tolerances(v, tolerances_sec) for k, v in by_kind.items()},
        "rows": rows,
    }


def load_meta_triples(out_root: Path) -> list[dict]:
    """Load every <piece>/<video>.meta.json under out_root with its bundles inlined."""
    loaded: list[dict] = []
    for meta_path in sorted(out_root.glob("*/*.meta.json")):
        meta = json.loads(meta_path.read_text())
        clean_path = Path(meta["clean_bundle"])
        corrupt_path = Path(meta["corrupt_bundle"])
        if not clean_path.exists() or not corrupt_path.exists():
            raise FileNotFoundError(
                f"bundle missing for {meta_path}: clean={clean_path} corrupt={corrupt_path}"
            )
        loaded.append({
            "piece": meta["piece"], "video": meta["video"],
            "spec_id": meta["spec_id"], "kind": meta["kind"],
            "clean_bundle": json.loads(clean_path.read_text()),
            "corrupt_bundle": json.loads(corrupt_path.read_text()),
            "warp_map": meta["warp_map"],
        })
    return loaded


def _fmt_block(name: str, agg: dict, tolerances: list[float]) -> str:
    md = agg["abs_delta_median"]
    p90 = agg["abs_delta_p90"]
    head = (
        f"{name:14s} n={agg['n_total']:4d} resolvable={agg['resolvable_rate']:.2f} "
        f"med|d|={'-' if md is None else f'{md:.3f}s'} p90|d|={'-' if p90 is None else f'{p90:.3f}s'}"
    )
    tol_bits = "  ".join(
        f"<={t}s:{agg['tolerances'][str(t)]['within_over_total']:.2f}" for t in tolerances
    )
    return head + "\n                 " + tol_bits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_taxonomy.gate1.analyze")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="gate1 output root containing <piece>/<video>.meta.json")
    parser.add_argument("--report", type=Path, default=None, help="write report JSON here")
    parser.add_argument("--tolerances", type=float, nargs="+", default=DEFAULT_TOLERANCES_SEC)
    args = parser.parse_args(argv)

    loaded = load_meta_triples(args.out_root)
    if not loaded:
        raise FileNotFoundError(f"no .meta.json triples under {args.out_root}")
    report = report_from_loaded(loaded, args.tolerances)

    print("=== GATE 1: bar-range localization robustness (within_over_total) ===")
    print(_fmt_block("OVERALL", report["overall"], args.tolerances))
    print("--- by corruption kind ---")
    for kind in sorted(report["by_kind"]):
        print(_fmt_block(kind, report["by_kind"][kind], args.tolerances))

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
