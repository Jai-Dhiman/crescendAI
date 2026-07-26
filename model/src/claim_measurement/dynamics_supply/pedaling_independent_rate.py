# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0"]
# ///
"""FRONT 9 pedaling oracle: GT-MIDI-anchored INDEPENDENT pedaling faithfulness rate (#101).

The pedaling analog of independent_rate.py (dynamics). Truth = ground-truth MIDI sustain-pedal
on-fraction (fraction of window CC64 is held down); score = the production PedalingMeasurer's
AMT on-fraction (from transkun CC64) + frozen router. Two INDEPENDENT measurements of the same
performance -> non-circular. Runs on MAESTRO real-audio bundles (render_maestro_bundles.py),
which carry gt_pedal_onfraction + gt_corpus_pedal_median.

Reference recalibration: REFERENCE_FRACTION (aria=0.4623, locked:false) is recalibrated to the
AMT (transkun) corpus median on-fraction over THESE bundles -- the same per-substrate step the
dynamics rate needed. Without it the number measures aria's pedal scale, not transkun's.

Over-pedal scoping: the taxonomy declares pedaling substrate_insensitive_polarity={whole_piece:
"+"} because ARIA's pedal head SATURATES (~0.55 ceiling) -> over-pedal "+" claims abstain. Transkun
does NOT saturate (real-audio AMT on-fraction reaches 1.0), so --lift-scoping re-enables "+" to
test whether transkun UNBLOCKS over-pedal detection (a research probe, not the production number).

Run:
    cd apps/evals && PYTHONPATH=$PWD python .../pedaling_independent_rate.py \
      --bundles .../model/data/evals/maestro_indep_bundles \
      --out     .../model/data/results/pedaling_maestro_rate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]
TAXONOMY_PATH = REPO / "apps/evals/claim_taxonomy/claim_taxonomy.json"
TAU_GT_DEFAULT = 0.25          # GT pedal-on-fraction deadband (mirrors the taxonomy tolerance)
TAU_GT_SWEEP = (0.15, 0.25, 0.35)

# reuse the dynamics aggregation/CI verbatim -- the record shape is identical
sys.path.insert(0, str(REPO / "model/src"))
from claim_measurement.dynamics_supply.independent_rate import aggregate, bootstrap_ci  # noqa: E402


def gt_pedal_polarity(gt_onfrac: float, median: float, tau_gt: float) -> str:
    d = gt_onfrac - median
    if d > tau_gt:
        return "+"      # over-pedal vs corpus
    if d < -tau_gt:
        return "-"      # under-pedal vs corpus
    return "neutral"


def _label(pol: str) -> str:
    return {"+": "over", "-": "under", "neutral": "normal"}[pol]


def amt_corpus_reference(bundles: list[dict]) -> float:
    """Transkun corpus median AMT on-fraction -- the recalibrated whole_piece reference."""
    fracs = []
    for b in bundles:
        pe = b.get("pedal_events") or []
        dur = b.get("duration_sec") or 0.0
        if dur <= 0:
            continue
        down, t, st = 0.0, 0.0, 0
        for e in sorted(pe, key=lambda e: e["time"]):
            if st >= 64:
                down += e["time"] - t
            t, st = e["time"], e["value"]
        if st >= 64:
            down += dur - t
        fracs.append(down / dur)
    return float(np.median(fracs)) if fracs else 0.4623


def score_bundle(bundle: dict, taxonomy: dict, tau_gt: float) -> dict:
    sys.path.insert(0, str(REPO / "apps/evals"))
    from claim_taxonomy.verifier.orchestrator import verify
    from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

    gt_pol = gt_pedal_polarity(bundle["gt_pedal_onfraction"], bundle["gt_corpus_pedal_median"], tau_gt)
    claim = {
        "dimension": "pedaling",
        "location": "whole_piece",
        "polarity": gt_pol,
        "proposition": f"overall pedaling is {_label(gt_pol)} (ground-truth-anchored)",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    r = verify(claim, bundle, taxonomy, engine)
    committed = r.verdict in ("SUPPORTED", "REFUTED")
    return {
        "segment": bundle["video_id"],
        "gt_pedal_onfraction": bundle["gt_pedal_onfraction"],
        "gt_label": _label(gt_pol),
        "gt_polarity": gt_pol,
        "amt_d": r.measured_value,
        "tau": r.tau,
        "error_bar": r.error_bar,
        "verdict": r.verdict,
        "reason": r.reason_code,
        "committed": committed,
    }


def _load_bundles(bundle_dir: Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        if "gt_pedal_onfraction" not in b:
            raise ValueError(f"{p} is not a pedaling oracle bundle (no gt_pedal_onfraction)")
        out.append(b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.pedaling_independent_rate")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau-gt", type=float, default=TAU_GT_DEFAULT)
    ap.add_argument("--lift-scoping", action="store_true",
                    help="remove substrate_insensitive_polarity to probe over-pedal detection")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    if args.lift_scoping:
        taxonomy["dimensions"]["pedaling"].pop("substrate_insensitive_polarity", None)

    bundles = _load_bundles(args.bundles)
    if not bundles:
        raise SystemExit(f"no pedaling oracle bundles in {args.bundles}")

    # recalibrate the AMT reference to this substrate/corpus BEFORE scoring
    from claim_taxonomy.verifier.measurers import pedaling as ped
    old_ref = ped.REFERENCE_FRACTION
    ped.REFERENCE_FRACTION = amt_corpus_reference(bundles)

    records = [score_bundle(b, taxonomy, args.tau_gt) for b in bundles]
    agg = aggregate(records)
    committed = [r for r in records if r["committed"]]
    ci = bootstrap_ci(committed, n_boot=args.n_boot)
    sweep = {}
    for t in TAU_GT_SWEEP:
        agg_t = aggregate([score_bundle(b, taxonomy, t) for b in bundles])
        sweep[str(t)] = {"rate": agg_t["faithfulness_rate"], "n_committed": agg_t["n_committed"]}

    gd_pass = (agg["n_committed"] >= 30) and (ci["half_width"] is not None) and (ci["half_width"] <= 0.05)
    result = {
        "gate": "G-F pedaling (independent, GT-anchored, real audio)",
        "dimension": "pedaling", "location": "whole_piece",
        "statistic_scored": "amt_sustain_on_fraction (transkun CC64)",
        "truth_signal": "ground_truth_midi_sustain_on_fraction (INDEPENDENT)",
        "reference_fraction_recalibrated": ped.REFERENCE_FRACTION,
        "reference_fraction_aria_old": old_ref,
        "over_pedal_scoping_lifted": args.lift_scoping,
        "tau_gt": args.tau_gt, "n_segments": len(bundles),
        **agg, "ci95": {**ci, "method": "segment_bootstrap"},
        "tau_gt_sensitivity": sweep, "gd_pass": gd_pass,
        "note": ("ORACLE: truth = GT MIDI pedal on-fraction, score = transkun AMT CC64 on-fraction. "
                 "Reference recalibrated to the transkun corpus median. Over-pedal '+' abstains unless "
                 "--lift-scoping (aria saturated; transkun may not)."),
        "per_segment": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== GT-anchored INDEPENDENT pedaling faithfulness (real audio, oracle) ===", flush=True)
    print(f"reference recalibrated {old_ref:.4f} -> {ped.REFERENCE_FRACTION:.4f}  "
          f"scoping_lifted={args.lift_scoping}", flush=True)
    print(f"segments={len(bundles)} committed={agg['n_committed']} "
          f"(sup={agg['n_supported']} ref={agg['n_refuted']})", flush=True)
    if agg["faithfulness_rate"] is not None:
        print(f"RATE = {agg['faithfulness_rate']:.3f}  95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]  "
              f"half={ci['half_width']:.3f}", flush=True)
    print(f"confusion: {agg['confusion_gt_label_x_verdict']}", flush=True)
    print(f"abstention: {agg['abstention_histogram']}", flush=True)
    print(f"G-D PASS = {gd_pass}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
