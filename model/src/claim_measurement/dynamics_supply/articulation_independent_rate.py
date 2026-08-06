# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0"]
# ///
"""FRONT 10 (#101): GT-MIDI-anchored INDEPENDENT articulation faithfulness rate.

The articulation analog of pedaling_independent_rate.py. Truth = the ground-truth MIDI
duration/IOI ratio; score = the production ArticulationMeasurer's Transkun-offset ratio
+ the frozen router. Two independent measurements of the same performance -> non-
circular. Runs on the 188 MAESTRO real-audio bundles (render_maestro_bundles.py), which
carry gt_notes.

ORACLE design (same as FRONT 8b/9): the claim polarity IS the GT label, so the rate
isolates the verifier's SUBSTRATE faithfulness with zero teacher/extractor noise. There
is no articulation claim-supply probe behind this -- FRONT 7a showed teacher supply is a
separate question per dimension, and this number does not assume any.

Reference recalibration: the whole_piece REFERENCE_RATIO is recalibrated to the Transkun
corpus median over THESE bundles, the same per-(substrate x corpus) step dynamics and
pedaling needed.

Run:
    cd apps/evals && PYTHONPATH=$PWD python \\
      ../../model/src/claim_measurement/dynamics_supply/\\
        articulation_independent_rate.py \\
      --bundles ../../model/data/evals/maestro_indep_bundles \\
      --out     ../../model/data/results/articulation_maestro_rate.json
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
DEFAULT_BUNDLES = REPO / "model/data/evals/maestro_indep_bundles"
DEFAULT_OUT = REPO / "model/data/results/articulation_maestro_rate.json"

# GT deadband in ratio units: how far the GROUND-TRUTH statistic must sit from the GT
# corpus median before we are willing to call the performance legato / detached at all.
# Mirrors the measurer's tau; swept below so the rate is not a single-threshold
# artifact.
TAU_GT_DEFAULT = 0.163
TAU_GT_SWEEP = (0.10, 0.163, 0.25)

sys.path.insert(0, str(REPO / "model/src"))
sys.path.insert(0, str(REPO / "apps/evals"))

from claim_measurement.dynamics_supply.independent_rate import (  # noqa: E402
    aggregate,
    bootstrap_ci,
)


def gt_articulation_polarity(gt_ratio: float, median: float, tau_gt: float) -> str:
    d = gt_ratio - median
    if d > tau_gt:
        return "+"  # more legato / over-held than the corpus
    if d < -tau_gt:
        return "-"  # more detached / staccato than the corpus
    return "neutral"


def _label(pol: str) -> str:
    return {"+": "legato", "-": "detached", "neutral": "normal"}[pol]


def gt_ratio(bundle: dict) -> float | None:
    from claim_taxonomy.verifier.measurers.articulation import _duration_ioi_ratios

    ratios = _duration_ioi_ratios(bundle["gt_notes"])
    return float(np.median(ratios)) if ratios.size else None


def amt_corpus_reference(bundles: list[dict]) -> float:
    """Transkun corpus median articulation ratio: the recalibrated whole_piece
    reference.
    """
    from claim_taxonomy.verifier.measurers.articulation import _duration_ioi_ratios

    per_window = []
    for b in bundles:
        ratios = _duration_ioi_ratios(b["notes"])
        if ratios.size:
            per_window.append(float(np.median(ratios)))
    if not per_window:
        raise ValueError("no bundle yields a measurable AMT articulation ratio")
    return float(np.median(per_window))


def gt_corpus_reference(bundles: list[dict]) -> float:
    per_window = [r for r in (gt_ratio(b) for b in bundles) if r is not None]
    if not per_window:
        raise ValueError("no bundle yields a measurable GT articulation ratio")
    return float(np.median(per_window))


def score_bundle(bundle: dict, taxonomy: dict, gt_median: float, tau_gt: float) -> dict:
    from claim_taxonomy.verifier.orchestrator import verify
    from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

    truth = gt_ratio(bundle)
    if truth is None:
        raise ValueError(
            f"bundle {bundle['video_id']} has no measurable GT articulation ratio"
        )
    gt_pol = gt_articulation_polarity(truth, gt_median, tau_gt)
    claim = {
        "dimension": "articulation",
        "location": "whole_piece",
        "polarity": gt_pol,
        "proposition": f"overall articulation is {_label(gt_pol)} (GT-anchored)",
        "magnitude": None,
    }
    r = verify(claim, bundle, taxonomy, SubstrateErrorEngine(seed=42))
    return {
        "segment": bundle["video_id"],
        "gt_articulation_ratio": round(truth, 4),
        "gt_label": _label(gt_pol),
        "gt_polarity": gt_pol,
        "amt_d": r.measured_value,
        "tau": r.tau,
        "error_bar": r.error_bar,
        "verdict": r.verdict,
        "reason": r.reason_code,
        "committed": r.verdict in ("SUPPORTED", "REFUTED"),
    }


def _load_bundles(bundle_dir: Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        if "gt_notes" not in b:
            raise ValueError(f"{p} is not an articulation oracle bundle (no gt_notes)")
        out.append(b)
    if not out:
        raise SystemExit(f"no articulation oracle bundles in {bundle_dir}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.articulation_independent_rate")
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tau-gt", type=float, default=TAU_GT_DEFAULT)
    ap.add_argument(
        "--reference",
        type=float,
        default=None,
        help="force the whole_piece reference instead of recalibrating to this corpus; "
        "use it to measure the RES-001 calibration-debt sensitivity",
    )
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    bundles = _load_bundles(args.bundles)

    from claim_taxonomy.verifier.measurers import articulation as art

    old_ref = art.REFERENCE_RATIO
    art.REFERENCE_RATIO = (
        args.reference if args.reference is not None else amt_corpus_reference(bundles)
    )
    gt_median = gt_corpus_reference(bundles)

    records = [score_bundle(b, taxonomy, gt_median, args.tau_gt) for b in bundles]
    agg = aggregate(records)
    ci = bootstrap_ci([r for r in records if r["committed"]], n_boot=args.n_boot)
    sweep = {}
    for t in TAU_GT_SWEEP:
        agg_t = aggregate([score_bundle(b, taxonomy, gt_median, t) for b in bundles])
        sweep[str(t)] = {
            "rate": agg_t["faithfulness_rate"],
            "n_committed": agg_t["n_committed"],
        }

    gd_pass = (
        (agg["n_committed"] >= 30)
        and (ci["half_width"] is not None)
        and (ci["half_width"] <= 0.05)
    )
    result = {
        "gate": "G-D articulation (independent, GT-anchored, real audio)",
        "dimension": "articulation",
        "location": "whole_piece",
        "statistic_scored": (
            f"median(duration/IOI), IOI floor {art.IOI_FLOOR_SEC}s (transkun offsets)"
        ),
        "truth_signal": "ground_truth_midi_duration_ioi_ratio (INDEPENDENT)",
        "reference_ratio_used": art.REFERENCE_RATIO,
        "reference_ratio_shipped_default": old_ref,
        "reference_recalibrated": args.reference is None,
        "gt_corpus_median": round(gt_median, 4),
        "tau_measurer": taxonomy["dimensions"]["articulation"]["tolerance"][
            "provisional"
        ],
        "tau_gt": args.tau_gt,
        "n_segments": len(bundles),
        **agg,
        "ci95": {**ci, "method": "segment_bootstrap"},
        "tau_gt_sensitivity": sweep,
        "gd_pass": gd_pass,
        "note": (
            "ORACLE: truth = GT MIDI duration/IOI ratio, score = transkun-offset ratio "
            "+ frozen "
            "router. Reference recalibrated to the transkun-MAESTRO median. This "
            ""
            "is "
            "SUBSTRATE faithfulness, not teacher faithfulness -- the claim IS the GT "
            "label."
        ),
        "per_segment": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(
        "\n=== GT-anchored INDEPENDENT articulation faithfulness "
        "(real audio, oracle) ==="
    )
    print(
        f"reference {old_ref:.4f} -> {art.REFERENCE_RATIO:.4f} "
        f"(recalibrated={args.reference is None})  gt_median={gt_median:.4f}"
    )
    print(
        f"segments={len(bundles)} committed={agg['n_committed']} "
        f"(sup={agg['n_supported']} ref={agg['n_refuted']})"
    )
    if agg["faithfulness_rate"] is not None:
        print(
            f"RATE = {agg['faithfulness_rate']:.3f}  95% CI [{ci['lo']:.3f}, "
            f"{ci['hi']:.3f}]  "
            f"half={ci['half_width']:.3f}"
        )
    print(f"confusion: {agg['confusion_gt_label_x_verdict']}")
    print(f"abstention: {agg['abstention_histogram']}")
    print(f"tau_gt sweep: {sweep}")
    print(f"G-D PASS = {gd_pass}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
