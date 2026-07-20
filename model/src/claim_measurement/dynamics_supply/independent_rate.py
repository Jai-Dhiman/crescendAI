"""FRONT 8b scorer: GT-MIDI-anchored INDEPENDENT dynamics faithfulness rate (#101 / #67).

Breaks the FRONT-8 signal-fidelity circularity. Front 8's rate cued the teacher from AMT
mean velocity and scored with AMT mean velocity (same statistic -> circular, rate=1.000
by construction). Here the CLAIM's polarity is fixed by GROUND-TRUTH MIDI velocity and the
SCORE is the production AMT-velocity measurer + frozen router -- two INDEPENDENT measurements
of the same performance. The rate answers the real question: *when ground truth says the
performance is loud / soft / balanced overall, does the deployed AMT verifier agree?*

This is the ORACLE (no-LLM) variant: the claim polarity IS the GT label (loud->+, soft->-,
balanced->neutral), so the rate isolates the verifier's SUBSTRATE faithfulness (AMT-vs-GT at
the tau decision boundary) with zero teacher/extractor noise. Truth-label purity holds: the
verdict comes only from the deterministic measurer + frozen route_verdict; GT MIDI velocity is
a non-LLM signal.

Reads FRONT-8b bundles (render_percepiano_bundles.py) that carry AMT `notes` + `gt_mean_velocity`
+ `gt_corpus_median`. tau_gt (the GT-label deadband) is swept to show the rate is not an
artifact of one threshold.

Run (claim_taxonomy importable from apps/evals):
    cd apps/evals && uv run --extra all python \
      .../dynamics_supply/independent_rate.py \
        --bundles .../model/data/evals/percepiano_indep_bundles \
        --out     .../model/data/results/dynamics_independent_rate.json
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
TAU_GT_DEFAULT = 6.5
TAU_GT_SWEEP = (4.0, 6.5, 9.0)


def gt_polarity(gt_vel: float, median: float, tau_gt: float) -> str:
    """The GROUND-TRUTH loudness direction vs the GT corpus median, in the same trichotomy
    the AMT router uses. This is the independent truth label (not the scored statistic)."""
    d = gt_vel - median
    if d > tau_gt:
        return "+"
    if d < -tau_gt:
        return "-"
    return "neutral"


def _label(pol: str) -> str:
    return {"+": "loud", "-": "soft", "neutral": "balanced"}[pol]


def score_bundle(bundle: dict, taxonomy: dict, tau_gt: float) -> dict:
    """Run the real verifier on the GT-anchored oracle claim for one segment."""
    # Lazy import: keeps the pure aggregation logic (and its tests) free of the verifier deps.
    sys.path.insert(0, str(REPO / "apps/evals"))
    from claim_taxonomy.verifier.orchestrator import verify
    from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

    gt_pol = gt_polarity(bundle["gt_mean_velocity"], bundle["gt_corpus_median"], tau_gt)
    claim = {
        "dimension": "dynamics",
        "location": "whole_piece",
        "polarity": gt_pol,
        "proposition": f"overall loudness is {_label(gt_pol)} (ground-truth-anchored)",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    r = verify(claim, bundle, taxonomy, engine)
    committed = r.verdict in ("SUPPORTED", "REFUTED")
    return {
        "segment": bundle["video_id"],
        "gt_mean_velocity": bundle["gt_mean_velocity"],
        "gt_label": _label(gt_pol),
        "gt_polarity": gt_pol,
        "amt_d": r.measured_value,
        "tau": r.tau,
        "error_bar": r.error_bar,
        "verdict": r.verdict,
        "reason": r.reason_code,
        "committed": committed,
    }


def aggregate(records: list[dict]) -> dict:
    """Independent faithfulness rate + GT-label x verdict confusion + abstention histogram."""
    committed = [r for r in records if r["committed"]]
    supported = [r for r in committed if r["verdict"] == "SUPPORTED"]
    rate = (len(supported) / len(committed)) if committed else None

    confusion: dict[str, int] = {}
    for r in records:
        v = r["verdict"] if r["committed"] else f"ABSTAIN:{r['reason']}"
        key = f"{r['gt_label']}->{v}"
        confusion[key] = confusion.get(key, 0) + 1

    hist: dict[str, int] = {}
    for r in records:
        if not r["committed"]:
            hist[r["reason"]] = hist.get(r["reason"], 0) + 1

    by_pol: dict[str, dict[str, int]] = {}
    for r in committed:
        b = by_pol.setdefault(r["gt_polarity"], {"supported": 0, "refuted": 0})
        b["supported" if r["verdict"] == "SUPPORTED" else "refuted"] += 1

    return {
        "n_records": len(records),
        "n_committed": len(committed),
        "n_supported": len(supported),
        "n_refuted": len(committed) - len(supported),
        "faithfulness_rate": rate,
        "confusion_gt_label_x_verdict": confusion,
        "abstention_histogram": hist,
        "polarity_breakdown": by_pol,
    }


def bootstrap_ci(committed: list[dict], n_boot: int = 5000, seed: int = 12345) -> dict:
    """Percentile CI on the SUPPORTED-rate, resampling SEGMENTS (each is one performance)."""
    if not committed:
        return {"lo": None, "hi": None, "half_width": None, "n_boot": n_boot}
    outs = np.array([1 if r["verdict"] == "SUPPORTED" else 0 for r in committed])
    rng = np.random.default_rng(seed)
    n = len(outs)
    rates = [outs[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "half_width": float((hi - lo) / 2), "n_boot": n_boot}


def _load_bundles(bundle_dir: Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        if "gt_mean_velocity" not in b:
            raise ValueError(f"{p} is not a FRONT-8b bundle (no gt_mean_velocity)")
        out.append(b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.independent_rate")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO / "model/data/results/dynamics_independent_rate.json")
    ap.add_argument("--tau-gt", type=float, default=TAU_GT_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    bundles = _load_bundles(args.bundles)
    if not bundles:
        raise SystemExit(f"no FRONT-8b bundles in {args.bundles}")
    gt_median = bundles[0]["gt_corpus_median"]

    records = [score_bundle(b, taxonomy, args.tau_gt) for b in bundles]
    agg = aggregate(records)
    committed = [r for r in records if r["committed"]]
    ci = bootstrap_ci(committed, n_boot=args.n_boot)

    # tau_gt sensitivity: rate + committed count under alternative GT deadbands (re-labels
    # only; the AMT verdict per segment is fixed, so this isolates threshold-sensitivity).
    sweep = {}
    for t in TAU_GT_SWEEP:
        recs_t = [score_bundle(b, taxonomy, t) for b in bundles]
        agg_t = aggregate(recs_t)
        sweep[str(t)] = {"rate": agg_t["faithfulness_rate"],
                         "n_committed": agg_t["n_committed"],
                         "n_supported": agg_t["n_supported"]}

    gd_pass = (agg["n_committed"] >= 30) and (ci["half_width"] is not None) and (ci["half_width"] <= 0.05)
    result = {
        "gate": "G-D (independent, GT-anchored)",
        "dimension": "dynamics",
        "location": "whole_piece",
        "statistic_scored": "mean_amt_note_velocity",
        "truth_signal": "ground_truth_midi_mean_velocity (INDEPENDENT of the scored statistic)",
        "tau_amt": 6.5,
        "tau_gt": args.tau_gt,
        "gt_corpus_median": gt_median,
        "n_segments": len(bundles),
        **agg,
        "ci95": {**ci, "method": "segment_bootstrap"},
        "tau_gt_sensitivity": sweep,
        "gd_pass": gd_pass,
        "gd_criteria": {"min_committed": 30, "max_ci_half_width": 0.05},
        "note": ("ORACLE rate: claim polarity = GT label, so this isolates AMT-substrate "
                 "faithfulness vs ground truth (no teacher/extractor). Circularity is broken: "
                 "truth = GT MIDI velocity, score = AMT velocity."),
        "per_segment": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== GT-anchored INDEPENDENT dynamics faithfulness (oracle) ===", flush=True)
    print(f"segments={len(bundles)}  committed={agg['n_committed']} "
          f"(supported={agg['n_supported']} refuted={agg['n_refuted']})", flush=True)
    if agg["faithfulness_rate"] is None:
        print("RATE = UNMEASURABLE (0 committed)", flush=True)
    else:
        print(f"RATE = {agg['faithfulness_rate']:.3f}  95% CI "
              f"[{ci['lo']:.3f}, {ci['hi']:.3f}]  half-width={ci['half_width']:.3f}", flush=True)
    print(f"confusion (GT label -> verdict): {agg['confusion_gt_label_x_verdict']}", flush=True)
    print(f"abstention: {agg['abstention_histogram']}", flush=True)
    print(f"tau_gt sensitivity: {sweep}", flush=True)
    print(f"G-D PASS (independent) = {gd_pass}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
