# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0"]
# ///
"""FRONT 8d: PAIRED aria-amt vs transkun substrate comparison for the independent
dynamics faithfulness rate (#101 / #125 / #128).

#128 swapped the production transcriber from aria-amt to Transkun. The 8b/8c dynamics
rate (0.919) was measured on aria-amt bundles, so it now describes a retired substrate.
This re-scores the SAME 168 stratified PercePiano segments under Transkun and compares the
two independent_rate.py result JSONs as a PAIRED experiment.

The design is paired because everything except the transcriber is held fixed: same segments,
same GT-MIDI truth labels (gt_polarity is GT-derived, not substrate-derived), same tau. So a
verdict flip or a rate delta is attributable to the substrate, not to sampling.

Outputs:
  - transition matrix  ARIA_verdict -> TK_verdict  (SUP / REF / ABSTAIN), over segments in both
  - exact two-sided McNemar on the committed-in-both subset (SUPPORTED vs not)
  - per-GT-polarity mean AMT velocity-d shift (tk_d - aria_d): does Transkun read systematically
    louder/softer, and is the soft stratum the weak axis (the n=3 smoke hinted +12 on soft)?

Run:
    model/.venv/bin/python compare_substrates.py \
      --aria     model/data/results/dynamics_independent_rate.json \
      --transkun model/data/results/dynamics_independent_rate_transkun.json \
      --out      model/data/results/dyn_substrate_compare.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _verdict_class(rec: dict) -> str:
    """SUPPORTED / REFUTED / ABSTAIN — the committed verdict, or ABSTAIN if not committed."""
    return rec["verdict"] if rec["committed"] else "ABSTAIN"


def join_records(aria: list[dict], transkun: list[dict]) -> list[dict]:
    """Inner-join per-segment records on `segment`. Raises if the paired design is violated
    (a segment scored under one substrate but not the other) -- loud failure over silent skew."""
    a_by = {r["segment"]: r for r in aria}
    t_by = {r["segment"]: r for r in transkun}
    only_a = sorted(set(a_by) - set(t_by))
    only_t = sorted(set(t_by) - set(a_by))
    if only_a or only_t:
        raise ValueError(
            f"paired design broken: {len(only_a)} segments only in aria "
            f"(e.g. {only_a[:3]}), {len(only_t)} only in transkun (e.g. {only_t[:3]})"
        )
    pairs = []
    for seg in sorted(a_by):
        a, t = a_by[seg], t_by[seg]
        pairs.append({
            "segment": seg,
            "gt_polarity": a["gt_polarity"],   # GT-derived; identical across substrates
            "gt_label": a["gt_label"],
            "aria_verdict": _verdict_class(a),
            "transkun_verdict": _verdict_class(t),
            "aria_d": a["amt_d"],
            "transkun_d": t["amt_d"],
        })
    return pairs


def transition_matrix(pairs: list[dict]) -> dict[str, int]:
    """ARIA_verdict -> TK_verdict counts over all paired segments."""
    m: dict[str, int] = {}
    for p in pairs:
        k = f"{p['aria_verdict']}->{p['transkun_verdict']}"
        m[k] = m.get(k, 0) + 1
    return dict(sorted(m.items()))


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: 2*sum_{i=0}^{min(b,c)} C(n,i) 0.5^n, n=b+c (capped at 1)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(pairs: list[dict]) -> dict:
    """McNemar on the committed-in-both subset, coding SUPPORTED=1 / REFUTED=0.
    b = aria SUPPORTED & transkun REFUTED; c = aria REFUTED & transkun SUPPORTED (discordant)."""
    both = [p for p in pairs
            if p["aria_verdict"] in ("SUPPORTED", "REFUTED")
            and p["transkun_verdict"] in ("SUPPORTED", "REFUTED")]
    b = sum(1 for p in both if p["aria_verdict"] == "SUPPORTED" and p["transkun_verdict"] == "REFUTED")
    c = sum(1 for p in both if p["aria_verdict"] == "REFUTED" and p["transkun_verdict"] == "SUPPORTED")
    concordant = len(both) - b - c
    return {
        "n_committed_both": len(both),
        "concordant": concordant,
        "discordant_aria_sup_tk_ref": b,
        "discordant_aria_ref_tk_sup": c,
        "exact_p_two_sided": _exact_mcnemar_p(b, c),
    }


def polarity_shift(pairs: list[dict]) -> dict[str, dict]:
    """Per GT polarity: mean AMT velocity-d under each substrate + mean paired shift (tk - aria).
    Positive shift = Transkun reads louder than aria on that stratum."""
    out: dict[str, dict] = {}
    for pol in ("+", "-", "neutral"):
        grp = [p for p in pairs if p["gt_polarity"] == pol]
        if not grp:
            continue
        aria_ds = [p["aria_d"] for p in grp]
        tk_ds = [p["transkun_d"] for p in grp]
        shifts = [t - a for a, t in zip(aria_ds, tk_ds)]
        out[pol] = {
            "label": {"+": "loud", "-": "soft", "neutral": "balanced"}[pol],
            "n": len(grp),
            "mean_aria_d": round(sum(aria_ds) / len(grp), 3),
            "mean_transkun_d": round(sum(tk_ds) / len(grp), 3),
            "mean_shift_tk_minus_aria": round(sum(shifts) / len(grp), 3),
        }
    return out


def compare(aria_result: dict, transkun_result: dict) -> dict:
    """Full paired comparison from two independent_rate.py result dicts."""
    pairs = join_records(aria_result["per_segment"], transkun_result["per_segment"])
    return {
        "n_paired_segments": len(pairs),
        "aria": {
            "rate": aria_result["faithfulness_rate"],
            "n_committed": aria_result["n_committed"],
            "ci_half_width": aria_result["ci95"]["half_width"],
            "gd_pass": aria_result["gd_pass"],
        },
        "transkun": {
            "rate": transkun_result["faithfulness_rate"],
            "n_committed": transkun_result["n_committed"],
            "ci_half_width": transkun_result["ci95"]["half_width"],
            "gd_pass": transkun_result["gd_pass"],
        },
        "rate_delta_tk_minus_aria": (
            None if aria_result["faithfulness_rate"] is None or transkun_result["faithfulness_rate"] is None
            else round(transkun_result["faithfulness_rate"] - aria_result["faithfulness_rate"], 4)
        ),
        "verdict_transition_matrix": transition_matrix(pairs),
        "mcnemar_committed_both": mcnemar(pairs),
        "polarity_velocity_shift": polarity_shift(pairs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.compare_substrates")
    ap.add_argument("--aria", type=Path, required=True)
    ap.add_argument("--transkun", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    aria = json.loads(args.aria.read_text())
    transkun = json.loads(args.transkun.read_text())
    result = compare(aria, transkun)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== PAIRED substrate comparison: aria-amt vs Transkun (dynamics, oracle) ===", flush=True)
    a, t = result["aria"], result["transkun"]
    print(f"aria     rate={a['rate']:.3f}  committed={a['n_committed']}  ci_half={a['ci_half_width']:.3f}  gd_pass={a['gd_pass']}", flush=True)
    print(f"transkun rate={t['rate']:.3f}  committed={t['n_committed']}  ci_half={t['ci_half_width']:.3f}  gd_pass={t['gd_pass']}", flush=True)
    print(f"rate delta (tk - aria) = {result['rate_delta_tk_minus_aria']:+.4f}", flush=True)
    print(f"transitions: {result['verdict_transition_matrix']}", flush=True)
    print(f"mcnemar: {result['mcnemar_committed_both']}", flush=True)
    print(f"polarity velocity shift: {result['polarity_velocity_shift']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
