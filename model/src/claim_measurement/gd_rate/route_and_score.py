"""G-D scoring harness: route extracted dynamics claims -> verify -> rate + CI + histogram.

The FIRST measured per-dimension faithfulness rate (#101 G-D / #67). Dynamics-only,
whole_piece (pedaling FAILS G-B on the AMT substrate -- front-4 -- so no pedaling rate).

Pipeline (truth label is fully deterministic; no LLM):
  1. Load LLM-extracted dynamics claims (gd_claims merged JSON) + the AMT bundles.
  2. Route each claim:
       - dynamics_subtype != "level"  -> abstain bucket `out_of_scope_statistic`
         (the G-B-validated statistic measures loudness LEVEL, not contrast/range/shape;
         a contrast claim is a category mismatch for a level measurement, NOT a verdict).
       - no bundle for recording_id   -> bucket `no_bundle` (clip not transcribed).
       - else verify(claim, bundle, taxonomy) with the FROZEN route_verdict (tau 6.5).
  3. Rate = SUPPORTED / (SUPPORTED + REFUTED) among committed level@whole_piece claims.
  4. Bootstrap 95% CI CLUSTERED by performance (recording_id): multiple prose docs about
     one performance are not independent, so resample performances, not claims.
  5. Abstention histogram over every non-committed outcome.

G-D PASS iff: distinct committed performances >= 30 AND bootstrap CI half-width <= 0.05.

Run (claim_taxonomy importable from apps/evals):
    cd apps/evals && uv run --extra all python \
      /ABS/.../model/src/claim_measurement/gd_rate/route_and_score.py \
        --claims  /ABS/.../scratchpad/gd_claims_merged.json \
        --bundles /ABS/.../model/data/evals/gd_bundles/chopin_ballade_1 \
        --out     /ABS/.../model/data/results/gd_dynamics_rate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]
sys.path.insert(0, str(REPO / "apps/evals"))

from claim_taxonomy.verifier.orchestrator import verify  # noqa: E402
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine  # noqa: E402

TAXONOMY_PATH = REPO / "apps/evals/claim_taxonomy/claim_taxonomy.json"

# Pre-verify routing buckets (G-D specific; complement the router's reason_codes).
PREFILTER_OUT_OF_SCOPE_STATISTIC = "out_of_scope_statistic"  # contrast/ambiguous subtype
PREFILTER_NO_BUNDLE = "no_bundle"  # clip not transcribed / failed


def _load_claims(path: Path) -> list[dict]:
    claims = json.loads(path.read_text())
    if not isinstance(claims, list):
        raise ValueError(f"{path} is not a JSON array")
    return claims


def _load_bundles(bundle_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        out[b["video_id"]] = b
    return out


def _route_one(claim: dict, bundles: dict[str, dict], taxonomy: dict,
              scope_map: dict[tuple[str, str], str]) -> dict:
    """Return a routed record: {recording_id, outcome, reason, verdict, d, ...}.

    Routing (pre-verify), in order:
      - subtype != "level" (contrast/ambiguous): out_of_scope_statistic -- the
        G-B-validated statistic measures whole-piece mean LEVEL, not contrast/range.
      - subtype == "level" but level_scope != "overall_loudness": still out of scope.
        The whole-piece MEAN cannot adjudicate register-specific ("make the pp softer",
        "push the forte") or non-falsifiable ("how soft can you go?") level claims --
        a high overall mean is compatible with insufficiently-soft pp passages. Only a
        falsifiable WHOLE-PIECE-OVERALL loudness claim is in scope.
      - else: verify() against the bundle's mean velocity.
    """
    rid = claim["recording_id"]
    subtype = claim.get("dynamics_subtype", "ambiguous")
    scope = scope_map.get((rid, claim.get("proposition", "")))
    base = {
        "recording_id": rid,
        "run_id": claim.get("run_id"),
        "proposition": claim.get("proposition"),
        "subtype": subtype,
        "level_scope": scope,
        "polarity": claim.get("polarity"),
        "location": claim.get("location"),
    }
    if subtype != "level":
        return {**base, "outcome": "ABSTAIN", "reason": PREFILTER_OUT_OF_SCOPE_STATISTIC,
                "reason_detail": subtype, "verdict": None, "d": None}
    if scope_map and scope != "overall_loudness":
        return {**base, "outcome": "ABSTAIN", "reason": PREFILTER_OUT_OF_SCOPE_STATISTIC,
                "reason_detail": scope or "level_unclassified", "verdict": None, "d": None}
    bundle = bundles.get(rid)
    if bundle is None:
        return {**base, "outcome": "ABSTAIN", "reason": PREFILTER_NO_BUNDLE,
                "verdict": None, "d": None}

    verify_claim = {
        "dimension": "dynamics",
        "location": claim.get("location", "whole_piece"),
        "polarity": claim.get("polarity"),
        "proposition": claim.get("proposition"),
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    r = verify(verify_claim, bundle, taxonomy, engine)
    if r.verdict == "SUPPORTED":
        outcome, reason = "COMMITTED", "supported"
    elif r.verdict == "REFUTED":
        outcome, reason = "COMMITTED", "refuted"
    else:
        outcome, reason = "ABSTAIN", r.reason_code
    return {**base, "outcome": outcome, "reason": reason,
            "verdict": r.verdict, "d": r.measured_value, "tau": r.tau,
            "error_bar": r.error_bar, "event_count": r.event_count}


def _cluster_bootstrap_ci(committed: list[dict], n_boot: int = 5000,
                         seed: int = 12345) -> tuple[float, float, float]:
    """95% percentile CI on the SUPPORTED-rate, resampling PERFORMANCES (clusters)."""
    by_perf: dict[str, list[int]] = {}
    for c in committed:
        by_perf.setdefault(c["recording_id"], []).append(1 if c["verdict"] == "SUPPORTED" else 0)
    perfs = list(by_perf.keys())
    rng = np.random.default_rng(seed)
    rates = []
    n = len(perfs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        num = den = 0
        for j in idx:
            outs = by_perf[perfs[j]]
            num += sum(outs)
            den += len(outs)
        if den:
            rates.append(num / den)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi), float((hi - lo) / 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gd_rate.route_and_score")
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--level-scope", type=Path, default=None,
                   help="JSON from the level-scope classifier; only overall_loudness "
                        "level claims are in scope for the whole-piece-mean statistic")
    ap.add_argument("--out", type=Path, default=REPO / "model/data/results/gd_dynamics_rate.json")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    claims = _load_claims(args.claims)
    bundles = _load_bundles(args.bundles)
    scope_map: dict[tuple[str, str], str] = {}
    if args.level_scope is not None:
        for r in json.loads(args.level_scope.read_text()):
            scope_map[(r["recording_id"], r["proposition"])] = r["level_scope"]
    print(f"{len(claims)} dynamics claims; {len(bundles)} bundles; "
         f"{len(scope_map)} level-scope labels", flush=True)

    routed = [_route_one(c, bundles, taxonomy, scope_map) for c in claims]
    committed = [r for r in routed if r["outcome"] == "COMMITTED"]
    supported = [r for r in committed if r["verdict"] == "SUPPORTED"]

    # abstention histogram (every non-committed reason); out_of_scope_statistic is
    # broken out by detail (contrast / register_specific / non_falsifiable / ...).
    hist: dict[str, int] = {}
    for r in routed:
        if r["outcome"] == "ABSTAIN":
            key = r["reason"]
            if r.get("reason_detail"):
                key = f"{r['reason']}:{r['reason_detail']}"
            hist[key] = hist.get(key, 0) + 1

    n_committed = len(committed)
    n_perf_committed = len({r["recording_id"] for r in committed})
    # None (JSON null), not NaN, when the rate is unmeasurable (no committed claims).
    rate = (len(supported) / n_committed) if n_committed else None

    if n_committed:
        lo, hi, half = _cluster_bootstrap_ci(committed, n_boot=args.n_boot)
    else:
        lo = hi = half = None

    gd_pass = (n_perf_committed >= 30) and (half is not None) and (half <= 0.05)

    # polarity breakdown among committed (faithfulness by claim direction)
    pol_break: dict[str, dict[str, int]] = {}
    for r in committed:
        pb = pol_break.setdefault(r["polarity"], {"supported": 0, "refuted": 0})
        pb["supported" if r["verdict"] == "SUPPORTED" else "refuted"] += 1

    # claim-yield summary (the headline of this front): what does the generator assert?
    n_perf_any = len({c["recording_id"] for c in claims})
    subtype_counts: dict[str, int] = {}
    for c in claims:
        st = c.get("dynamics_subtype", "ambiguous")
        subtype_counts[st] = subtype_counts.get(st, 0) + 1
    scope_counts: dict[str, int] = {}
    for c in claims:
        if c.get("dynamics_subtype") == "level":
            sc = scope_map.get((c["recording_id"], c.get("proposition", "")), "unclassified")
            scope_counts[sc] = scope_counts.get(sc, 0) + 1
    n_localized = sum(1 for c in claims if c.get("location") != "whole_piece")

    result = {
        "gate": "G-D",
        "dimension": "dynamics",
        "location": "whole_piece",
        "statistic": "mean_amt_note_velocity",
        "tau": 6.5,
        "yield": {
            "n_performances_with_dynamics_claim": n_perf_any,
            "subtype_counts": subtype_counts,
            "level_scope_counts": scope_counts,
            "n_bar_localized": n_localized,
            "note": ("in-scope for the whole-piece mean-velocity statistic = "
                     "level AND overall_loudness; contrast/register/non-falsifiable abstain"),
        },
        "n_claims_total": len(claims),
        "n_committed": n_committed,
        "n_supported": len(supported),
        "n_refuted": n_committed - len(supported),
        "n_distinct_performances_committed": n_perf_committed,
        "faithfulness_rate": rate,
        "ci95": {"lo": lo, "hi": hi, "half_width": half, "method": "cluster_bootstrap_by_performance",
                 "n_boot": args.n_boot},
        "abstention_histogram": hist,
        "polarity_breakdown": pol_break,
        "gd_pass": gd_pass,
        "gd_criteria": {"min_performances": 30, "max_ci_half_width": 0.05},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== G-D dynamics @whole_piece faithfulness ===", flush=True)
    print(f"claims={len(claims)}  committed={n_committed} "
         f"(supported={len(supported)} refuted={n_committed-len(supported)})", flush=True)
    print(f"distinct committed performances = {n_perf_committed}", flush=True)
    if rate is None:
        print("RATE = UNMEASURABLE (0 committed claims)", flush=True)
    else:
        print(f"RATE = {rate:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  half-width={half:.3f}", flush=True)
    print(f"polarity breakdown (committed): {pol_break}", flush=True)
    print(f"abstention histogram: {hist}", flush=True)
    print(f"G-D PASS = {gd_pass}  (need >=30 perf AND half-width<=0.05)", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
