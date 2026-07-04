"""FRONT 7a scorer: timing-claim supply histogram + GO/NO-GO (#101 / #67).

Reads the LLM-extracted timing claims (merged JSON array; schema = extract_prompt.md)
and computes, fully deterministically (NO LLM in the truth path):
  - subtype histogram (rush_drag / evenness / rubato / note_value / hesitation / ambiguous)
  - location histogram (whole_piece vs bar-localized)
  - polarity histogram within rush_drag
  - IN-SCOPE count for the signed onset-deviation-vs-score statistic
      := subtype == "rush_drag" AND location in {whole_piece, bar}
    (neutral rush_drag is IN scope: "well-paced" is a falsifiable |d| < tau claim.)
  - distinct performances with >=1 in-scope claim (the G-D feasibility bar is >=30)
  - abstention histogram over every out-of-scope subtype

GO/NO-GO (institutionalizing the front-5 lesson: check supply BEFORE building):
  - NO_GO   if in-scope supply is materially ~0 (like dynamics @ G-D: 0/146).
            STOP and report; the front re-plans before any measurer work.
  - GO      if in-scope supply is materially present. STRONG if >=30 distinct
            performances carry an in-scope claim (a single-piece G-D rate is feasible);
            CONDITIONAL if in-scope > 0 but < 30 perfs (build 7b, but the G-D rate needs
            a corpus extension / generate-on-bundles to clear >=30).

Run:
    uv run python model/src/claim_measurement/timing_supply/classify_supply.py \
        --claims /ABS/scratchpad/timing_claims_merged.json \
        --corpus /ABS/scratchpad/timing_supply_corpus.json \
        --out    model/data/results/timing_supply_probe.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]

IN_SCOPE_SUBTYPE = "rush_drag"
OUT_OF_SCOPE_SUBTYPES = ("evenness", "rubato", "note_value", "hesitation", "ambiguous")
MIN_INSCOPE_PERFS_STRONG = 30  # G-D pass needs >=30 committed performances
MIN_INSCOPE_SUPPLY = 10        # below this, treat as materially ~0 (NO_GO)


def _is_bar_localized(loc) -> bool:
    return isinstance(loc, dict) and "bar_start" in loc


def classify(claims: list[dict], n_corpus_docs: int, n_corpus_perfs: int) -> dict:
    subtype_hist = Counter(c.get("timing_subtype", "ambiguous") for c in claims)
    loc_hist = Counter(
        "bar_localized" if _is_bar_localized(c.get("location")) else "whole_piece"
        for c in claims
    )

    in_scope = [c for c in claims
                if c.get("timing_subtype") == IN_SCOPE_SUBTYPE]
    in_scope_bar = [c for c in in_scope if _is_bar_localized(c.get("location"))]
    rush_drag_pol = Counter(c.get("polarity", "n/a") for c in in_scope)

    in_scope_perfs = {c["recording_id"] for c in in_scope}
    any_timing_perfs = {c["recording_id"] for c in claims}

    # abstention histogram: every out-of-scope subtype, tagged with the statistic reason
    abstain = Counter()
    for c in claims:
        st = c.get("timing_subtype", "ambiguous")
        if st != IN_SCOPE_SUBTYPE:
            abstain[f"out_of_scope_statistic:{st}"] += 1

    n_in = len(in_scope)
    n_in_perfs = len(in_scope_perfs)
    if n_in < MIN_INSCOPE_SUPPLY or n_in_perfs < MIN_INSCOPE_SUPPLY:
        verdict = "NO_GO"
        verdict_detail = (
            f"in-scope supply materially ~0 (claims={n_in}, perfs={n_in_perfs} < "
            f"{MIN_INSCOPE_SUPPLY}); mirrors dynamics G-D (0/146). STOP + re-plan.")
    elif n_in_perfs >= MIN_INSCOPE_PERFS_STRONG:
        verdict = "GO"
        verdict_detail = (
            f"in-scope supply strong: {n_in} claims across {n_in_perfs} performances "
            f">= {MIN_INSCOPE_PERFS_STRONG}; a single-piece G-D timing rate is feasible.")
    else:
        verdict = "GO_CONDITIONAL"
        verdict_detail = (
            f"in-scope supply present ({n_in} claims, {n_in_perfs} perfs) but "
            f"< {MIN_INSCOPE_PERFS_STRONG} perfs; build 7b, but a G-D rate needs a "
            f"corpus extension / generate-on-bundles to clear >=30 committed perfs.")

    return {
        "front": "7a",
        "dimension": "timing",
        "statistic": "signed_mean_onset_deviation_ms_vs_score",
        "in_scope_definition": ("timing_subtype == 'rush_drag' AND location in "
                                "{whole_piece, bar}; neutral polarity is in scope"),
        "corpus": {
            "n_prose_docs": n_corpus_docs,
            "n_distinct_performances": n_corpus_perfs,
            "piece": "chopin_ballade_1",
            "note": "same 162-doc / 94-perf corpus front-5 measured dynamics supply on",
        },
        "n_timing_claims_total": len(claims),
        "n_performances_with_any_timing_claim": len(any_timing_perfs),
        "subtype_histogram": dict(subtype_hist),
        "location_histogram": dict(loc_hist),
        "in_scope": {
            "n_claims": n_in,
            "n_bar_localized": len(in_scope_bar),
            "n_whole_piece": n_in - len(in_scope_bar),
            "n_distinct_performances": n_in_perfs,
            "rush_drag_polarity_histogram": dict(rush_drag_pol),
        },
        "abstention_histogram": dict(abstain),
        "gd_feasibility_bar": MIN_INSCOPE_PERFS_STRONG,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="timing_supply.classify_supply")
    ap.add_argument("--claims", type=Path, required=True,
                    help="merged JSON array of LLM-extracted timing claims")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="the build_corpus.py JSON (for doc/perf denominators)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "model/data/results/timing_supply_probe.json")
    args = ap.parse_args(argv)

    claims = json.loads(args.claims.read_text())
    if not isinstance(claims, list):
        raise ValueError(f"{args.claims} is not a JSON array")
    corpus = json.loads(args.corpus.read_text())
    n_docs = len(corpus)
    n_perfs = len({d["recording_id"] for d in corpus})

    result = classify(claims, n_docs, n_perfs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== FRONT 7a timing claim-supply probe ===", flush=True)
    print(f"corpus: {n_docs} prose docs / {n_perfs} performances (chopin_ballade_1)",
          flush=True)
    print(f"timing claims total = {result['n_timing_claims_total']} "
          f"across {result['n_performances_with_any_timing_claim']} performances",
          flush=True)
    print(f"subtype histogram: {result['subtype_histogram']}", flush=True)
    print(f"location histogram: {result['location_histogram']}", flush=True)
    print(f"IN-SCOPE (rush_drag): {result['in_scope']['n_claims']} claims "
          f"({result['in_scope']['n_whole_piece']} whole_piece + "
          f"{result['in_scope']['n_bar_localized']} bar-localized) across "
          f"{result['in_scope']['n_distinct_performances']} performances", flush=True)
    print(f"rush_drag polarity: {result['in_scope']['rush_drag_polarity_histogram']}",
          flush=True)
    print(f"abstention histogram: {result['abstention_histogram']}", flush=True)
    print(f"\nVERDICT = {result['verdict']}: {result['verdict_detail']}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
