# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0", "scipy>=1.10.0"]
# ///
"""Phase 3d -- the label-noise CEILING: how much do graders disagree on the SAME piece?

Our model self-evals against PSyllabus `ps` grades. But `ps` is reconciled from up to 11
source syllabi (ABRSM/RCM/AMEB/...), and each work's `related_entries` records the grade
EACH syllabus gave it. Those disagree -- e.g. one work is {AGME:7, NZMEB:7, Piano St:7,
AMEB:11}. The pairwise tau-c BETWEEN two syllabi over the works they both grade is the
inter-rater ceiling: no model predicting one rater's grades can beat how well another rater
predicts them. If that ceiling is ~0.75-0.85 and we sit at 0.72, we are LABEL-gated, not
feature-gated -- more features cannot cross a wall the labels themselves don't agree on.

This reads ONLY the label JSON (no MIDI, no model). Reports:
  * per-syllabus-pair tau-c (variant='c') over shared works, with shared-work counts,
  * the shared-work-weighted mean pairwise tau-c = the label-noise ceiling estimate,
  * grade-spread stats on multi-graded works (how many grades the syllabi span).

Run:  uv run --script phase3d_label_ceiling.py [--min-shared 40]
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
LABELS = PRIMARY / "model/data/raw/psyllabus/new_clean_data.json"
OUT = PRIMARY / "model/data/results/mirex_phase3d_label_ceiling.json"


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def main():
    min_shared = int(sys.argv[sys.argv.index("--min-shared") + 1]) if "--min-shared" in sys.argv else 40
    labels = json.loads(LABELS.read_text())

    # per-work grade map: {syllabus: grade}. Use related_entries (the multi-syllabus record).
    works = []           # list of {syllabus: grade} dicts, one per work
    spreads = []
    for meta in labels.values():
        re_ = meta.get("related_entries")
        if not isinstance(re_, dict):
            continue
        graded = {}
        for syl, g in re_.items():
            try:
                graded[str(syl).strip()] = int(g)
            except (TypeError, ValueError):
                continue
        if graded:
            works.append(graded)
        if len(graded) >= 2:
            gs = list(graded.values())
            spreads.append(max(gs) - min(gs))

    # collect, per ORDERED work, each syllabus's grade -> pairwise shared lists
    syllabi = sorted({s for w in works for s in w})
    pair_grades = {}     # (A,B) -> ([gA...],[gB...]) over works graded by both
    for w in works:
        present = [s for s in syllabi if s in w]
        for a, b in combinations(present, 2):
            key = (a, b)
            pair_grades.setdefault(key, ([], []))
            pair_grades[key][0].append(w[a])
            pair_grades[key][1].append(w[b])

    rows = []
    for (a, b), (ga, gb) in pair_grades.items():
        if len(ga) < min_shared:
            continue
        t = tau_c(ga, gb)
        if t is not None:
            rows.append({"a": a, "b": b, "n_shared": len(ga), "tau_c": t})
    rows.sort(key=lambda r: -r["n_shared"])

    weights = np.array([r["n_shared"] for r in rows], float)
    taus = np.array([r["tau_c"] for r in rows], float)
    weighted_mean = float(np.sum(weights * taus) / np.sum(weights)) if rows else None
    plain_mean = float(np.mean(taus)) if rows else None

    # --- split-half panel reliability: the NON-circular ceiling for an independent predictor.
    # Randomly split each multi-graded work's raters into halves A,B; tau-c(meanA, meanB) across
    # works. This is "how well one independent rater-panel predicts another" = our model's hope.
    multi = [list(w.values()) for w in works if len(w) >= 2]
    rng = np.random.default_rng(2026)
    sh_taus = []
    for _ in range(15):
        A, B = [], []
        for gs in multi:
            arr = np.asarray(gs); perm = rng.permutation(len(arr)); h = len(arr) // 2
            if h == 0:
                continue
            A.append(arr[perm[:h]].mean()); B.append(arr[perm[h:2 * h]].mean())
        t = tau_c(A, B)
        if t is not None:
            sh_taus.append(t)
    split_half = float(np.mean(sh_taus)) if sh_taus else None
    spearman_brown = (2 * split_half / (1 + split_half)) if split_half is not None else None

    spreads = np.array(spreads)
    summary = {
        "n_works_with_labels": len(works),
        "n_multigraded_works": int((spreads >= 0).sum()) if spreads.size else 0,
        "grade_spread": {
            "mean": float(spreads.mean()) if spreads.size else None,
            "median": float(np.median(spreads)) if spreads.size else None,
            "max": int(spreads.max()) if spreads.size else None,
            "frac_exact_agree": float(np.mean(spreads == 0)) if spreads.size else None,
            "frac_within1": float(np.mean(spreads <= 1)) if spreads.size else None,
        },
        "n_syllabus_pairs": len(rows),
        "min_shared": min_shared,
        "ceiling_tau_c_pairwise_weighted": weighted_mean,
        "ceiling_tau_c_pairwise_unweighted": plain_mean,
        "ceiling_tau_c_split_half": split_half,
        "ceiling_tau_c_split_half_spearman_brown": spearman_brown,
        "n_multigraded_for_splithalf": len(multi),
        "pairs": rows[:25],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("=== PHASE 3d: LABEL-NOISE CEILING (inter-syllabus tau-c) ===")
    print(f"  works with syllabus grades: {len(works)}   multi-graded (>=2 syllabi): {len(spreads)}")
    if spreads.size:
        print(f"  grade spread across syllabi: mean={spreads.mean():.2f} median={np.median(spreads):.0f} "
              f"max={spreads.max()}  exact-agree={np.mean(spreads==0):.2f}  within-1={np.mean(spreads<=1):.2f}")
    print(f"\n  top syllabus pairs by shared works (n>= {min_shared}):")
    print(f"    {'A':10s} {'B':10s} {'n':>6s} {'tau_c':>8s}")
    for r in rows[:15]:
        print(f"    {r['a']:10s} {r['b']:10s} {r['n_shared']:>6d} {r['tau_c']:>8.4f}")
    print(f"\n  LABEL-NOISE CEILING estimates (tau-c an independent model can hope to reach):")
    print(f"    pairwise inter-syllabus (weighted) = {weighted_mean:.4f}   (predict ONE arbitrary rater)")
    print(f"    split-half panel reliability       = {split_half:.4f}   (one independent panel vs another)")
    print(f"    Spearman-Brown full-panel step-up  = {spearman_brown:.4f}   (reproducibility of the consensus)")
    print(f"  => achievable band ~ {split_half:.3f}-{spearman_brown:.3f}; our model at 0.764 sits at the lower/pairwise end.")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
