# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""Phase 5a -- AMT-robust feature selection: shrink the DEPLOYMENT gap, not the clean ceiling.

The submission number is gated by the aria-amt transcription gap (~0.09, Phase 3e), not the
clean-MIDI ceiling. Some features degrade badly under transcription (velocity, chord-span,
fine-window: clean<->amt rho < 0.65) while others survive (pitch-class/LZ/density: rho > 0.8).
If the model leans on fragile features, it pays the gap for signal that AMT corrupts.

Hypothesis: training on ONLY the transcription-robust features costs little clean tau-c but
RAISES the deployed (amt) tau-c and shrinks the clean->amt gap. This is a near-free robustness
win -- no new data, no transcription infra.

Measures, for FULL-37 vs ROBUST(rho>0.8) vs ROBUST(rho>0.75):
  * clean 5-fold tau-c   (composer-disjoint, full 7,899 -- the reference ceiling)
  * deploy tau-c         (full-clean-trained model, tested on the 43-piece aria-amt features)
  * on-subset clean tau-c + gap (clean43 - amt43) -- the pure transcription cost on that set

Run from model/:  uv run python src/claim_measurement/difficulty/phase5a_robust_features.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features, load_or_extract  # noqa: E402
from phase3e_amt_gap import _stitch_amt_notes  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
AMT_MANIFEST = PRIMARY / "model/data/results/amt_gap/manifest.json"
PHASE3E = PRIMARY / "model/data/results/mirex_phase3e_amt_gap.json"
OUT = PRIMARY / "model/data/results/mirex_phase5a_robust_features.json"

N_FOLDS = 5


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _lgbm(seed=2026):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                             num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, reg_lambda=1.0, random_state=seed,
                             n_jobs=-1, verbosity=-1)


def clean_cv_tau(X, y, groups, cols, seeds=(2026, 2027, 2028)):
    """Composer-disjoint 5-fold LGBM tau-c on the given feature columns, averaged over seeds."""
    from sklearn.model_selection import GroupKFold
    Xs = X[:, cols]
    taus = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=N_FOLDS)
        fold_taus = []
        for tr, te in gkf.split(Xs, y, groups=groups):
            reg = _lgbm(seed).fit(Xs[tr], y[tr])
            t = tau_c(reg.predict(Xs[te]), y[te])
            if t is not None:
                fold_taus.append(t)
        taus.append(np.mean(fold_taus))
    return float(np.mean(taus)), float(np.std(taus))


def main():
    # robustness rho from Phase 3e
    pf = json.loads(PHASE3E.read_text())["summary"]["per_feature"]
    rho = {f: v["clean_amt_spearman"] for f, v in pf.items() if v["clean_amt_spearman"] is not None}

    # full clean feature matrix (reuse the phase3c cache: 7,899 rows x 37 feats + grade/composer)
    records, _ = load_records(LABELS, MID_ZIP)
    rows, _ = load_or_extract(records, use_cache=True)
    feats = [k for k in rows[0] if k not in ("grade", "composer")]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["grade"] for r in rows], int)
    groups = np.array([r["composer"] for r in rows])

    # 43-piece amt + clean features (same 37 cols), for the deployment test
    manifest = json.loads(AMT_MANIFEST.read_text())
    sub_clean, sub_amt, sub_y = [], [], []
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            amt_notes = _stitch_amt_notes(m["seg_id"])
            if not amt_notes or len(amt_notes) < 5:
                continue
            clean_notes = notes_from_midi_bytes(zf.read(m["midi_name"]))
            if len(clean_notes) < 5:
                continue
            cf, af = candidate_features(clean_notes), candidate_features(amt_notes)
            sub_clean.append([cf[f] for f in feats])
            sub_amt.append([af[f] for f in feats])
            sub_y.append(m["grade"])
    sub_clean = np.array(sub_clean, float)
    sub_amt = np.array(sub_amt, float)
    sub_y = np.array(sub_y, int)
    print(f"full clean rows: {len(rows)}   deployment subset (clean+amt): {len(sub_y)}\n", flush=True)

    subsets = {
        "FULL_37": feats,
        "ROBUST_rho080": [f for f in feats if rho.get(f, 0) > 0.80],
        "ROBUST_rho075": [f for f in feats if rho.get(f, 0) > 0.75],
    }

    results = {}
    for name, cols_names in subsets.items():
        cols = [feats.index(f) for f in cols_names]
        clean_mean, clean_std = clean_cv_tau(X, y, groups, cols)
        # deployment: train on FULL clean set (these cols), test on the 43 clean vs amt
        reg = _lgbm().fit(X[:, cols], y)
        tau_clean43 = tau_c(reg.predict(sub_clean[:, cols]), sub_y)
        tau_amt43 = tau_c(reg.predict(sub_amt[:, cols]), sub_y)
        gap = (tau_clean43 - tau_amt43) if (tau_clean43 and tau_amt43) else None
        results[name] = {"n_features": len(cols), "features": cols_names,
                         "clean_5fold_tau_c": {"mean": clean_mean, "std": clean_std},
                         "deploy_clean43_tau_c": tau_clean43, "deploy_amt43_tau_c": tau_amt43,
                         "clean_amt_gap": gap}
        print(f"[{name}] ({len(cols)} feats)")
        print(f"    clean 5-fold tau-c   = {clean_mean:.4f} +/- {clean_std:.4f}")
        print(f"    deploy clean43 tau-c = {tau_clean43}")
        print(f"    deploy AMT43   tau-c = {tau_amt43}   <-- the deployed number")
        print(f"    clean->amt gap       = {gap}\n")

    # verdict: which subset gives the best deployed (amt43) tau-c?
    best = max(results, key=lambda k: results[k]["deploy_amt43_tau_c"] or -1)
    full_amt = results["FULL_37"]["deploy_amt43_tau_c"]
    best_amt = results[best]["deploy_amt43_tau_c"]
    verdict = {
        "best_subset_by_deploy_amt": best,
        "deploy_amt_full_37": full_amt,
        "deploy_amt_best": best_amt,
        "deploy_amt_improvement": (best_amt - full_amt) if (best_amt and full_amt) else None,
        "robust_helps_deployment": bool(best != "FULL_37" and best_amt and full_amt and best_amt > full_amt),
    }
    OUT.write_text(json.dumps({"results": results, "verdict": verdict}, indent=2, default=float))
    print("=== VERDICT ===")
    print(f"  best subset by deployed AMT tau-c: {best} ({best_amt})")
    print(f"  vs FULL_37 AMT tau-c: {full_amt}   improvement: {verdict['deploy_amt_improvement']}")
    print(f"  robust feature selection helps deployment: {verdict['robust_helps_deployment']}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
