# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""Phase 5c -- consensus / multi-syllabus label modeling: is 0.764 a LABEL wall we can move?

PSyllabus `ps` is a per-piece consensus grade, but 32% of pieces (2,501) carry >=2 syllabus grades
(`related_entries`), and among those 43% disagree by >=2 grades -- real label noise. Phase 3d put the
consensus reliability at ~0.857 (Spearman-Brown), well above 0.764, so there may be headroom the
single-`ps` target leaves on the table.

Three experiments (composer-disjoint 5-fold LGBM on the 37 features throughout):
  1. RELIABILITY WEIGHTING (all pieces, eval vs ps): weight each piece by how many syllabi grade it
     (more syllabi = more reliable label). Submission-usable. Gate: beats unweighted 0.764.
  2. DENOISED TARGET (all pieces, eval vs ps): train against mean(related_entries) instead of ps
     (falls back to ps for single-syllabus). Submission-usable. Gate: beats ps-target.
  3. LABEL-NOISE CEILING (multi-syllabus subset only): train+eval vs ps  vs  train+eval vs
     mean(related). If the mean-target tau-c >> ps-target, the limiter is label noise, not features --
     a diagnostic (MIREX gives no syllabus metadata at test, so this is a ceiling probe, not a lever).

Run from model/:  uv run --script src/claim_measurement/difficulty/phase5c_consensus_labels.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
FEAT_CACHE = PRIMARY / "model/data/results/mirex_phase5c_features.json"
OUT = PRIMARY / "model/data/results/mirex_phase5c_consensus.json"

N_FOLDS = 5


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean_related(meta) -> tuple[float | None, int]:
    """mean of the numeric related_entries grades, and the count of syllabi."""
    re = meta.get("related_entries")
    if not isinstance(re, dict):
        return None, 0
    vals = [v for v in (_num(x) for x in re.values()) if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def build_table() -> list[dict]:
    if FEAT_CACHE.exists():
        cached = json.loads(FEAT_CACHE.read_text())
        print(f"loaded {len(cached)} keyed feature rows from cache", flush=True)
        return cached
    labels = json.loads(LABELS.read_text())
    records, report = load_records(LABELS, MID_ZIP)
    print(f"loaded {report}; extracting keyed features...", flush=True)
    rows, fail = [], 0
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, rec in enumerate(records):
            try:
                notes = notes_from_midi_bytes(zf.read(rec.midi_name))
                if len(notes) < 5:
                    raise ValueError("too few notes")
                feats = candidate_features(notes)
                mean_rel, n_syl = _mean_related(labels.get(rec.key, {}))
                rows.append({"key": rec.key, "ps": rec.grade, "composer": rec.composer,
                             "mean_related": mean_rel if mean_rel is not None else float(rec.grade),
                             "n_syllabi": max(n_syl, 1), **feats})
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if fail <= 5:
                    print(f"  parse fail [{rec.key[:40]}]: {exc!r}", flush=True)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(records)} ({fail} fails)", flush=True)
    FEAT_CACHE.write_text(json.dumps(rows))
    print(f"extracted {len(rows)} rows ({fail} fails); cached", flush=True)
    return rows


def _lgbm(seed=2026):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.03,
                             num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, reg_lambda=1.0, random_state=seed,
                             n_jobs=-1, verbosity=-1)


def cv_tau(X, groups, y_train, y_eval, sample_weight=None, seeds=(2026, 2027, 2028)):
    """Composer-disjoint 5-fold: fit on y_train (optionally weighted), score tau-c vs y_eval."""
    from sklearn.model_selection import GroupKFold
    taus = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=N_FOLDS)
        fold = []
        for tr, te in gkf.split(X, y_train, groups=groups):
            reg = _lgbm(seed)
            sw = None if sample_weight is None else sample_weight[tr]
            reg.fit(X[tr], y_train[tr], sample_weight=sw)
            t = tau_c(reg.predict(X[te]), y_eval[te])
            if t is not None:
                fold.append(t)
        taus.append(np.mean(fold))
    return float(np.mean(taus)), float(np.std(taus))


def main():
    rows = build_table()
    feats = [k for k in rows[0] if k not in ("key", "ps", "composer", "mean_related", "n_syllabi")]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    ps = np.array([r["ps"] for r in rows], float)
    mean_rel = np.array([r["mean_related"] for r in rows], float)
    n_syl = np.array([r["n_syllabi"] for r in rows], float)
    groups = np.array([r["composer"] for r in rows])
    multi = n_syl >= 2
    print(f"rows: {len(rows)}   multi-syllabus (>=2): {int(multi.sum())}\n", flush=True)

    res = {}

    # 1. reliability weighting (all pieces, eval vs ps)
    res["baseline_ps_unweighted"] = dict(zip(("mean", "std"), cv_tau(X, groups, ps, ps)))
    res["weight_n_syllabi"] = dict(zip(("mean", "std"),
                                       cv_tau(X, groups, ps, ps, sample_weight=n_syl)))
    res["weight_sqrt_n_syllabi"] = dict(zip(("mean", "std"),
                                            cv_tau(X, groups, ps, ps, sample_weight=np.sqrt(n_syl))))

    # 2. denoised training target (all pieces, eval vs ps)
    res["train_mean_related_eval_ps"] = dict(zip(("mean", "std"),
                                                 cv_tau(X, groups, mean_rel, ps)))

    # 3. label-noise ceiling on the multi-syllabus subset
    Xm, gm, psm, mrm = X[multi], groups[multi], ps[multi], mean_rel[multi]
    res["multi_train_eval_ps"] = dict(zip(("mean", "std"), cv_tau(Xm, gm, psm, psm)))
    res["multi_train_eval_mean_related"] = dict(zip(("mean", "std"), cv_tau(Xm, gm, mrm, mrm)))

    base = res["baseline_ps_unweighted"]["mean"]
    submission_variants = ["weight_n_syllabi", "weight_sqrt_n_syllabi", "train_mean_related_eval_ps"]
    best_sub = max(submission_variants, key=lambda k: res[k]["mean"])
    verdict = {
        "baseline_ps_0764": base,
        "best_submission_variant": best_sub,
        "best_submission_tau_c": res[best_sub]["mean"],
        "submission_improvement": res[best_sub]["mean"] - base,
        "label_noise_ceiling_lift": (res["multi_train_eval_mean_related"]["mean"]
                                     - res["multi_train_eval_ps"]["mean"]),
    }
    OUT.write_text(json.dumps({"results": res, "verdict": verdict}, indent=2, default=float))

    print("=== PHASE 5c: CONSENSUS / MULTI-SYLLABUS LABEL MODELING ===")
    for k, v in res.items():
        print(f"  {k:34s} {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"\n  baseline (ps, unweighted)          = {base:.4f}")
    print(f"  best submission-usable variant     = {best_sub} ({res[best_sub]['mean']:.4f}, "
          f"{verdict['submission_improvement']:+.4f})")
    print(f"  label-noise ceiling lift (multi, mean_related - ps eval) = "
          f"{verdict['label_noise_ceiling_lift']:+.4f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
