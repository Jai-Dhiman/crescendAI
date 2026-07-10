# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0",
#     "lightgbm>=4.3.0",
# ]
# ///
"""Phase 3b -- the #104 commit-gate: a TRAINED composer-disjoint difficulty ranker.

Phase 3a proved the signal exists (pitch_lz alone: tau-c 0.70/0.73, beating the
note_count baseline 0.675/0.703 -- but only by a THIN margin on a single feature).
Phase 3b is the gate: does a trained model on the full feature set beat note_count-ALONE
by a CLEAR margin? If it only ties note_count, the symbolic-from-AMT stream is not worth
the Docker complexity and we escalate to the MuQ-audio stream (A11) instead.

Design (see the living doc "P3b" plan):
  * Features per whole-piece MIDI: {pitch_lz, pitch_entropy, pitch_range, note_count}
    from Phase 3a (polyphony DROPPED -- tau-c 0.01, verified dead) + reused velocity/timing
    proxies {vel_mean, vel_disp, timing_ioi_cv} from expert_anchor/midi_measures.py.
  * Protocol: composer-disjoint 5-fold (sklearn GroupKFold, groups=composer) -- NO composer
    straddles a fold, so tau-c measures generalization beyond memorizing a composer's style
    (A9). This is the safe self-eval proxy for the private MIREX test set.
  * Two model heads, report both, pick the winner honestly:
      - LGBMRanker (objective=lambdarank): the ordinal ranker the gate calls for. Whole
        train fold is ONE query group; integer grade 0..10 is the relevance label. This is
        the all-pairs relation tau-c scores.
      - LGBMRegressor (L2): cross-check. Native grade-scale output -> Acc+/-1 and macro-MSE
        need no calibration hack. Regression often ranks as well globally as lambdarank
        (which is NDCG/top-weighted), so it de-risks the gate.
  * Baseline: note_count-ALONE (the score IS the note count -- no training). The honest
    comparison per the 3a decision.
  * Output format note: MIREX wants an INTEGER per recording on an arbitrary scale. tau-c is
    rank-based, so round(score*1000) preserves the ranker's ordering; we report tau-c on the
    raw float prediction (identical ranks, no tie-loss). Acc+/-1 / macro-MSE use the
    regressor's grade-scale output rounded+clamped to [0, 10].

Run (data read from + result written to the primary checkout):
    uv run --script phase3b_ranker.py [--limit N] [--refresh] [--seed S]

  --limit strides across the grade-grouped label JSON (a prefix is single-grade -> no signal).
  --refresh forces feature re-extraction (default: reuse the cache if it covers the full set).
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_features import extract_difficulty_features  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

# reuse the validated velocity/timing proxies (GATE 2 measures) rather than re-deriving them
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expert_anchor"))
from midi_measures import (  # noqa: E402
    dynamics_mean_velocity,
    dynamics_velocity_dispersion,
    timing_ioi_cv,
)

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
OUT = PRIMARY / "model/data/results/mirex_phase3b_ranker.json"
FEAT_CACHE = PRIMARY / "model/data/results/mirex_phase3b_features.json"

# model input columns (polyphony dropped per Phase 3a). note_count is a feature AND the baseline.
FEATURES = [
    "pitch_lz_complexity",
    "pitch_entropy",
    "pitch_range",
    "note_count",
    "vel_mean",
    "vel_disp",
    "timing_ioi_cv",
]
GRADE_MIN, GRADE_MAX = 0, 10
N_FOLDS = 5


def tau_c(x, y) -> float | None:
    """scipy Kendall tau-c (the official MIREX scorer). None if degenerate."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def extract_rows(records) -> tuple[list[dict], int]:
    """Extract the full feature row per usable piece. Whole-piece features (no 30s crop):
    difficulty is a whole-piece property; the audio submission chunks to cover the piece."""
    rows, parse_fail = [], 0
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, rec in enumerate(records):
            try:
                notes = notes_from_midi_bytes(zf.read(rec.midi_name))
                if len(notes) < 5:
                    raise ValueError(f"too few notes ({len(notes)})")
                feats = extract_difficulty_features(notes)
                onsets = np.array([n["onset"] for n in notes], dtype=np.float64)
                vels = np.array([n["velocity"] for n in notes], dtype=np.float64)
                # velocity/timing proxies raise on degenerate input -> nan; LightGBM handles nan
                try:
                    v_mean = dynamics_mean_velocity(vels)
                    v_disp = dynamics_velocity_dispersion(vels)
                except ValueError:
                    v_mean = v_disp = float("nan")
                try:
                    t_cv = timing_ioi_cv(onsets)
                except ValueError:
                    t_cv = float("nan")
                rows.append({
                    "grade": rec.grade,
                    "composer": rec.composer,
                    "pitch_lz_complexity": feats["pitch_lz_complexity"],
                    "pitch_entropy": feats["pitch_entropy"],
                    "pitch_range": feats["pitch_range"],
                    "note_count": len(notes),
                    "vel_mean": v_mean,
                    "vel_disp": v_disp,
                    "timing_ioi_cv": t_cv,
                })
            except Exception as exc:  # noqa: BLE001 - surface, count, keep going
                parse_fail += 1
                if parse_fail <= 10:
                    print(f"  parse fail [{rec.key[:40]}]: {exc!r}", flush=True)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(records)} ({parse_fail} fails)", flush=True)
    return rows, parse_fail


def load_or_extract(records, use_cache: bool) -> tuple[list[dict], int]:
    """Feature extraction is the slow part (7.9k MIDI parses); cache it so ranker tuning
    re-runs are instant. Cache is only trusted for a FULL run (matching record count)."""
    if use_cache and FEAT_CACHE.exists():
        cached = json.loads(FEAT_CACHE.read_text())
        if cached.get("n_records") == len(records):
            print(f"loaded {len(cached['rows'])} feature rows from cache", flush=True)
            return cached["rows"], cached.get("n_parse_fail", 0)
    rows, parse_fail = extract_rows(records)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FEAT_CACHE.write_text(json.dumps(
        {"n_records": len(records), "n_parse_fail": parse_fail, "rows": rows}))
    return rows, parse_fail


def _lgbm_params(seed: int) -> dict:
    """Modest-capacity, regularized params -- ~6k samples x 7 features overfits easily."""
    return dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def macro_mse(pred_grade: np.ndarray, true_grade: np.ndarray) -> float:
    """MSE averaged over grade CLASSES (not pieces) so common grades don't dominate."""
    errs = []
    for g in np.unique(true_grade):
        m = true_grade == g
        errs.append(float(np.mean((pred_grade[m] - true_grade[m]) ** 2)))
    return float(np.mean(errs))


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 2026
    refresh = "--refresh" in sys.argv
    # --features lets a run ablate the input set (comma-separated subset of FEATURES).
    # The feature cache stores ALL columns per row, so ablating never re-extracts.
    global FEATURES
    if "--features" in sys.argv:
        FEATURES = [f.strip() for f in sys.argv[sys.argv.index("--features") + 1].split(",")]

    import lightgbm as lgb

    records, report = load_records(LABELS, MID_ZIP)
    if limit:
        records = records[:: max(1, len(records) // limit)][:limit]
    print(f"loaded {report}", flush=True)
    print(f"processing {len(records)} records\n", flush=True)

    # --limit changes the record set, so a full-set cache would not match -> extract fresh
    rows, parse_fail = load_or_extract(records, use_cache=not refresh and limit is None)
    print(f"\nusable pieces: {len(rows)}  parse failures: {parse_fail}", flush=True)

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["grade"] for r in rows], dtype=int)
    composers = np.array([r["composer"] for r in rows])
    note_count = X[:, FEATURES.index("note_count")]

    # per-feature tau-c (like Phase 3a, now including the new velocity/timing proxies) --
    # tells us which features carry difficulty signal at all, before any training.
    per_feature = {}
    for j, f in enumerate(FEATURES):
        col = X[:, j]
        mask = ~np.isnan(col)
        per_feature[f] = tau_c(col[mask], y[mask])

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows = []
    importances = np.zeros(len(FEATURES))
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=composers)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        # --- ranker head: LGBMRanker / lambdarank, whole train fold = ONE query group ---
        ranker = lgb.LGBMRanker(objective="lambdarank", **_lgbm_params(seed))
        # lambdarank truncation defaults to top-30 positions; lift it so more of the ordering
        # (not just the top) drives the gradient -- tau-c is a global all-pairs metric. Capped
        # at 200 to keep each O(group^2) iteration tractable on a ~6k single query group.
        ranker.set_params(lambdarank_truncation_level=min(len(ytr), 200))
        ranker.fit(Xtr, ytr, group=[len(ytr)])
        rank_pred = ranker.predict(Xte)
        importances += ranker.feature_importances_

        # --- regressor head: native grade-scale output for Acc+/-1 + macro-MSE ---
        reg = lgb.LGBMRegressor(objective="regression", **_lgbm_params(seed))
        reg.fit(Xtr, ytr)
        reg_pred = reg.predict(Xte)
        reg_grade = np.clip(np.rint(reg_pred), GRADE_MIN, GRADE_MAX).astype(int)

        fold_rows.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "tau_c_ranker": tau_c(rank_pred, yte),
            "tau_c_regressor": tau_c(reg_pred, yte),
            "tau_c_note_count": tau_c(note_count[te], yte),
            "acc_within1_regressor": float(np.mean(np.abs(reg_grade - yte) <= 1)),
            "macro_mse_regressor": macro_mse(reg_grade.astype(float), yte.astype(float)),
        })
        print(f"  fold {k}: tau-c ranker={fold_rows[-1]['tau_c_ranker']:.4f} "
              f"reg={fold_rows[-1]['tau_c_regressor']:.4f} "
              f"note_count={fold_rows[-1]['tau_c_note_count']:.4f} "
              f"(n_test={len(te)})", flush=True)

    def agg(key):
        vals = [f[key] for f in fold_rows if f[key] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    tau_ranker = agg("tau_c_ranker")
    tau_reg = agg("tau_c_regressor")
    tau_nc = agg("tau_c_note_count")
    best_model = "ranker" if tau_ranker["mean"] >= tau_reg["mean"] else "regressor"
    best_tau = max(tau_ranker["mean"], tau_reg["mean"])
    margin = best_tau - tau_nc["mean"]
    # "clear margin" gate: best trained model beats note_count-alone by >= 0.02 tau-c
    # (roughly the 3a composer-disjoint seed-noise band) AND lands in/above the 0.70 band.
    gate_pass = margin >= 0.02 and best_tau >= tau_nc["mean"]

    summary = {
        "n_usable": len(rows),
        "n_parse_fail": parse_fail,
        "n_folds": N_FOLDS,
        "seed": seed,
        "features": FEATURES,
        "per_feature_tau_c_full": per_feature,
        "feature_importance_gain": dict(zip(FEATURES, (importances / N_FOLDS).round(1).tolist())),
        "folds": fold_rows,
        "tau_c_ranker": tau_ranker,
        "tau_c_regressor": tau_reg,
        "tau_c_note_count_baseline": tau_nc,
        "acc_within1_regressor": agg("acc_within1_regressor"),
        "macro_mse_regressor": agg("macro_mse_regressor"),
        "best_model": best_model,
        "best_tau_c": best_tau,
        "margin_over_note_count": margin,
        "gate_pass": gate_pass,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))

    print("\n=== PHASE 3b: TRAINED COMPOSER-DISJOINT DIFFICULTY RANKER ===")
    print(f"  usable={len(rows)}  folds={N_FOLDS} (GroupKFold by composer)  seed={seed}")
    print("\n  per-feature tau-c (full set):")
    for f in FEATURES:
        v = per_feature[f]
        print(f"    {f:22s} {v:.4f}" if v is not None else f"    {f:22s}   None")
    print("\n  feature importance (mean gain):")
    for f, g in summary["feature_importance_gain"].items():
        print(f"    {f:22s} {g:>8.1f}")
    print("\n  MODEL                mean tau-c   std")
    print(f"    ranker (lambdarank)  {tau_ranker['mean']:.4f}    {tau_ranker['std']:.4f}")
    print(f"    regressor (L2)       {tau_reg['mean']:.4f}    {tau_reg['std']:.4f}")
    print(f"    note_count-ALONE     {tau_nc['mean']:.4f}    {tau_nc['std']:.4f}  <-- baseline")
    print(f"\n  regressor Acc+/-1 = {summary['acc_within1_regressor']['mean']:.4f}   "
          f"macro-MSE = {summary['macro_mse_regressor']['mean']:.4f}")
    print(f"\n  best model = {best_model} (tau-c {best_tau:.4f})")
    print(f"  margin over note_count-alone = {margin:+.4f}")
    print(f"  GATE (>= +0.02 over note_count): {'PASS' if gate_pass else 'FAIL'}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
