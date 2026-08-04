# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""#138 step 2 -- the 37 hand features vs the frozen encoder arms, SAME FOLDS.

    uv run --no-project --script features37_compare.py [--data-root PATH]

--data-root matters from a worktree: worktrees have their own (empty) model/data,
so point it at the checkout that actually holds results/bakeoff/.

Run `run_bakeoff.py --stage features37` first; this reads the .npz arms it writes.
Standalone `# /// script` (like tk_ablation.py) only because lightgbm is not in
model/.venv and `uv run --with` from inside model/ would mutate the shared venv.

WHY THIS EXISTS. `bakeoff_cv.py` (RidgeCV + seeded composer folds) and
`tk_ablation.py` (LightGBM + GroupKFold) are two protocols; comparing their tau-c
values is the #135 cross-protocol mirage. Phase 1 needs one number it must clear,
measured on the folds it will itself be measured on. That number is
`features37|ridge` below.

Every arm here reuses bakeoff_cv.composer_disjoint_folds with the same N_FOLDS and
SEEDS, over rows whose grades and composer ids come from the same .npz files, so
the deltas are paired per piece and the bootstrap resamples both arms together.

The LightGBM arm answers a fairness objection rather than the headline question:
RidgeCV is the matched model class (what the encoders got), but #137's anchor was
LightGBM, so a linear-only feature number could understate the baseline.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from claim_measurement.difficulty.bakeoff_cv import (  # noqa: E402
    composer_disjoint_folds,
    tau_c,
)
from claim_measurement.difficulty.bakeoff_paths import resolve_paths  # noqa: E402

N_FOLDS, SEEDS, N_BOOT = 5, list(range(2026, 2031)), 2000
ALPHAS = np.logspace(-1, 5, 25)
# #137's REG_PARAMS verbatim, so the LightGBM arm is that model class on these folds.
LGBM_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.03,
                   num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.9, reg_lambda=1.0, random_state=2026,
                   n_jobs=-1, verbosity=-1)


def load_arm(emb_root: Path, arm: str, pooling: str):
    X, y, comp = [], [], []
    for path in sorted((emb_root / "emb" / arm).glob("*.npz")):
        z = np.load(path)
        X.append(z[f"emb__{pooling}"])
        y.append(int(z["grade"]))
        comp.append(int(z["composer_id"]))
    if not X:
        raise SystemExit(f"no .npz files for arm {arm!r} under {emb_root / 'emb'}")
    return np.stack(X), np.array(y), np.array(comp)


def oof_ridge(X, y, comp, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for te in composer_disjoint_folds(comp, N_FOLDS, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[tr]), y[tr])
        oof[te] = model.predict(np.nan_to_num(X[te]))
    return oof


def oof_lgbm(X, y, comp, seed):
    import lightgbm as lgb

    oof = np.full(len(y), np.nan)
    for te in composer_disjoint_folds(comp, N_FOLDS, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        oof[te] = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X[tr], y[tr]).predict(X[te])
    return oof


def paired_boot(oof_a, oof_b, y, seed=2026):
    """Bootstrap the tau-c difference over PIECES, resampling the same indices for
    both arms so the fold noise they share cancels."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        diffs[b] = (stats.kendalltau(oof_b[i], y[i], variant="c").statistic
                    - stats.kendalltau(oof_a[i], y[i], variant="c").statistic)
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return float(np.mean(diffs)), lo, hi, float(np.mean(diffs <= 0))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None)
    emb_root = resolve_paths(ap.parse_args(argv).data_root).emb_root
    Xf, y, comp = load_arm(emb_root, "features37", "raw37")
    arms_spec = [("features37|ridge", oof_ridge, Xf),
                 ("features37|lgbm", oof_lgbm, Xf)]
    for arm, pooling, label in (("moonbeam", "mean_pool", "moonbeam_mean|ridge"),
                                ("aria", "embedding", "aria_eos|ridge")):
        X, y_arm, comp_arm = load_arm(emb_root, arm, pooling)
        # Identical rows are the whole basis for calling these folds the same folds.
        if not (np.array_equal(y, y_arm) and np.array_equal(comp, comp_arm)):
            raise SystemExit(f"arm {arm!r} rows do not align with features37; the "
                             f"comparison would be unpaired")
        arms_spec.append((label, oof_ridge, X))

    print(f"n={len(y)} pieces, {len(set(comp))} composers, grades {y.min()}..{y.max()}")
    oofs = {}
    for name, fn, X in arms_spec:
        taus = [tau_c(fn(X, y, comp, s), y) for s in SEEDS]
        # Pooled OOF at seed 2026 is what the bootstrap resamples.
        oofs[name] = fn(X, y, comp, SEEDS[0])
        print(f"{name:22s} tau-c {np.mean(taus):.4f} +/- {np.std(taus):.4f}")

    print()
    for a, b in (("features37|ridge", "moonbeam_mean|ridge"),
                 ("features37|lgbm", "moonbeam_mean|ridge"),
                 ("features37|ridge", "features37|lgbm"),
                 ("aria_eos|ridge", "features37|ridge")):
        d, lo, hi, p = paired_boot(oofs[a], oofs[b], y)
        print(f"{b} - {a}: {d:+.4f} CI95[{lo:+.4f},{hi:+.4f}] P(diff<=0)={p:.3f} "
              f"{'SIG' if lo > 0 else 'noise'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
