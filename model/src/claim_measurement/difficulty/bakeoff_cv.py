"""Composer-disjoint CV + Kendall tau-c, ported from the unmerged
issue-104-mirex-difficulty branch's phase5b_aria_probe.py (commit 7976b5e6).

Ported, not imported cross-branch: worktrees are separate checkouts and this
file does not exist on issue-138-encoder-finetune or on main. The RidgeCV
pipeline, alpha grid, and tau-c convention are unchanged from the original;
the private `_folds` closure inside the original `_oof_tau` is promoted to
the public `composer_disjoint_folds` (added in Task 3) so it is independently
testable per the #138 Phase 0 design's TDD targets.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def tau_c(x, y) -> float | None:
    """Kendall tau-c, nan-safe. None (never 0.0) when the input cannot
    support a rank correlation -- fewer than 3 points, or either side
    constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def composer_disjoint_folds(composers: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Split row indices into n_folds folds such that no composer's rows
    straddle two folds. Greedy bin-packing: composers are shuffled
    deterministically by seed, then each composer's whole row group is
    assigned to the fold with the fewest rows so far."""
    composers = np.asarray(composers)
    uniq = sorted(set(composers))
    sizes = {c: int(np.sum(composers == c)) for c in uniq}
    order = np.random.default_rng(seed).permutation(len(uniq))
    counts = [0] * n_folds
    fold_of = {}
    for idx in order:
        c = uniq[idx]
        f = int(np.argmin(counts))
        fold_of[c] = f
        counts[f] += sizes[c]
    return [np.array([i for i, c in enumerate(composers) if fold_of[c] == f])
            for f in range(n_folds)]


def oof_tau_ridge(X: np.ndarray, y: np.ndarray, composers: np.ndarray,
                   n_folds: int, seeds: list[int]) -> dict:
    """Composer-disjoint grouped n_folds-fold RidgeCV, repeated per seed
    (each seed re-draws the fold assignment). Returns mean/std tau-c over
    seeds where a fold produced a valid tau-c, and n_seeds actually used."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-1, 5, 25)
    taus = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for te in composer_disjoint_folds(composers, n_folds, seed):
            tr = np.setdiff1d(np.arange(len(y)), te)
            if len(tr) < 3 or len(te) == 0:
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(np.nan_to_num(X[tr]), y[tr])
            oof[te] = model.predict(np.nan_to_num(X[te]))
        t = tau_c(oof, y)
        if t is not None:
            taus.append(t)
    return {"mean": float(np.mean(taus)) if taus else None,
            "std": float(np.std(taus)) if taus else None,
            "n_seeds": len(taus)}


def paired_boot(oof_a: np.ndarray, oof_b: np.ndarray, y: np.ndarray, seed: int = 2026,
                 n_boot: int = 2000) -> tuple[float, float, float, float]:
    """Bootstrap the tau-c(b) - tau-c(a) difference over PIECES, resampling the
    SAME indices for both arms so the fold noise they share cancels. Promoted
    from features37_compare.py (a standalone `# /// script` that ft_eval.py
    cannot import -- lightgbm is not in model/.venv) so the gate (ft_eval.py,
    realaudio_check.py) and the Phase 0 baseline share one bootstrap
    implementation. Returns (mean_diff, ci_lo, ci_hi, P(diff <= 0))."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs[b] = (stats.kendalltau(oof_b[i], y[i], variant="c").statistic
                    - stats.kendalltau(oof_a[i], y[i], variant="c").statistic)
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return float(np.mean(diffs)), lo, hi, float(np.mean(diffs <= 0))
