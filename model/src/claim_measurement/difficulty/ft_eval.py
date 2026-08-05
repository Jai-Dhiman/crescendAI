"""#138 Phase 1 gate (i): encoder-as-feature-extractor. Discards every fold's
trained head, scores fold f's mean-pooled emb_fold{f}.npz embeddings with
RidgeCV through bakeoff_cv.py's OWN composer-disjoint folds, and reports the
paired-bootstrap delta against features37|ridge on the SAME folds.

    cd model && uv run python -m claim_measurement.difficulty.ft_eval \\
        --data-root /path/to/model/data --fold-emb-dir /path/to/fold_embeddings

Per-fold X differs (each fold has its own adapter), which is why this needs
oof_tau_per_fold rather than bakeoff_cv.oof_tau_ridge -- see that function's
docstring. Seed is FIXED at 2026 (not averaged over multiple seeds like the
Phase 0 comparison): a set of per-fold adapters is welded to the (n_folds,
seed) pair that produced their training pools -- see the design spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds

N_FOLDS, SEED = 5, 2026
ALPHAS = np.logspace(-1, 5, 25)


def oof_tau_per_fold(emb_by_fold: dict, y: np.ndarray, composers: np.ndarray,
                      n_folds: int, seed: int) -> np.ndarray:
    """OOF predictions where X differs per fold: for fold f, BOTH the ridge
    head's train rows and its test rows come from emb_by_fold[f] -- the
    embeddings extracted by fold f's own adapter. Mixing rows across adapters
    would score a head fit on one encoder against another encoder's
    features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for f, test_idx in enumerate(composer_disjoint_folds(composers, n_folds, seed)):
        if f not in emb_by_fold:
            raise KeyError(f"emb_by_fold is missing fold {f}")
        X = emb_by_fold[f]
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict(np.nan_to_num(X[test_idx]))
    return oof


if __name__ == "__main__":
    sys.exit(0)  # placeholder exit; main() is added in Task 15
