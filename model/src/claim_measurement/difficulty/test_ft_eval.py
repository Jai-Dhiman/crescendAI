"""Tests for ft_eval (#149 / #138 Phase 1) -- the gate: OOF where X differs
per fold, plus the CLI wiring against features37 + emb_fold{F}.npz files.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_cv import tau_c
from claim_measurement.difficulty.ft_eval import oof_tau_per_fold


def test_oof_tau_per_fold_recovers_a_strong_per_fold_linear_signal():
    rng = np.random.default_rng(2026)
    n = 200
    composers = np.array([f"composer_{i}" for i in range(n)])  # all distinct -> vacuous disjointness
    y = rng.integers(0, 11, size=n).astype(float)

    emb_by_fold = {}
    for f in range(5):
        rng_f = np.random.default_rng(1000 + f)
        noise = rng_f.normal(size=(n, 3)) * 0.01
        emb_by_fold[f] = np.column_stack([y * (f + 1), noise])

    oof = oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)

    assert not np.isnan(oof).any()
    assert tau_c(oof, y) > 0.9


def test_oof_tau_per_fold_raises_on_missing_fold_embeddings():
    composers = np.array([f"composer_{i}" for i in range(50)])
    y = np.arange(50, dtype=float) % 11
    emb_by_fold = {0: np.random.default_rng(0).normal(size=(50, 2))}  # folds 1-4 gone

    with pytest.raises(KeyError):
        oof_tau_per_fold(emb_by_fold, y, composers, n_folds=5, seed=2026)
