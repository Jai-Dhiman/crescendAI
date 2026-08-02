"""Tests for bakeoff_cv (ported from the unmerged issue-104-mirex-difficulty
branch's phase5b_aria_probe.py, commit 7976b5e6 -- see the design spec for why
this is a port, not a cross-branch import).

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import math

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import (
    composer_disjoint_folds,
    oof_tau_ridge,
    tau_c,
)


def test_tau_c_perfect_agreement_is_one():
    assert tau_c([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_tau_c_perfect_disagreement_is_minus_one():
    assert tau_c([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_tau_c_none_for_constant_y():
    assert tau_c([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_tau_c_none_for_fewer_than_three_points():
    assert tau_c([1, 2], [1, 2]) is None


def test_tau_c_handles_ties_without_raising():
    result = tau_c([1, 1, 2, 3], [1, 2, 2, 3])
    assert result is not None
    assert not math.isnan(result)


def test_composer_disjoint_folds_no_composer_straddles_a_fold():
    rng = np.random.default_rng(0)
    composers = np.array(
        [f"composer_{i % 30}" for i in range(300)]
    )
    rng.shuffle(composers)

    folds = composer_disjoint_folds(composers, n_folds=5, seed=2026)

    assert len(folds) == 5
    all_indices = np.concatenate(folds)
    assert sorted(all_indices) == list(range(300))  # every row covered exactly once
    fold_composer_sets = [set(composers[f]) for f in folds]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (fold_composer_sets[i] & fold_composer_sets[j]), (
                f"fold {i} and fold {j} share a composer"
            )


def test_oof_tau_ridge_recovers_a_strong_linear_signal():
    rng = np.random.default_rng(2026)
    n = 200
    composers = np.array([f"composer_{i % 20}" for i in range(n)])
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 10  # near-perfectly linearly predictable from feature 0

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[2026, 2027])

    assert result["n_seeds"] == 2
    assert result["mean"] > 0.5  # a strong linear signal should rank well OOF
    assert result["std"] >= 0.0


def test_oof_tau_ridge_reports_zero_seeds_when_target_is_constant():
    rng = np.random.default_rng(0)
    n = 60
    composers = np.array([f"composer_{i % 10}" for i in range(n)])
    X = rng.normal(size=(n, 3))
    y = np.zeros(n)  # constant target -> tau_c is always None

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[2026])

    assert result == {"mean": None, "std": None, "n_seeds": 0}
