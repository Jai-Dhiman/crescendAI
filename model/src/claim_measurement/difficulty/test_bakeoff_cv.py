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
    paired_boot,
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


def test_composer_disjoint_folds_leaves_excess_folds_empty_when_composers_below_n_folds():
    # Fewer distinct composers (3) than requested folds (5): the extra folds
    # come back as empty arrays rather than raising or under-filling n_folds.
    composers = np.array([f"composer_{i % 3}" for i in range(30)])

    folds = composer_disjoint_folds(composers, n_folds=5, seed=1)

    assert len(folds) == 5
    non_empty = [f for f in folds if len(f) > 0]
    empty = [f for f in folds if len(f) == 0]
    assert len(non_empty) == 3
    assert len(empty) == 2
    all_indices = np.concatenate(folds)
    assert sorted(all_indices) == list(range(30))


def test_oof_tau_ridge_skips_empty_folds_when_composers_below_n_folds():
    # oof_tau_ridge must not crash on the empty test-folds produced above --
    # it should silently skip them (via the `len(te) == 0: continue` guard)
    # and still recover a valid tau-c from the non-empty folds.
    rng = np.random.default_rng(0)
    n = 30
    composers = np.array([f"composer_{i % 3}" for i in range(n)])
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 5

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[1, 2])

    assert result["n_seeds"] == 2
    assert result["mean"] is not None


def test_oof_tau_ridge_reports_zero_seeds_when_target_is_constant():
    rng = np.random.default_rng(0)
    n = 60
    composers = np.array([f"composer_{i % 10}" for i in range(n)])
    X = rng.normal(size=(n, 3))
    y = np.zeros(n)  # constant target -> tau_c is always None

    result = oof_tau_ridge(X, y, composers, n_folds=5, seeds=[2026])

    assert result == {"mean": None, "std": None, "n_seeds": 0}


def test_paired_boot_ci_is_strictly_positive_when_b_is_uniformly_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=200).astype(float)
    oof_a = y + rng.normal(scale=3.0, size=200)  # noisy
    oof_b = y + rng.normal(scale=0.2, size=200)  # much less noisy -> higher tau-c

    mean_diff, lo, hi, p_le_0 = paired_boot(oof_a, oof_b, y, seed=2026, n_boot=500)

    assert mean_diff > 0
    assert lo > 0
    assert p_le_0 < 0.05


def test_paired_boot_ci_straddles_zero_when_arms_are_identical():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, size=200).astype(float)
    oof_a = y + rng.normal(scale=1.0, size=200)
    oof_b = oof_a.copy()  # identical arm -> diff is exactly zero every resample

    mean_diff, lo, hi, p_le_0 = paired_boot(oof_a, oof_b, y, seed=2026, n_boot=200)

    assert abs(mean_diff) < 1e-9
    assert lo <= 0 <= hi
