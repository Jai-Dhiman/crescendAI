import numpy as np

from masterclass_experiments.diagnostics import (
    coefficient_stability,
    per_fold_metrics,
    permutation_test,
)


def _make_separable_data(rng, n=60):
    """Create data where feature 0 predicts the label."""
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] > 0).astype(int)
    video_ids = np.array(["v1"] * 20 + ["v2"] * 20 + ["v3"] * 20)
    return X, y, video_ids


def test_permutation_test_returns_pvalue_and_null():
    rng = np.random.default_rng(42)
    X, y, video_ids = _make_separable_data(rng)

    result = permutation_test(X, y, video_ids, n_permutations=50, random_state=42)

    assert "observed_auc" in result
    assert "p_value" in result
    assert "null_distribution" in result
    assert len(result["null_distribution"]) == 50
    assert 0.0 <= result["p_value"] <= 1.0


def test_permutation_test_real_signal_has_low_pvalue():
    rng = np.random.default_rng(42)
    X, y, video_ids = _make_separable_data(rng)

    result = permutation_test(X, y, video_ids, n_permutations=100, random_state=42)

    # With a clearly separable signal, p-value should be very low
    assert result["p_value"] < 0.1
    assert result["observed_auc"] > 0.7


def test_per_fold_metrics_returns_one_entry_per_video():
    rng = np.random.default_rng(42)
    X, y, video_ids = _make_separable_data(rng)

    folds = per_fold_metrics(X, y, video_ids)

    assert len(folds) == 3
    for fold in folds:
        assert "held_out_video" in fold
        assert "auc" in fold
        assert "n_test" in fold
        assert "n_train" in fold
        assert "coefficients" in fold


def test_coefficient_stability_returns_mean_and_std():
    rng = np.random.default_rng(42)
    X, y, video_ids = _make_separable_data(rng)

    result = coefficient_stability(X, y, video_ids)

    assert "mean_coefs" in result
    assert "std_coefs" in result
    assert "sign_consistency" in result
    assert result["mean_coefs"].shape == (5,)
    assert result["std_coefs"].shape == (5,)
    assert result["sign_consistency"].shape == (5,)
    # Sign consistency is a ratio between 0 and 1
    assert all(0.0 <= s <= 1.0 for s in result["sign_consistency"])
