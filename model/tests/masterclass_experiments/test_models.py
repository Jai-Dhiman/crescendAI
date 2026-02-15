import numpy as np

from masterclass_experiments.models import train_classifier


def test_train_classifier_returns_predictions():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    y = rng.integers(0, 2, size=20)

    result = train_classifier(
        X, y, train_idx=list(range(15)), test_idx=list(range(15, 20))
    )

    assert "y_pred_proba" in result
    assert "y_pred" in result
    assert "y_true" in result
    assert "coefficients" in result
    assert len(result["y_pred_proba"]) == 5
    assert len(result["y_true"]) == 5


def test_train_classifier_coefficients_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 19))
    y = rng.integers(0, 2, size=30)

    result = train_classifier(
        X, y, train_idx=list(range(25)), test_idx=list(range(25, 30))
    )

    assert result["coefficients"].shape == (19,)
