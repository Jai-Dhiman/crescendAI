"""Binary classifiers for STOP/CONTINUE prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    test_idx: list[int],
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Train logistic regression and return predictions on test set.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N] (1=stop, 0=continue).
        train_idx: Indices for training.
        test_idx: Indices for testing.
        C: Regularization strength (inverse).
        max_iter: Maximum iterations.

    Returns:
        Dict with y_true, y_pred, y_pred_proba, coefficients.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)

    return {
        "y_true": y_test,
        "y_pred": clf.predict(X_test),
        "y_pred_proba": clf.predict_proba(X_test)[:, 1],
        "coefficients": clf.coef_.squeeze(),
    }
