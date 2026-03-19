"""Tests for Aria linear probe evaluation."""

import numpy as np
import torch
import pytest


class TestPairwiseFromRegression:
    """Test converting pointwise regression to pairwise accuracy."""

    def test_perfect_predictions_give_high_accuracy(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        predictions = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        ])
        labels = {
            "a": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            "b": np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
            "c": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
            "d": np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
        }
        keys = ["a", "b", "c", "d"]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert result["overall"] > 0.95

    def test_random_predictions_near_50_pct(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        torch.manual_seed(123)
        np.random.seed(123)
        n = 50
        predictions = torch.rand(n, 6)
        labels = {
            f"s{i}": np.random.rand(6).astype(np.float32)
            for i in range(n)
        }
        keys = [f"s{i}" for i in range(n)]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert 0.35 < result["overall"] < 0.65

    def test_per_dimension_breakdown_returned(self):
        from model_improvement.aria_linear_probe import (
            compute_pairwise_from_regression,
        )

        predictions = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        ])
        labels = {
            "a": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            "b": np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
        }
        keys = ["a", "b"]
        result = compute_pairwise_from_regression(predictions, keys, labels)
        assert "per_dimension" in result
        assert len(result["per_dimension"]) == 6


class TestTrainLinearProbe:
    def test_probe_trains_and_returns_predictions(self):
        from model_improvement.aria_linear_probe import train_linear_probe

        torch.manual_seed(42)
        n_train, n_val = 20, 5
        dim = 32
        train_emb = torch.randn(n_train, dim)
        train_labels = torch.rand(n_train, 6)
        val_emb = torch.randn(n_val, dim)
        val_labels = torch.rand(n_val, 6)

        val_preds, train_preds = train_linear_probe(
            train_emb, train_labels, val_emb, val_labels,
            lr=1e-2, weight_decay=0.0, max_epochs=50, patience=10,
        )
        assert val_preds.shape == (n_val, 6)
        assert train_preds.shape == (n_train, 6)

    def test_probe_on_perfectly_linear_data(self):
        from model_improvement.aria_linear_probe import train_linear_probe

        torch.manual_seed(42)
        dim = 16
        W = torch.randn(dim, 6)
        train_emb = torch.randn(100, dim)
        train_labels = train_emb @ W
        val_emb = torch.randn(20, dim)
        val_labels = val_emb @ W

        val_preds, _ = train_linear_probe(
            train_emb, train_labels, val_emb, val_labels,
            lr=1e-2, weight_decay=0.0, max_epochs=200, patience=20,
        )
        from model_improvement.metrics import MetricsSuite
        suite = MetricsSuite()
        r2 = suite.regression_r2(val_preds, val_labels)
        assert r2 > 0.90


class TestErrorCorrelation:
    def test_identical_errors_give_high_phi(self):
        from model_improvement.aria_linear_probe import (
            compute_error_correlation,
        )

        correct_a = torch.tensor([True, True, False, False, True])
        correct_b = torch.tensor([True, True, False, False, True])
        phi = compute_error_correlation(correct_a, correct_b)
        assert phi > 0.95

    def test_independent_errors_give_low_phi(self):
        from model_improvement.aria_linear_probe import (
            compute_error_correlation,
        )

        torch.manual_seed(42)
        n = 1000
        correct_a = torch.randint(0, 2, (n,), dtype=torch.bool)
        correct_b = torch.randint(0, 2, (n,), dtype=torch.bool)
        phi = compute_error_correlation(correct_a, correct_b)
        assert -0.15 < phi < 0.15
