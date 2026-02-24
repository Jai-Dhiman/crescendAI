import torch
import numpy as np
import pytest
from model_improvement.metrics import (
    MetricsSuite,
    compute_robustness_metrics,
    format_comparison_table,
    stop_auc,
    competition_spearman,
    per_dimension_breakdown,
)


def test_metrics_pairwise_accuracy():
    suite = MetricsSuite(ambiguous_threshold=0.05)
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.8]])
    labels_a = torch.tensor([[0.8, 0.3], [0.6, 0.9]])
    labels_b = torch.tensor([[0.3, 0.8], [0.4, 0.5]])
    result = suite.pairwise_accuracy(logits, labels_a, labels_b)
    assert "overall" in result
    assert "per_dimension" in result
    assert 0.0 <= result["overall"] <= 1.0


def test_metrics_regression_r2():
    suite = MetricsSuite()
    predictions = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    targets = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    r2 = suite.regression_r2(predictions, targets)
    assert r2 > 0.99


def test_metrics_difficulty_correlation():
    suite = MetricsSuite()
    predictions = torch.randn(100, 19)
    difficulties = torch.randn(100)
    result = suite.difficulty_correlation(predictions, difficulties)
    assert "overall_rho" in result
    assert "per_dimension" in result


def test_robustness_metrics():
    clean_scores = torch.randn(50, 19)
    augmented_scores = clean_scores + torch.randn(50, 19) * 0.1
    result = compute_robustness_metrics(clean_scores, augmented_scores)
    assert "pearson_r" in result
    assert "score_drop_pct" in result


def test_format_comparison_table():
    results = {
        "A1": {"r2": 0.55, "pairwise": 0.85, "difficulty_rho": 0.63, "robustness": 0.92, "gpu_hours": 2.5},
        "A2": {"r2": 0.58, "pairwise": 0.87, "difficulty_rho": 0.65, "robustness": 0.94, "gpu_hours": 8.0},
    }
    table = format_comparison_table(results)
    assert isinstance(table, str)
    assert "A1" in table
    assert "A2" in table


def test_stop_auc():
    embeddings = torch.randn(50, 1024)
    is_stop = torch.tensor([1] * 25 + [0] * 25)
    video_ids = [f"v{i % 5}" for i in range(50)]
    result = stop_auc(embeddings, is_stop, video_ids)
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_competition_spearman():
    predictions = torch.randn(20, 6)
    placements = torch.arange(1, 21, dtype=torch.float32)
    result = competition_spearman(predictions, placements)
    assert "rho" in result
    assert "p_value" in result


def test_competition_spearman_empty():
    result = competition_spearman(None, None)
    assert result is None


def test_per_dimension_breakdown():
    suite = MetricsSuite()
    logits = torch.randn(100, 6)
    labels_a = torch.rand(100, 6)
    labels_b = torch.rand(100, 6)
    result = per_dimension_breakdown(suite, logits, labels_a, labels_b)
    assert len(result) == 6
    assert "dynamics" in result
