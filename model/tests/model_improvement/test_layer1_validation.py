"""Tests for Layer 1 validation experiment helpers."""

import numpy as np
import pytest
import torch

from model_improvement.layer1_validation import (
    score_competition_segments,
    competition_correlation,
)


def test_score_competition_segments_returns_dict():
    """score_competition_segments returns {segment_id: scores_array}."""

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1024, 6)

        def predict_scores(self, x, mask=None):
            return torch.tensor([[0.5, 0.6, 0.7, 0.4, 0.3, 0.8]])

    model = FakeModel()
    embeddings = {
        "seg_001": torch.randn(10, 1024),
        "seg_002": torch.randn(15, 1024),
    }
    result = score_competition_segments(model, embeddings)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"seg_001", "seg_002"}
    for scores in result.values():
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (6,)


def test_competition_correlation_spearman():
    """competition_correlation computes Spearman rho per aggregation method."""
    segment_scores = {
        "perf1_seg000": np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.9]),
        "perf1_seg001": np.array([0.7, 0.8, 0.5, 0.6, 0.5, 0.8]),
        "perf2_seg000": np.array([0.3, 0.4, 0.3, 0.2, 0.3, 0.4]),
        "perf2_seg001": np.array([0.4, 0.3, 0.4, 0.3, 0.2, 0.3]),
    }
    metadata = [
        {"segment_id": "perf1_seg000", "performer": "Alice", "placement": 1, "round": "final"},
        {"segment_id": "perf1_seg001", "performer": "Alice", "placement": 1, "round": "final"},
        {"segment_id": "perf2_seg000", "performer": "Bob", "placement": 5, "round": "stage2"},
        {"segment_id": "perf2_seg001", "performer": "Bob", "placement": 5, "round": "stage2"},
    ]
    result = competition_correlation(segment_scores, metadata)
    assert "mean" in result
    assert "median" in result
    assert "min" in result
    for agg_name, corr in result.items():
        assert "rho" in corr
        assert "p_value" in corr
        assert "per_dimension" in corr
        assert isinstance(corr["per_dimension"], dict)
        assert len(corr["per_dimension"]) == 6
