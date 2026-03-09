"""Tests for Layer 1 validation experiment helpers."""

import numpy as np
import pytest
import torch

from model_improvement.layer1_validation import (
    score_competition_segments,
    competition_correlation,
    amt_degradation_comparison,
    select_maestro_subset,
    dynamic_range_analysis,
)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 6)

    def predict_scores(self, x, mask=None):
        return torch.tensor([[0.5, 0.6, 0.7, 0.4, 0.3, 0.8]])


def test_score_competition_segments_returns_dict():
    """score_competition_segments returns {segment_id: scores_array}."""
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


def test_score_competition_segments_from_directory(tmp_path):
    """score_competition_segments loads .pt files lazily from a directory."""
    model = FakeModel()
    torch.save(torch.randn(10, 1024), tmp_path / "seg_a.pt")
    torch.save(torch.randn(15, 1024), tmp_path / "seg_b.pt")

    result = score_competition_segments(model, tmp_path)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"seg_a", "seg_b"}
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


def test_amt_degradation_comparison():
    """amt_degradation_comparison returns per-dimension pairwise accuracy drop."""
    pairwise_results = {
        "ground_truth": {
            "overall": 0.72,
            "per_dimension": {0: 0.77, 1: 0.65, 2: 0.72, 3: 0.70, 4: 0.63, 5: 0.77},
        },
        "yourmt3": {
            "overall": 0.68,
            "per_dimension": {0: 0.73, 1: 0.62, 2: 0.68, 3: 0.65, 4: 0.60, 5: 0.72},
        },
        "bytedance": {
            "overall": 0.60,
            "per_dimension": {0: 0.65, 1: 0.55, 2: 0.60, 3: 0.58, 4: 0.52, 5: 0.64},
        },
    }
    result = amt_degradation_comparison(pairwise_results, baseline="ground_truth")
    assert "yourmt3" in result
    assert "bytedance" in result
    for source, drops in result.items():
        assert "overall_drop_pct" in drops
        assert "per_dimension_drop_pct" in drops
        assert len(drops["per_dimension_drop_pct"]) == 6
        # All drops should be non-negative (AMT should be worse)
        assert drops["overall_drop_pct"] >= 0


def test_select_maestro_subset():
    """select_maestro_subset picks pieces with multiple performers."""
    contrastive_mapping = {
        "piece_a": ["perf1", "perf2", "perf3"],
        "piece_b": ["perf4", "perf5"],
        "piece_c": ["perf6"],  # single performer, should be excluded
    }
    result = select_maestro_subset(contrastive_mapping, n_recordings=4)
    assert len(result) == 4
    # All selected recordings should come from multi-performer pieces
    for key in result:
        found = False
        for piece, perfs in contrastive_mapping.items():
            if key in perfs and len(perfs) >= 2:
                found = True
                break
        assert found, f"{key} not from multi-performer piece"


def test_select_maestro_subset_deduplicates_segments():
    """select_maestro_subset deduplicates segment IDs to unique recordings."""
    contrastive_mapping = {
        "piece_a": [
            "maestro_rec1_seg000", "maestro_rec1_seg001",  # same recording
            "maestro_rec2_seg000", "maestro_rec2_seg001",  # different recording
        ],
        "piece_b": [
            "maestro_rec3_seg000",
            "maestro_rec4_seg000",
        ],
    }
    result = select_maestro_subset(contrastive_mapping, n_recordings=10)
    # Should return 4 unique recordings, not 6 segments
    assert len(result) == 4
    assert set(result) == {"maestro_rec1", "maestro_rec2", "maestro_rec3", "maestro_rec4"}


def test_dynamic_range_analysis():
    """dynamic_range_analysis returns separation and variance stats."""
    scores_by_group = {
        "intermediate": {
            "player1_seg0": np.array([0.4, 0.5, 0.3, 0.4, 0.3, 0.4]),
            "player1_seg1": np.array([0.5, 0.4, 0.4, 0.5, 0.4, 0.3]),
            "player2_seg0": np.array([0.3, 0.3, 0.2, 0.3, 0.2, 0.3]),
        },
        "advanced": {
            "adv1_seg0": np.array([0.7, 0.8, 0.6, 0.7, 0.6, 0.8]),
            "adv1_seg1": np.array([0.8, 0.7, 0.7, 0.8, 0.7, 0.7]),
        },
    }
    result = dynamic_range_analysis(scores_by_group)
    assert "separation" in result
    assert "within_group_variance" in result
    assert "per_dimension" in result
    # Separation should be positive (advanced > intermediate)
    assert result["separation"]["overall"] > 0
    # Per-dimension should have 6 entries
    assert len(result["per_dimension"]) == 6
