"""Tests for T5 skill discrimination pairwise metric."""

import numpy as np
import pytest

from model_improvement.skill_discrimination import (
    skill_discrimination_pairwise,
)


class TestSkillDiscrimination:
    def test_perfect_discrimination(self):
        """Model scores perfectly correlate with skill buckets -> 100% accuracy."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        buckets = np.array([1, 2, 3, 4, 5])
        result = skill_discrimination_pairwise(scores, buckets)
        assert result["pairwise_accuracy"] == 1.0
        assert result["n_pairs"] == 10  # C(5,2) = 10

    def test_single_recording_bucket(self):
        """Bucket with 1 recording still generates cross-bucket pairs."""
        scores = np.array([0.1, 0.3, 0.5])
        buckets = np.array([1, 1, 5])
        result = skill_discrimination_pairwise(scores, buckets)
        assert result["n_pairs"] == 2
        assert result["pairwise_accuracy"] == 1.0

    def test_per_dimension_shape(self):
        """Per-dimension breakdown returns one entry per dimension."""
        n_dims = 6
        scores = np.random.rand(10, n_dims)
        buckets = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        result = skill_discrimination_pairwise(scores, buckets)
        assert len(result["per_dimension"]) == n_dims
        for d in range(n_dims):
            assert 0.0 <= result["per_dimension"][d] <= 1.0
