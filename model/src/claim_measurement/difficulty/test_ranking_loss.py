"""Tests for ranking_loss (#149 / #138 Phase 1) -- pairwise ranking + ordinal
auxiliary loss, real torch on CPU, no mocks.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import torch

from claim_measurement.difficulty.ranking_loss import ordered_pairs


def test_ordered_pairs_finds_all_strictly_grade_ordered_index_pairs():
    grades = torch.tensor([3, 1, 3, 2])

    pairs = ordered_pairs(grades)

    pair_set = {tuple(p.tolist()) for p in pairs}
    assert pair_set == {(0, 1), (0, 3), (2, 1), (2, 3), (3, 1)}
