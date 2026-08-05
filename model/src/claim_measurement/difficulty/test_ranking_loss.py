"""Tests for ranking_loss (#149 / #138 Phase 1) -- pairwise ranking + ordinal
auxiliary loss, real torch on CPU, no mocks.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import torch

from claim_measurement.difficulty.ranking_loss import ordered_pairs
from claim_measurement.difficulty.ranking_loss import pairwise_ranking_loss


def test_ordered_pairs_finds_all_strictly_grade_ordered_index_pairs():
    grades = torch.tensor([3, 1, 3, 2])

    pairs = ordered_pairs(grades)

    pair_set = {tuple(p.tolist()) for p in pairs}
    assert pair_set == {(0, 1), (0, 3), (2, 1), (2, 3), (3, 1)}


def test_pairwise_ranking_loss_is_lower_for_correctly_ranked_scores():
    grades = torch.tensor([1, 2, 3])
    correct_scores = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    reversed_scores = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)

    correct_loss = pairwise_ranking_loss(correct_scores, grades)
    reversed_loss = pairwise_ranking_loss(reversed_scores, grades)

    assert correct_loss.item() < reversed_loss.item()
