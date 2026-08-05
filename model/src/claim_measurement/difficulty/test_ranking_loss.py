"""Tests for ranking_loss (#149 / #138 Phase 1) -- pairwise ranking + ordinal
auxiliary loss, real torch on CPU, no mocks.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import pytest
import torch

from claim_measurement.difficulty.ranking_loss import (
    combined_loss,
    ordered_pairs,
    ordinal_loss,
    pairwise_ranking_loss,
)


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


def test_pairwise_ranking_loss_is_a_finite_zero_for_a_degenerate_batch():
    # every piece shares one grade -> zero ordered pairs
    grades = torch.tensor([4, 4, 4])
    scores = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

    loss = pairwise_ranking_loss(scores, grades)

    assert loss.item() == 0.0
    loss.backward()  # must not raise -- the zero is still attached to the graph
    assert scores.grad is not None


def test_ordinal_loss_penalizes_wrong_threshold_predictions():
    grades = torch.tensor([0, 10])  # 11-level scale: min and max grade
    n_levels = 11
    correct_logits = torch.stack([torch.full((n_levels - 1,), -10.0),
                                   torch.full((n_levels - 1,), 10.0)])
    wrong_logits = torch.stack([torch.full((n_levels - 1,), 10.0),
                                 torch.full((n_levels - 1,), -10.0)])

    correct_loss = ordinal_loss(correct_logits, grades, n_levels)
    wrong_loss = ordinal_loss(wrong_logits, grades, n_levels)

    assert correct_loss.item() < wrong_loss.item()


def test_combined_loss_equals_pairwise_plus_weighted_ordinal():
    grades = torch.tensor([1, 3])
    scores = torch.tensor([0.2, 0.8])
    n_levels = 11
    ordinal_logits = torch.zeros((2, n_levels - 1))
    weight = 0.1

    combined = combined_loss(
        scores, ordinal_logits, grades, n_levels, ordinal_weight=weight
    )
    expected = (pairwise_ranking_loss(scores, grades)
                + weight * ordinal_loss(ordinal_logits, grades, n_levels))

    assert combined.item() == pytest.approx(expected.item())
