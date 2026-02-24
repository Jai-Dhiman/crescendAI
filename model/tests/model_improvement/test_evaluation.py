import torch
import pytest
from model_improvement.evaluation import (
    aggregate_folds,
    select_winner,
)


def test_aggregate_folds():
    folds = [
        {"pairwise": 0.80, "r2": 0.50},
        {"pairwise": 0.85, "r2": 0.55},
        {"pairwise": 0.82, "r2": 0.52},
        {"pairwise": 0.78, "r2": 0.48},
    ]
    result = aggregate_folds(folds)
    assert "pairwise_mean" in result
    assert "pairwise_std" in result
    assert abs(result["pairwise_mean"] - 0.8125) < 0.001


def test_aggregate_folds_handles_missing_keys():
    folds = [
        {"pairwise": 0.80},
        {"pairwise": 0.85, "r2": 0.55},
    ]
    result = aggregate_folds(folds)
    assert "pairwise_mean" in result
    assert "r2_mean" in result


def test_select_winner():
    results = {
        "A1": {"pairwise_mean": 0.85, "r2_mean": 0.50, "score_drop_pct": 5.0},
        "A2": {"pairwise_mean": 0.87, "r2_mean": 0.55, "score_drop_pct": 8.0},
        "A3": {"pairwise_mean": 0.83, "r2_mean": 0.60, "score_drop_pct": 20.0},
    }
    winner = select_winner(results)
    assert winner == "A2"  # A3 vetoed (20% > 15%), A2 beats A1 on pairwise


def test_select_winner_all_vetoed():
    results = {
        "A1": {"pairwise_mean": 0.85, "score_drop_pct": 20.0},
    }
    winner = select_winner(results)
    assert winner is None
