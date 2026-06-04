"""Verify recall@k, MRR, and open-set curve computations against known answers."""
from __future__ import annotations

import numpy as np

from piece_id_eval.metrics import mrr, open_set_curve, open_set_ok, recall_at_k


def test_recall_at_1_perfect() -> None:
    # Each query: true piece is ranked #1
    rankings = [
        ("piece_a", [("piece_a", 1.0), ("piece_b", 0.5)]),
        ("piece_b", [("piece_b", 0.9), ("piece_a", 0.3)]),
    ]
    assert recall_at_k(rankings, k=1) == 1.0


def test_recall_at_1_miss() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_a", 0.5)]),
    ]
    assert recall_at_k(rankings, k=1) == 0.0


def test_recall_at_k_partial() -> None:
    # piece_a is at rank 3; recall@2 = 0.0, recall@3 = 1.0
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_c", 0.9), ("piece_a", 0.8)]),
    ]
    assert recall_at_k(rankings, k=2) == 0.0
    assert recall_at_k(rankings, k=3) == 1.0


def test_recall_at_k_mixed() -> None:
    # 2 of 4 queries have truth in top-2
    rankings = [
        ("p1", [("p1", 1.0), ("p2", 0.5)]),        # hit at rank 1
        ("p2", [("p1", 1.0), ("p2", 0.5)]),        # miss at rank 2 (rank 2 is p2, wait no p2 IS at rank 2)
        ("p3", [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]),  # miss at rank 2
        ("p4", [("p5", 1.0), ("p6", 0.9)]),        # miss
    ]
    # p1: rank 1 hit; p2: rank 2 hit; p3: miss@2; p4: miss@2
    assert recall_at_k(rankings, k=2) == 0.5


def test_mrr_perfect() -> None:
    rankings = [
        ("piece_a", [("piece_a", 1.0)]),
        ("piece_b", [("piece_b", 0.9)]),
    ]
    assert mrr(rankings) == 1.0


def test_mrr_rank2() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_a", 0.5)]),
    ]
    assert abs(mrr(rankings) - 0.5) < 1e-9


def test_mrr_not_found_contributes_zero() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_c", 0.5)]),
    ]
    assert mrr(rankings) == 0.0


def test_open_set_curve_shape() -> None:
    # in-catalog scores high, out-of-catalog scores low
    in_scores = np.array([0.9, 0.8, 0.7])
    out_scores = np.array([0.3, 0.2])
    thresholds = np.linspace(0.0, 1.0, 11)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    assert fa.shape == thresholds.shape
    assert ta.shape == thresholds.shape
    # At threshold=0: all accepted -> fa=1.0, ta=1.0
    assert fa[0] == 1.0
    assert ta[0] == 1.0


def test_open_set_ok_passes_when_criteria_met() -> None:
    in_scores = np.array([0.9, 0.8, 0.85, 0.75])
    out_scores = np.array([0.1, 0.05])
    thresholds = np.linspace(0.0, 1.0, 101)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    # With clean separation a threshold exists where fa<=0.10, ta>=0.75
    assert open_set_ok(fa, ta, max_fa=0.10, min_ta=0.75)


def test_open_set_ok_fails_when_no_threshold_exists() -> None:
    # All scores identical -> no threshold can separate in/out
    in_scores = np.array([0.5, 0.5])
    out_scores = np.array([0.5, 0.5])
    thresholds = np.linspace(0.0, 1.0, 101)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    assert not open_set_ok(fa, ta, max_fa=0.10, min_ta=0.75)
