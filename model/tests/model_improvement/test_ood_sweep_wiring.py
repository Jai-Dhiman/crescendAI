"""Tests for OOD aggregation wired into every sweep (issue #76).

Every sweep must emit OOD pairwise + OOD-minus-fold gap. The gap here is the
clean-fold pairwise minus OOD pairwise (a positive number = degradation; the
practice-augmentation exit criterion is gap <= 0.10).
"""

from __future__ import annotations

import numpy as np

from model_improvement.evaluation import (
    summarize_ood_folds,
    build_validation_gate_block,
    print_validation_gate_summary,
)


def test_summarize_ood_folds_empty_is_skipped():
    out = summarize_ood_folds([{"skipped": "empty_ood_dataset", "n_samples": 0}])
    assert out["skipped"] == "empty_ood_dataset"
    assert out["n_folds_scored"] == 0


def test_summarize_ood_folds_computes_gap():
    ood_folds = [
        {"pairwise": 0.70, "n_samples": 30},
        {"pairwise": 0.66, "n_samples": 30},
    ]
    out = summarize_ood_folds(ood_folds, fold_pairwise_mean=0.80)
    assert abs(out["ood_pairwise_mean"] - 0.68) < 1e-9
    # clean - ood = degradation
    assert abs(out["ood_minus_fold_gap"] - (0.80 - 0.68)) < 1e-9
    assert out["n_folds_scored"] == 2


def test_summarize_ood_folds_mixed_skipped_and_scored():
    ood_folds = [
        {"pairwise": 0.72, "n_samples": 30},
        {"skipped": "empty_ood_dataset", "n_samples": 0},
    ]
    out = summarize_ood_folds(ood_folds, fold_pairwise_mean=0.80)
    assert out["n_folds_scored"] == 1
    assert abs(out["ood_pairwise_mean"] - 0.72) < 1e-9


def test_block_embeds_ood_summary():
    fold_metrics = [{"dimension_collapse_score": 0.30, "pairwise": 0.80}]
    ood_summary = summarize_ood_folds(
        [{"pairwise": 0.70, "n_samples": 30}], fold_pairwise_mean=0.80
    )
    block = build_validation_gate_block(fold_metrics, ood_summary=ood_summary)
    assert "ood" in block
    assert abs(block["ood"]["ood_minus_fold_gap"] - 0.10) < 1e-9


def test_block_ood_skipped_when_absent():
    block = build_validation_gate_block([{"dimension_collapse_score": 0.30}])
    assert block["ood"]["skipped"] == "not_run"


def test_print_summary_includes_ood(capsys):
    ood_summary = summarize_ood_folds(
        [{"pairwise": 0.70, "n_samples": 30}], fold_pairwise_mean=0.80
    )
    block = build_validation_gate_block(
        [{"dimension_collapse_score": 0.30}], ood_summary=ood_summary
    )
    print_validation_gate_summary(block)
    out = capsys.readouterr().out
    assert "OOD" in out
    assert "gap" in out.lower()
