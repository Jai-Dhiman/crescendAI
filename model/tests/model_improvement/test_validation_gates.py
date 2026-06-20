"""Tests for the WS2 validation gates wired into every sweep (issue #75).

Three gates:
  1. dimension_collapse_score (already in evaluation.py; covered by the block test)
  2. MuQ<->Aria error-correlation (G2) with explicit <0.5 pass/fail
  3. per-piece pairwise + bootstrap CI with single-piece regression flags
"""

from __future__ import annotations

import numpy as np
import torch

import torch.nn as nn

from model_improvement.evaluation import (
    evaluate_model,
    pairwise_overall_masks,
    error_correlation_gate,
    per_piece_pairwise,
    per_piece_pairwise_bootstrap,
    flag_single_piece_regressions,
    build_validation_gate_block,
    print_validation_gate_summary,
)
from model_improvement.taxonomy import NUM_DIMS


# --- pairwise_overall_masks ---------------------------------------------------


def test_pairwise_overall_masks_shapes_and_correctness():
    # Pair 0: logits say A>B (sum>0), labels say A>B (sum>0)   -> correct, non-amb
    # Pair 1: logits say A>B, labels say B>A                   -> wrong,   non-amb
    # Pair 2: labels equal (ambiguous)                          -> excluded
    logits = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    labels_a = torch.tensor([[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]])
    labels_b = torch.tensor([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])

    correct, non_amb = pairwise_overall_masks(logits, labels_a, labels_b, threshold=0.05)

    assert correct.shape == (3,)
    assert non_amb.tolist() == [True, True, False]
    assert correct[0]  # predicted A>B, true A>B
    assert not correct[1]  # predicted A>B, true B>A


# --- error_correlation_gate (G2) ----------------------------------------------


def test_g2_gate_independent_errors_pass():
    # Two streams whose errors are independent -> phi ~ 0 -> PASS (<0.5)
    rng = np.random.RandomState(0)
    n = 400
    correct_a = rng.rand(n) > 0.3
    correct_b = rng.rand(n) > 0.3
    non_amb = np.ones(n, dtype=bool)
    masks_a = {"correct": correct_a, "non_ambiguous": non_amb}
    masks_b = {"correct": correct_b, "non_ambiguous": non_amb}

    gate = error_correlation_gate(masks_a, masks_b, threshold=0.5)

    assert gate["pass"] is True
    assert gate["threshold"] == 0.5
    assert abs(gate["phi_mean"]) < 0.5
    assert "phi_per_fold" in gate


def test_g2_gate_identical_errors_fail():
    # Two streams that make the SAME mistakes -> phi ~ 1 -> FAIL (redundant)
    rng = np.random.RandomState(1)
    n = 400
    correct = rng.rand(n) > 0.3
    non_amb = np.ones(n, dtype=bool)
    masks_a = {"correct": correct.copy(), "non_ambiguous": non_amb}
    masks_b = {"correct": correct.copy(), "non_ambiguous": non_amb}

    gate = error_correlation_gate(masks_a, masks_b, threshold=0.5)

    assert gate["pass"] is False
    assert gate["phi_mean"] > 0.5


def test_g2_gate_accepts_per_fold_lists():
    n = 200
    rng = np.random.RandomState(2)
    folds_a = [
        {"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}
        for _ in range(3)
    ]
    folds_b = [
        {"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}
        for _ in range(3)
    ]
    gate = error_correlation_gate(folds_a, folds_b, threshold=0.5)
    assert len(gate["phi_per_fold"]) == 3
    assert gate["pass"] is True


def test_g2_gate_handles_constant_vector_gracefully():
    # A stream that gets everything correct on shared pairs -> phi undefined.
    n = 100
    masks_a = {"correct": np.ones(n, dtype=bool), "non_ambiguous": np.ones(n, dtype=bool)}
    masks_b = {"correct": (np.arange(n) % 2 == 0), "non_ambiguous": np.ones(n, dtype=bool)}
    gate = error_correlation_gate(masks_a, masks_b, threshold=0.5)
    # Degenerate fold is excluded; with no valid folds phi_mean is nan and gate
    # is not a spurious pass.
    assert np.isnan(gate["phi_mean"])
    assert gate["pass"] is False
    assert gate["n_valid_folds"] == 0


# --- per-piece pairwise + bootstrap -------------------------------------------


def _toy_pieces():
    # 4 keys in piece P, 4 in piece Q. Predictions perfectly rank within each.
    keys = [f"P_{i}" for i in range(4)] + [f"Q_{i}" for i in range(4)]
    piece_of = {k: ("P" if k.startswith("P") else "Q") for k in keys}
    preds = torch.tensor(
        [[float(i)] * 2 for i in range(4)] + [[float(i)] * 2 for i in range(4)]
    )
    labels = {k: preds[i].tolist() for i, k in enumerate(keys)}
    return preds, keys, labels, piece_of


def test_per_piece_pairwise_groups_by_piece():
    preds, keys, labels, piece_of = _toy_pieces()
    out = per_piece_pairwise(preds, keys, labels, piece_of, threshold=0.05)
    assert set(out.keys()) == {"P", "Q"}
    # Predictions equal labels -> perfect ranking within each piece.
    assert out["P"]["pairwise"] == 1.0
    assert out["P"]["n_keys"] == 4


def test_per_piece_bootstrap_ci_brackets_point_estimate():
    preds, keys, labels, piece_of = _toy_pieces()
    out = per_piece_pairwise_bootstrap(
        preds, keys, labels, piece_of, n_boot=200, seed=42
    )
    for piece in ("P", "Q"):
        assert out[piece]["ci_low"] <= out[piece]["pairwise"] <= out[piece]["ci_high"] + 1e-9
        assert 0.0 <= out[piece]["ci_low"] <= 1.0


def test_flag_single_piece_regressions():
    current = {"P": {"pairwise": 0.70}, "Q": {"pairwise": 0.90}}
    baseline = {"P": {"pairwise": 0.85}, "Q": {"pairwise": 0.88}}
    flagged = flag_single_piece_regressions(current, baseline, tol=0.02)
    pieces = {f["piece"] for f in flagged}
    assert pieces == {"P"}  # P dropped 15pp; Q improved


# --- block + summary ----------------------------------------------------------


def test_build_validation_gate_block_single_stream_skips_g2():
    fold_metrics = [
        {"dimension_collapse_score": 0.30, "pairwise": 0.80},
        {"dimension_collapse_score": 0.34, "pairwise": 0.82},
    ]
    block = build_validation_gate_block(fold_metrics)
    assert abs(block["dimension_collapse_mean"] - 0.32) < 1e-6
    assert block["g2_error_correlation"]["skipped"] == "single_stream"


def test_build_validation_gate_block_with_two_streams_runs_g2():
    n = 300
    rng = np.random.RandomState(3)
    muq = [{"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}]
    aria = [{"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}]
    fold_metrics = [{"dimension_collapse_score": 0.31, "pairwise": 0.79}]
    block = build_validation_gate_block(
        fold_metrics, muq_masks=muq, aria_masks=aria
    )
    assert "pass" in block["g2_error_correlation"]


def test_print_validation_gate_summary_prints_g2_verdict(capsys):
    n = 300
    rng = np.random.RandomState(4)
    muq = [{"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}]
    aria = [{"correct": rng.rand(n) > 0.3, "non_ambiguous": np.ones(n, dtype=bool)}]
    block = build_validation_gate_block(
        [{"dimension_collapse_score": 0.31}], muq_masks=muq, aria_masks=aria
    )
    print_validation_gate_summary(block)
    out = capsys.readouterr().out
    assert "G2" in out
    assert ("PASS" in out or "FAIL" in out)
    assert "0.5" in out  # threshold shown


# --- sweep-style integration --------------------------------------------------


class _IdentityModel(nn.Module):
    """Stub whose prediction == its input, so preds==labels when labels match.

    Mirrors the a1_max_sweep call shape: encode -> compare -> predict_scores.
    Each embedding is a [NUM_DIMS] vector; one segment per row.
    """

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def encode(self, inp, mask):  # [1, NUM_DIMS] -> [1, NUM_DIMS]
        return inp

    def compare(self, z_a, z_b):  # ranking logits
        return z_a - z_b

    def predict_scores(self, inp, mask):
        return inp


def test_sweep_style_accumulation_emits_all_three_gates(capsys):
    # Two folds, two pieces. Labels == embeddings so the model ranks perfectly.
    rng = np.random.RandomState(7)
    keys = [f"P_{i}" for i in range(6)] + [f"Q_{i}" for i in range(6)]
    piece_of = {k: ("P" if k.startswith("P") else "Q") for k in keys}
    embeddings = {k: torch.tensor(rng.rand(NUM_DIMS), dtype=torch.float32) for k in keys}
    labels = {k: embeddings[k].tolist() for k in keys}
    folds = [
        {"val": keys[:6]},   # all of P
        {"val": keys[6:]},   # all of Q
    ]

    model = _IdentityModel()
    fold_metrics = []
    muq_fold_masks = []
    combined_preds, combined_keys = [], []
    for fold in folds:
        res = evaluate_model(
            model, fold["val"], labels,
            get_input_fn=lambda key: (embeddings[key].unsqueeze(0), None),
            encode_fn=lambda m, inp, mask: m.encode(inp, mask),
            compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
            predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
            return_pairwise_masks=True,
            return_predictions=True,
        )
        fold_metrics.append(res)
        muq_fold_masks.append({
            "correct": res["pairwise_masks"]["correct"],
            "non_ambiguous": res["pairwise_masks"]["non_ambiguous"],
        })
        combined_preds.extend(res["predictions"])
        combined_keys.extend(res["pred_keys"])

    # The eval path produced per-key predictions and per-pair masks.
    assert len(combined_preds) == len(keys)
    assert all("dimension_collapse_score" in m for m in fold_metrics)

    preds_tensor = torch.tensor(combined_preds, dtype=torch.float32)
    per_piece = per_piece_pairwise_bootstrap(
        preds_tensor, combined_keys, labels, piece_of, n_boot=50, seed=1
    )

    # Supply a synthetic Aria stream so G2 actually computes a pass/fail.
    aria_masks = [
        {"correct": np.asarray(m["correct"]), "non_ambiguous": np.asarray(m["non_ambiguous"])}
        for m in muq_fold_masks
    ]
    block = build_validation_gate_block(
        fold_metrics,
        muq_masks=muq_fold_masks,
        aria_masks=aria_masks,
        per_piece=per_piece,
    )

    # All three numbers present.
    assert block["dimension_collapse_mean"] is not None
    assert "pass" in block["g2_error_correlation"]
    assert set(block["per_piece_pairwise"].keys()) == {"P", "Q"}

    print_validation_gate_summary(block)
    out = capsys.readouterr().out
    assert "dimension_collapse_mean" in out
    assert "G2" in out and ("PASS" in out or "FAIL" in out)
    assert "per-piece pairwise" in out
