"""Tests for external held-out validation loaders (issue #77).

External sets are ingested as HELD-OUT eval loaders, never training. Licenses
are verified before any use: NeuroPiano (MIT) and PianoVAM (CC-BY-NC-SA-4.0,
non-commercial eval) are allowed; PERiScoPe and PianoCoRe are blocked from this
path (PERiScoPe is NC-SA with no ratings; PianoCoRe's license is unconfirmed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from model_improvement.external_validation import (
    LicenseError,
    verify_license,
    NEUROPIANO_QUESTION_TO_DIM,
    map_neuropiano_axes_to_dims,
    ExternalEvalDataset,
    run_external_eval,
    skill_rank_spearman,
    d4_aria_gt_vs_amt,
)
from model_improvement.taxonomy import NUM_DIMS


# --- license gate -------------------------------------------------------------


def test_verify_license_allows_neuropiano_and_pianovam():
    assert verify_license("neuropiano")["spdx"] == "MIT"
    assert verify_license("pianovam")["commercial_ok"] is False


def test_verify_license_blocks_periscope_for_this_path():
    with pytest.raises(LicenseError):
        verify_license("periscope")


def test_verify_license_blocks_unconfirmed_pianocore():
    with pytest.raises(LicenseError):
        verify_license("pianocore")


def test_verify_license_unknown_raises():
    with pytest.raises(LicenseError):
        verify_license("some_random_set")


# --- NeuroPiano 13-axis -> 6-dim mapping --------------------------------------


def test_neuropiano_mapping_covers_known_axes_and_marks_pedaling_uncovered():
    # All rating axes (q3..q13) at max score 6 -> covered dims normalize to 1.0.
    scores = {qid: 6 for qid in NEUROPIANO_QUESTION_TO_DIM}
    vec = map_neuropiano_axes_to_dims(scores)
    assert len(vec) == NUM_DIMS
    # pedaling has no NeuroPiano axis -> NaN (uncovered), not a fabricated value.
    from model_improvement.taxonomy import DIMENSIONS
    ped_idx = DIMENSIONS.index("pedaling")
    assert math.isnan(vec[ped_idx])
    # a covered dim (timing) is normalized into [0, 1].
    timing_idx = DIMENSIONS.index("timing")
    assert vec[timing_idx] == pytest.approx(1.0)


def test_neuropiano_mapping_normalizes_zero_to_zero():
    scores = {qid: 0 for qid in NEUROPIANO_QUESTION_TO_DIM}
    vec = map_neuropiano_axes_to_dims(scores)
    from model_improvement.taxonomy import DIMENSIONS
    timing_idx = DIMENSIONS.index("timing")
    assert vec[timing_idx] == pytest.approx(0.0)


# --- skill-rank ---------------------------------------------------------------


def test_skill_rank_spearman_perfect_rank():
    pred = np.array([0.1, 0.5, 0.9])
    skill = np.array([1, 2, 3])
    assert skill_rank_spearman(pred, skill) == pytest.approx(1.0)


def test_skill_rank_spearman_inverse():
    pred = np.array([0.9, 0.5, 0.1])
    skill = np.array([1, 2, 3])
    assert skill_rank_spearman(pred, skill) == pytest.approx(-1.0)


# --- runner -------------------------------------------------------------------


class _IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def encode(self, inp, mask):
        return inp

    def compare(self, z_a, z_b):
        return z_a - z_b

    def predict_scores(self, inp, mask):
        return inp


def test_run_external_eval_reports_per_dim_pairwise_and_skill_rank():
    rng = np.random.RandomState(0)
    n = 8
    keys = [f"rec_{i}" for i in range(n)]
    # Monotone skill -> labels increasing; model is identity so predictions track.
    emb = {k: torch.full((NUM_DIMS,), float(i)) for i, k in enumerate(keys)}
    labels = {k: emb[k].tolist() for k in keys}
    skill = {k: float(i) for i, k in enumerate(keys)}
    ds = ExternalEvalDataset(
        name="neuropiano", keys=keys, labels=labels, skill=skill,
    )
    result = run_external_eval(
        _IdentityModel(), ds,
        encode_fn=lambda m, inp, mask: m.encode(inp, mask),
        compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
        predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
        get_input_fn=lambda key: (emb[key].unsqueeze(0), None),
    )
    assert "pairwise_detail" in result
    assert "per_dimension" in result["pairwise_detail"]
    assert result["skill_rank_spearman"] == pytest.approx(1.0)
    assert result["external_source"] == "neuropiano"


# --- D4 stub ------------------------------------------------------------------


def test_run_external_eval_handles_uncovered_dim_nan_labels():
    # NeuroPiano has no pedaling axis -> pedaling label is NaN for every record.
    # The collapse diagnostics (conditional_independence) must not crash on a
    # column that is NaN across all rows.
    from model_improvement.taxonomy import DIMENSIONS
    ped = DIMENSIONS.index("pedaling")
    n = 8
    keys = [f"rec_{i}" for i in range(n)]
    emb = {}
    labels = {}
    skill = {}
    for i, k in enumerate(keys):
        vec = [float(i)] * NUM_DIMS
        vec[ped] = float("nan")  # uncovered dim
        labels[k] = vec
        emb[k] = torch.nan_to_num(torch.tensor(vec))
        skill[k] = float(i)
    ds = ExternalEvalDataset(name="neuropiano", keys=keys, labels=labels, skill=skill)
    result = run_external_eval(
        _IdentityModel(), ds,
        encode_fn=lambda m, inp, mask: m.encode(inp, mask),
        compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
        predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
        get_input_fn=lambda key: (emb[key].unsqueeze(0), None),
    )
    # pedaling is uncovered -> excluded from pairwise (defaults to 0.5).
    assert result["pairwise_detail"]["per_dimension"][ped] == pytest.approx(0.5)
    # a covered dim still reports a real pairwise.
    timing = DIMENSIONS.index("timing")
    assert result["pairwise_detail"]["per_dimension"][timing] == pytest.approx(1.0)


def test_d4_skipped_without_amt_midi():
    ds = ExternalEvalDataset(
        name="pianovam", keys=["a"], labels={"a": [0.5] * NUM_DIMS},
        skill={"a": 3.0}, native_midi={"a": "/tmp/a.mid"},
    )
    out = d4_aria_gt_vs_amt(ds, aria_encode_midi_fn=lambda p: None, amt_midi_dir=None)
    assert out["skipped"].startswith("amt_midi_unavailable")
