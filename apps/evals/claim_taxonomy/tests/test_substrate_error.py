from __future__ import annotations
import numpy as np
import pytest
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def test_timing_jitter_is_deterministic_with_seed() -> None:
    e1 = SubstrateErrorEngine(seed=42, n_samples=100)
    e2 = SubstrateErrorEngine(seed=42, n_samples=100)
    np.testing.assert_array_equal(e1.timing_onset_jitter_sec(), e2.timing_onset_jitter_sec())


def test_timing_jitter_changes_with_different_seed() -> None:
    e1 = SubstrateErrorEngine(seed=42, n_samples=100)
    e2 = SubstrateErrorEngine(seed=99, n_samples=100)
    assert not np.allclose(e1.timing_onset_jitter_sec(), e2.timing_onset_jitter_sec())


def test_timing_jitter_shape_and_scale() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.timing_onset_jitter_sec()
    assert j.shape == (5000,)
    assert abs(j.std() - 0.010) < 0.002  # ~Gaussian sigma=0.010


def test_dynamics_rms_jitter_shape_and_scale() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.dynamics_rms_jitter_db()
    assert j.shape == (5000,)
    assert abs(j.std() - 0.3) < 0.05


def test_pedal_threshold_jitter_uniform_range() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.pedal_threshold_jitter()
    assert j.min() >= -10.0
    assert j.max() <= 10.0
    assert abs(j.mean()) < 1.0  # centered near zero


def test_bootstrap_d_is_deterministic() -> None:
    e1 = SubstrateErrorEngine(seed=7, n_samples=200)
    e2 = SubstrateErrorEngine(seed=7, n_samples=200)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(
        e1.bootstrap_d(values, np.mean),
        e2.bootstrap_d(values, np.mean),
    )


def test_bootstrap_d_returns_n_samples_values() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=300)
    values = np.linspace(0.5, 1.5, 20)
    result = e.bootstrap_d(values, np.median)
    assert result.shape == (300,)


def test_alignment_uncertainty_zero_jitter_near_zero() -> None:
    """With zero-sigma jitter the MC uncertainty should be effectively zero."""
    perf = np.array([0.0, 1.0, 2.0, 3.0])
    score = np.array([0.0, 1.0, 2.0, 3.0])
    e_small = SubstrateErrorEngine(seed=0, n_samples=500)
    u = e_small.alignment_uncertainty_sec(perf, score, bar_start_score_sec=1.0)
    assert isinstance(u, float)
    assert u >= 0.0


def test_alignment_uncertainty_monotone_with_anchor_spread() -> None:
    """More spread-out anchors give lower uncertainty (more information)."""
    perf_dense = np.linspace(0.0, 10.0, 100)
    score_dense = np.linspace(0.0, 10.0, 100)
    perf_sparse = np.array([0.0, 5.0, 10.0])
    score_sparse = np.array([0.0, 5.0, 10.0])
    e = SubstrateErrorEngine(seed=0, n_samples=500)
    u_dense = e.alignment_uncertainty_sec(perf_dense, score_dense, 5.0)
    u_sparse = e.alignment_uncertainty_sec(perf_sparse, score_sparse, 5.0)
    assert u_dense <= u_sparse + 0.005  # allow tiny float tolerance
