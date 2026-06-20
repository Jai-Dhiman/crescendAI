from __future__ import annotations
import numpy as np
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


def test_alignment_uncertainty_identity_equals_jitter_sigma() -> None:
    """For an identity alignment, MC uncertainty equals the onset jitter sigma (~0.010s).

    The engine perturbs all anchors by the same scalar per sample, so the interpolated
    estimate at any query is query + jitter; its std is the jitter sigma.
    """
    perf = np.array([0.0, 1.0, 2.0, 3.0])
    score = np.array([0.0, 1.0, 2.0, 3.0])
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    u = e.alignment_uncertainty_sec(perf, score, bar_start_score_sec=1.0)
    assert abs(u - 0.010) < 0.002, f"identity-alignment uncertainty should be ~sigma=0.010s, got {u}"


def test_alignment_uncertainty_independent_of_anchor_density_under_global_shift() -> None:
    """Separate engines, same seed: dense and sparse identity alignments give equal
    uncertainty, because the engine applies a global scalar shift (uncertainty == jitter
    sigma) independent of anchor density. The original <= invariant holds for the right reason.
    """
    perf_dense = np.linspace(0.0, 10.0, 100)
    score_dense = np.linspace(0.0, 10.0, 100)
    perf_sparse = np.array([0.0, 5.0, 10.0])
    score_sparse = np.array([0.0, 5.0, 10.0])
    e_dense = SubstrateErrorEngine(seed=0, n_samples=500)
    e_sparse = SubstrateErrorEngine(seed=0, n_samples=500)
    u_dense = e_dense.alignment_uncertainty_sec(perf_dense, score_dense, 5.0)
    u_sparse = e_sparse.alignment_uncertainty_sec(perf_sparse, score_sparse, 5.0)
    assert abs(u_dense - u_sparse) < 1e-9, (
        f"global-shift engine: dense and sparse should give equal uncertainty, "
        f"got dense={u_dense}, sparse={u_sparse}"
    )
    assert u_dense <= u_sparse + 0.005
