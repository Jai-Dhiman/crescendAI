#!/usr/bin/env python3
import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not available for SSL loss tests")

from src.objectives.ssl_losses import info_nce, l2_normalize


def test_info_nce_identical_views_low_loss_and_high_pos_sim():
    key = jax.random.PRNGKey(0)
    B, D = 8, 16
    z = jax.random.normal(key, (B, D))
    z = l2_normalize(z, axis=-1)
    out = info_nce(z, z, temperature=0.07)
    # Positive sims near 1.0; allow tolerance
    assert float(out["pos_sim_mean"]) > 0.9
    # Loss should be significantly less than log(B)
    assert float(out["loss"]) < 0.5 * float(jnp.log(B))


def test_info_nce_negatives_affect_loss():
    key = jax.random.PRNGKey(42)
    B, D = 8, 16
    k1, k2 = jax.random.split(key)
    z1 = jax.random.normal(k1, (B, D))
    z2 = z1 + 0.01 * jax.random.normal(k2, (B, D))  # close positives
    out_close = info_nce(z1, z2, temperature=0.07)

    # Make positives random (hard)
    k3, _ = jax.random.split(k2)
    z2_far = jax.random.normal(k3, (B, D))
    out_far = info_nce(z1, z2_far, temperature=0.07)

    assert float(out_close["loss"]) < float(out_far["loss"]) 
    assert float(out_close["pos_sim_mean"]) > float(out_far["pos_sim_mean"]) 
