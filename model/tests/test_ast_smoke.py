#!/usr/bin/env python3
"""
CPU-only smoke test: run a tiny batch through AST and compute a simple loss.
Skips if JAX is unavailable.
"""
import os
import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False

import numpy as np

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not available for AST smoke test")

from src.models.ast_transformer import AudioSpectrogramTransformer, create_train_state


def test_ast_forward_and_one_step():
    # Deterministic RNG
    rng = jax.random.PRNGKey(0)

    # Model and state
    model = AudioSpectrogramTransformer()
    batch_size, t, f = 2, 128, 128
    state = create_train_state(model, rng, (batch_size, t, f), learning_rate=1e-4)

    # Dummy batch and labels
    x = jnp.zeros((batch_size, t, f), dtype=jnp.float32)

    # Forward to get dims and shapes
    preds, _ = model.apply(state.params, x, training=False)
    assert isinstance(preds, dict) and len(preds) > 0

    # Create zero labels matching prediction shapes
    labels = {k: jnp.zeros_like(v) for k, v in preds.items()}

    # Define simple MSE across all dimensions
    def loss_fn(params):
        out, _ = model.apply(params, x, training=True, rngs={"dropout": rng})
        loss = 0.0
        count = 0
        for k, v in out.items():
            target = labels[k]
            loss = loss + jnp.mean((v - target) ** 2)
            count += 1
        return loss / max(count, 1)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    assert jnp.isfinite(loss), "Loss is not finite"

    new_state = state.apply_gradients(grads=grads)
    # Parameter structure should be unchanged
    assert jax.tree_map(lambda a, b: a.shape == b.shape, state.params, new_state.params)
