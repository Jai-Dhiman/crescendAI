#!/usr/bin/env python3
"""
Local window head(s) for time-local proxy predictions and simple robust pooling.

- LocalWindowHead: maps encoder patch tokens [B, P, D] to per-window K-dim outputs.
- robust_pool: aggregates per-window outputs to global descriptors.

Note: K defaults to 5 basic proxies to start with: [rms_dr, onset_density, centroid_mean,
rolloff_mean, tempo_bpm]. Extend as needed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax.numpy as jnp
from flax import linen as nn


class LocalWindowHead(nn.Module):
    """Light-weight MLP head applied to each token/window independently."""
    out_dims: int = 5
    hidden_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        tokens: [B, P, D]
        returns: [B, P, out_dims]
        """
        x = nn.Dense(self.hidden_dim, name="lw_fc1")(tokens)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.hidden_dim // 2, name="lw_fc2")(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        out = nn.Dense(self.out_dims, name="lw_out")(x)
        return out


def robust_pool(window_outputs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Aggregate per-window outputs to global descriptors with robust statistics.

    window_outputs: [B, P, K]
    returns dict of global stats per channel: median, mean, std
    """
    if window_outputs.ndim != 3:
        raise ValueError(f"robust_pool expects [B,P,K], got {window_outputs.shape}")
    median = jnp.median(window_outputs, axis=1)  # [B, K]
    mean = jnp.mean(window_outputs, axis=1)      # [B, K]
    std = jnp.std(window_outputs, axis=1)        # [B, K]
    return {"median": median, "mean": mean, "std": std}
