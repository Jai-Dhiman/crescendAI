from __future__ import annotations
from typing import Dict
import jax
import jax.numpy as jnp

def l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    """L2-normalize vectors along a given axis with explicit input validation."""
    if x is None:
        raise ValueError("l2_normalize: input x is None")
    if x.ndim < 1:
        raise ValueError(f"l2_normalize: expected tensor with ndim>=1, got shape {x.shape}")
    denom = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (denom + eps)

def _cross_entropy_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross entropy for integer targets with explicit shape checks."""
    if logits.ndim != 2:
        raise ValueError(f"cross entropy expects 2D logits, got {logits.shape}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets shape mismatch: {targets.shape} vs {logits.shape}")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Gather log prob of the correct class
    gather = jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
    return -jnp.mean(gather)

def info_nce(z1: jnp.ndarray, z2: jnp.ndarray, temperature: float = 0.07) -> Dict[str, jnp.ndarray]:
    """
    Symmetric InfoNCE loss.
    Args:
      z1, z2: [B, D] embeddings (same shape)
      temperature: > 0 scalar
    Returns a dict with:
      - loss: scalar
      - pos_sim_mean: mean cosine similarity of positives (diagonal)
      - neg_sim_mean: mean cosine similarity of off-diagonal negatives
    """
    if z1 is None or z2 is None:
        raise ValueError("info_nce: z1/z2 cannot be None")
    if z1.shape != z2.shape:
        raise ValueError(f"info_nce: z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")
    if z1.ndim != 2:
        raise ValueError(f"info_nce: expects 2D inputs [B,D], got {z1.shape}")
    if temperature <= 0:
        raise ValueError(f"info_nce: temperature must be > 0, got {temperature}")

    z1n = l2_normalize(z1, axis=-1)
    z2n = l2_normalize(z2, axis=-1)

    # Cross-view logits
    logits_ab = (z1n @ z2n.T) / temperature  # [B, B]
    logits_ba = (z2n @ z1n.T) / temperature  # [B, B]
    labels = jnp.arange(z1.shape[0], dtype=jnp.int32)

    loss_ab = _cross_entropy_logits(logits_ab, labels)
    loss_ba = _cross_entropy_logits(logits_ba, labels)
    loss = 0.5 * (loss_ab + loss_ba)

    # Metrics in cosine similarity space (no temperature)
    sims = z1n @ z2n.T  # [B, B]
    pos_sim = jnp.diag(sims)
    pos_sim_mean = jnp.mean(pos_sim)

    # off-diagonal negatives
    B = sims.shape[0]
    neg_mask = jnp.ones_like(sims, dtype=bool) & (~jnp.eye(B, dtype=bool))
    neg_sims = jnp.where(neg_mask, sims, jnp.nan)
    neg_sim_mean = jnp.nanmean(neg_sims)

    return {
        "loss": loss,
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
    }
