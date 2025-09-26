import jax
import jax.numpy as jnp
import optax

@jax.jit(donate_argnums=(0,))
def ultra_small_train_step_masked(train_state_obj, batch_specs, pad_masks, dropout_rng, stochastic_rng):
    """Training step that excludes padded patches from all loss components.

    Args:
        train_state_obj: flax.training.train_state.TrainState
        batch_specs: jnp.ndarray [B, T, F]
        pad_masks: jnp.ndarray [B, T, F] booleans (True means padded)
        dropout_rng, stochastic_rng: PRNGKeys
    Returns:
        new_train_state, metrics (dict)
    """
    patch_size = 16

    def loss_fn(params):
        features = train_state_obj.apply_fn(
            params, batch_specs,
            training=True,
            rngs={'dropout': dropout_rng, 'stochastic_depth': stochastic_rng}
        )  # [B, P, D]

        # Build patch weights from pad_masks ([B, T, F])
        b, t, f = pad_masks.shape
        pt = t // patch_size
        pf = f // patch_size
        pm = jnp.reshape(pad_masks, (b, pt, patch_size, pf, patch_size))
        pm = jnp.any(pm, axis=(2, 4))                   # [B, pt, pf]
        patch_mask = jnp.reshape(pm, (b, pt * pf))      # True = padded
        weights = (1.0 - patch_mask.astype(features.dtype))  # [B, P]
        weights_sum = jnp.clip(jnp.sum(weights, axis=1, keepdims=True), a_min=1.0)

        # 1) Consistency (masked variance across patches)
        patch_mean = jnp.sum(features * weights[:, :, None], axis=1, keepdims=True) / weights_sum[:, :, None]
        diff = features - patch_mean
        consistency_loss = jnp.sum((diff * diff) * weights[:, :, None]) / jnp.sum(weights)

        # 2) Magnitude regularization over valid tokens
        magnitude_loss = jnp.sum((features * features) * weights[:, :, None]) / jnp.sum(weights)

        # 3) Diversity (variance across batch and patches, masked)
        weights_b = weights[:, :, None]
        denom = jnp.sum(weights_b) + 1e-8
        mu_feat = jnp.sum(features * weights_b, axis=(0, 1)) / denom
        var_feat = jnp.sum(((features - mu_feat) ** 2) * weights_b, axis=(0, 1)) / denom
        feature_std = jnp.sqrt(var_feat + 1e-8)
        diversity_loss = -jnp.mean(jnp.log(feature_std + 1e-8))

        # Anti-collapse metrics (Priority 5)
        embedding_variance = jnp.mean(var_feat)
        collapse_indicator = jnp.mean((var_feat < 1e-6).astype(features.dtype))

        # 4) Contrastive across samples using masked global features
        batch_size = features.shape[0]
        if batch_size > 1:
            global_features = (jnp.sum(features * weights[:, :, None], axis=1) / weights_sum)  # [B, D]
            sims = jnp.dot(global_features, global_features.T)
            norms = jnp.sqrt(jnp.sum(global_features**2, axis=1))
            sims = sims / (norms[:, None] * norms[None, :] + 1e-8)
            mask = 1.0 - jnp.eye(batch_size)
            contrastive_loss = jnp.sum(sims * mask) / jnp.sum(mask)
        else:
            contrastive_loss = jnp.array(0.0, dtype=features.dtype)

        total_loss = (0.1 * consistency_loss +
                      0.05 * magnitude_loss +
                      0.05 * diversity_loss +
                      0.1  * contrastive_loss)

    metrics = {
        'total_loss': total_loss,
        'consistency_loss': consistency_loss,
        'magnitude_loss': magnitude_loss,
        'diversity_loss': diversity_loss,
        'contrastive_loss': contrastive_loss,
        'embedding_variance': embedding_variance,
        'collapse_indicator': collapse_indicator,
        'output_mean': jnp.mean(features * (weights[:, :, None] > 0)),
        'output_std': jnp.std(features * (weights[:, :, None] > 0)),
        'valid_patch_ratio': jnp.mean(1.0 - patch_mask.astype(features.dtype)),
    }
    return total_loss, metrics

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state_obj.params)
    grad_norm = optax.global_norm(grads)
    new_train_state = train_state_obj.apply_gradients(grads=grads)

    try:
        current_lr = new_train_state.opt_state[1].hyperparams['learning_rate']
    except (AttributeError, KeyError, IndexError):
        current_lr = jnp.array(2e-5)

    metrics.update({
        'grad_norm': grad_norm,
        'learning_rate': current_lr,
    })
    return new_train_state, metrics

# --- Priority 4 helper: reusable masked objective for validation parity ---
# Note: Kept weights identical to training composition; Priority 5 may adjust.
CONSISTENCY_W = 0.1
MAGNITUDE_W = 0.05
DIVERSITY_W = 0.05
CONTRASTIVE_W = 0.1


def _mask_to_patch_weights(pad_masks: jnp.ndarray, patch_size: int, dtype=jnp.float32):
    b, t, f = pad_masks.shape
    pt = t // patch_size
    pf = f // patch_size
    pm = jnp.reshape(pad_masks, (b, pt, patch_size, pf, patch_size))
    pm = jnp.any(pm, axis=(2, 4))
    patch_mask = jnp.reshape(pm, (b, pt * pf))
    weights = (1.0 - patch_mask.astype(dtype))
    return weights, patch_mask


def compute_masked_objective(features: jnp.ndarray, pad_masks: jnp.ndarray, patch_size: int = 16):
    """Compute masked objective components and total loss.

    Returns a dict with: total_loss, consistency_loss, magnitude_loss,
    diversity_loss, contrastive_loss, valid_patch_ratio, output_mean, output_std
    """
    weights, patch_mask = _mask_to_patch_weights(pad_masks, patch_size, dtype=features.dtype)
    weights_sum = jnp.clip(jnp.sum(weights, axis=1, keepdims=True), a_min=1.0)

    # Consistency
    patch_mean = jnp.sum(features * weights[:, :, None], axis=1, keepdims=True) / weights_sum[:, :, None]
    diff = features - patch_mean
    consistency_loss = jnp.sum((diff * diff) * weights[:, :, None]) / jnp.sum(weights)

    # Magnitude
    magnitude_loss = jnp.sum((features * features) * weights[:, :, None]) / jnp.sum(weights)

    # Diversity
    weights_b = weights[:, :, None]
    denom = jnp.sum(weights_b) + 1e-8
    mu_feat = jnp.sum(features * weights_b, axis=(0, 1)) / denom
    var_feat = jnp.sum(((features - mu_feat) ** 2) * weights_b, axis=(0, 1)) / denom
    feature_std = jnp.sqrt(var_feat + 1e-8)
    diversity_loss = -jnp.mean(jnp.log(feature_std + 1e-8))

    # Contrastive
    batch_size = features.shape[0]
    def _contrastive(feats, wts, wsum):
        global_features = (jnp.sum(feats * wts[:, :, None], axis=1) / wsum)
        sims = jnp.dot(global_features, global_features.T)
        norms = jnp.sqrt(jnp.sum(global_features**2, axis=1))
        sims = sims / (norms[:, None] * norms[None, :] + 1e-8)
        mask = 1.0 - jnp.eye(global_features.shape[0], dtype=features.dtype)
        return jnp.sum(sims * mask) / jnp.sum(mask)

    contrastive_loss = jax.lax.cond(
        batch_size > 1,
        lambda _: _contrastive(features, weights, weights_sum),
        lambda _: jnp.array(0.0, dtype=features.dtype),
        operand=None,
    )

    total_loss = (CONSISTENCY_W * consistency_loss +
                  MAGNITUDE_W   * magnitude_loss   +
                  DIVERSITY_W   * diversity_loss   +
                  CONTRASTIVE_W * contrastive_loss)

    metrics = {
        'total_loss': total_loss,
        'consistency_loss': consistency_loss,
        'magnitude_loss': magnitude_loss,
        'diversity_loss': diversity_loss,
        'contrastive_loss': contrastive_loss,
        'embedding_variance': jnp.mean(var_feat),
        'collapse_indicator': jnp.mean((var_feat < 1e-6).astype(features.dtype)),
        'output_mean': jnp.mean(features * (weights[:, :, None] > 0)),
        'output_std': jnp.std(features * (weights[:, :, None] > 0)),
        'valid_patch_ratio': jnp.mean(1.0 - patch_mask.astype(features.dtype)),
    }
    return metrics

# === Phase 2: InfoNCE objective integration ===
from typing import Dict as _Dict, Any as _Any, Tuple as _Tuple
from src.objectives.ssl_losses import info_nce as _info_nce
from src.augment.piano_augment import conservative_spec_augment as _conservative_spec_augment


def _weighted_mean_pool(features: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    features: [B, N, D] or [B, T, M, D]
    weights:  [B, N] or [B, T, M] (1 valid, 0 pad)
    returns:  [B, D]
    """
    if features.ndim == 4 and weights.ndim == 3:
        B, T, F, D = features.shape
        feats = features.reshape(B, T * F, D)
        w = weights.reshape(B, T * F)
    elif features.ndim == 3 and weights.ndim == 2:
        feats, w = features, weights
    else:
        raise ValueError(f"_weighted_mean_pool: incompatible shapes {features.shape} and {weights.shape}")
    wsum = jnp.sum(w, axis=1, keepdims=True) + 1e-8
    return jnp.sum(feats * w[..., None], axis=1) / wsum


@jax.jit(donate_argnums=(0,))
def ssl_infonce_train_step(
    train_state_obj,
    batch_specs: jnp.ndarray,
    pad_masks: jnp.ndarray,
    dropout_rng: jax.random.KeyArray,
    stochastic_rng: jax.random.KeyArray,
    *,
    temperature: float = 0.07,
    augment: bool = True,
    aug_params: _Dict[str, _Any] | None = None,
) -> _Tuple[any, _Dict[str, jnp.ndarray]]:
    """
    InfoNCE-based SSL train step using two views (orig + conservative spectrogram augment).

    Args:
      train_state_obj: flax.training.train_state.TrainState
      batch_specs: [B, T, F] in dB [-80, 0]
      pad_masks:  [B, T, F] booleans (True means padded) used for pooling
    Returns:
      new_train_state, metrics dict
    """
    if batch_specs is None or batch_specs.ndim != 3:
        raise ValueError(f"ssl_infonce_train_step: expected batch_specs [B,T,F], got {None if batch_specs is None else batch_specs.shape}")
    if pad_masks is None or pad_masks.shape[:2] != batch_specs.shape[:2]:
        raise ValueError("ssl_infonce_train_step: pad_masks must align with [B,T,F] of batch_specs")

    # Prepare augmentation
    aug_rng, enc_rng = jax.random.split(stochastic_rng)
    view2 = _conservative_spec_augment(batch_specs, aug_rng, aug_params or {}) if augment else batch_specs

    patch_size = 16
    # Build patch weights from pad_masks ([B, T, F]) reusing the existing logic
    b, t, f = pad_masks.shape
    pt = t // patch_size
    pf = f // patch_size
    pm = jnp.reshape(pad_masks, (b, pt, patch_size, pf, patch_size))
    pm = jnp.any(pm, axis=(2, 4))                   # [B, pt, pf]
    patch_mask = jnp.reshape(pm, (b, pt * pf))      # True = padded
    weights = (1.0 - patch_mask.astype(batch_specs.dtype))  # [B, P]

    def loss_fn(params):
        # Encode both views
        # Note: Support models that return dicts (e.g., SSAST) and raw features.
        out1 = train_state_obj.apply_fn(params, batch_specs, enc_rng, training=True, rngs={'dropout': dropout_rng})
        out2 = train_state_obj.apply_fn(params, view2,       enc_rng, training=True, rngs={'dropout': dropout_rng})
        feats1 = out1.get('encoder_embeddings', out1) if isinstance(out1, dict) else out1
        feats2 = out2.get('encoder_embeddings', out2) if isinstance(out2, dict) else out2
        # Pool to global representations
        z1 = _weighted_mean_pool(feats1, weights)
        z2 = _weighted_mean_pool(feats2, weights)
        metrics = _info_nce(z1, z2, temperature=temperature)
        return metrics['loss'], metrics

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state_obj.params)
    grad_norm = optax.global_norm(grads)
    new_train_state = train_state_obj.apply_gradients(grads=grads)

    out = {
        'total_loss': loss_val,
        'info_nce_loss': metrics['loss'],
        'pos_sim_mean': metrics['pos_sim_mean'],
        'neg_sim_mean': metrics['neg_sim_mean'],
        'grad_norm': grad_norm,
        'valid_patch_ratio': jnp.mean(weights),
        'temperature': jnp.array(temperature, dtype=jnp.float32),
        'batch_size': jnp.array(batch_specs.shape[0], dtype=jnp.int32),
    }
    return new_train_state, out


def ssl_train_step_dispatch(objective: str):
    """Select SSL train step function by objective name."""
    if objective == 'infonce':
        return ssl_infonce_train_step
    elif objective == 'repulsive':
        return ultra_small_train_step_masked
    else:
        raise ValueError(f"Unknown ssl objective: {objective}")
