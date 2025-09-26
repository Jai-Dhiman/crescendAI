from __future__ import annotations
from typing import Dict
import jax
import jax.numpy as jnp

_MIN_DB = -80.0
_MAX_DB = 0.0

def _clip_db(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(x, _MIN_DB, _MAX_DB)

def _time_mask(x: jnp.ndarray, rng: jax.random.KeyArray, max_width: int) -> jnp.ndarray:
    """Time mask a contiguous range with _MIN_DB. x: [T, M]"""
    T, M = x.shape
    if max_width <= 0:
        return x
    width = jax.random.randint(rng, (), 1, max_width + 1)
    max_start = jnp.maximum(1, T - width)
    start = jax.random.randint(rng, (), 0, max_start)
    end = start + width
    mask = jnp.ones((T,), dtype=bool).at[start:end].set(False)
    return jnp.where(mask[:, None], x, _MIN_DB)

def _freq_mask(x: jnp.ndarray, rng: jax.random.KeyArray, max_width: int) -> jnp.ndarray:
    """Frequency mask a contiguous range with _MIN_DB. x: [T, M]"""
    T, M = x.shape
    if max_width <= 0:
        return x
    width = jax.random.randint(rng, (), 1, max_width + 1)
    max_start = jnp.maximum(1, M - width)
    start = jax.random.randint(rng, (), 0, max_start)
    end = start + width
    mask = jnp.ones((M,), dtype=bool).at[start:end].set(False)
    return jnp.where(mask[None, :], x, _MIN_DB)

def _time_shift_zero_fill(x: jnp.ndarray, shift: int) -> jnp.ndarray:
    """Shift in time by `shift` frames with _MIN_DB fill; no wrap-around. x: [T, M]"""
    T, M = x.shape
    if shift == 0:
        return x
    fill = jnp.full((abs(shift), M), _MIN_DB, dtype=x.dtype)
    if shift > 0:
        return jnp.concatenate([fill, x[:-shift, :]], axis=0)
    else:
        return jnp.concatenate([x[-shift:, :], fill], axis=0)

def conservative_spec_augment(
    mel_db: jnp.ndarray,
    rng: jax.random.KeyArray,
    params: Dict | None = None,
) -> jnp.ndarray:
    """
    Conservative JAX-based spectrogram augmentations in dB space, respecting [-80, 0].
    Accepts [B, T, M] or [T, M]; returns [B, T, M].

    Params (optional; defaults shown):
      - time_mask_width: int = 8
      - freq_mask_width: int = 8
      - num_time_masks: int = 1
      - num_freq_masks: int = 1
      - db_jitter: float = 1.5  (uniform +/- range in dB)
      - db_noise_std: float = 0.2  (Gaussian std in dB)
      - max_time_shift: int = 2
      - p_time_mask, p_freq_mask, p_jitter, p_noise, p_shift: all default 1.0 except p_shift=0.5
    """
    if mel_db is None:
        raise ValueError("conservative_spec_augment: mel_db is None")
    if mel_db.ndim == 2:
        mel_db = mel_db[None, ...]
    if mel_db.ndim != 3:
        raise ValueError(f"conservative_spec_augment: expected [B,T,M] or [T,M], got {mel_db.shape}")

    B, T, M = mel_db.shape
    p = {
        "time_mask_width": 8,
        "freq_mask_width": 8,
        "num_time_masks": 1,
        "num_freq_masks": 1,
        "db_jitter": 1.5,
        "db_noise_std": 0.2,
        "max_time_shift": 2,
        "p_time_mask": 1.0,
        "p_freq_mask": 1.0,
        "p_jitter": 1.0,
        "p_noise": 1.0,
        "p_shift": 0.5,
    }
    if params:
        p.update(params)

    def _augment_one(x: jnp.ndarray, key: jax.random.KeyArray) -> jnp.ndarray:
        tkey, fkey, jkey, nkey, skey = jax.random.split(key, 5)
        out = x

        # Time mask(s)
        def apply_time_mask(carry, k):
            c = carry
            do = jax.random.bernoulli(k, p["p_time_mask"])
            masked = _time_mask(c, k, p["time_mask_width"])
            return jnp.where(do, masked, c), None
        out, _ = jax.lax.scan(apply_time_mask, out, jax.random.split(tkey, p["num_time_masks"]))

        # Freq mask(s)
        def apply_freq_mask(carry, k):
            c = carry
            do = jax.random.bernoulli(k, p["p_freq_mask"])
            masked = _freq_mask(c, k, p["freq_mask_width"])
            return jnp.where(do, masked, c), None
        out, _ = jax.lax.scan(apply_freq_mask, out, jax.random.split(fkey, p["num_freq_masks"]))

        # Small gain jitter (uniform +/- dB)
        if p["db_jitter"] > 0:
            do_j = jax.random.bernoulli(jkey, p["p_jitter"])
            gain = jax.random.uniform(jkey, (), minval=-p["db_jitter"], maxval=p["db_jitter"])
            out = jnp.where(do_j, out + gain, out)

        # Small Gaussian noise in dB
        if p["db_noise_std"] > 0:
            do_n = jax.random.bernoulli(nkey, p["p_noise"])
            noise = jax.random.normal(nkey, out.shape) * p["db_noise_std"]
            out = jnp.where(do_n, out + noise, out)

        # Small time shift with zero-fill
        if p["max_time_shift"] > 0:
            do_s = jax.random.bernoulli(skey, p["p_shift"])
            shift = jax.random.randint(skey, (), -p["max_time_shift"], p["max_time_shift"] + 1)
            shifted = _time_shift_zero_fill(out, int(shift))
            out = jnp.where(do_s, shifted, out)

        return _clip_db(out)

    keys = jax.random.split(rng, B)
    out = jax.vmap(_augment_one)(mel_db, keys)
    return out
