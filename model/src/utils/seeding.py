#!/usr/bin/env python3
"""
Deterministic seeding utilities for Python, NumPy, and JAX.
- Explicit exceptions preferred over silent fallbacks.
- Use set_seed() at program start to ensure reproducibility.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

try:
    import jax
except Exception:  # JAX may be optional in some environments
    jax = None  # type: ignore


def set_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Set global seeds for Python, NumPy, and JAX (if available).

    Args:
        seed: Integer seed value
        deterministic: If True, set flags to encourage deterministic behavior

    Returns:
        Dict with keys: {"seed": int, "jax_key": jax.random.PRNGKey or None}
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        # Encourage determinism in XLA/JAX (best-effort)
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_reductions")
        os.environ.setdefault("JAX_DISABLE_MOST_OPTIMIZATIONS", "true")

    key = None
    if jax is not None:
        try:
            key = jax.random.PRNGKey(seed)
        except Exception as e:
            raise RuntimeError(f"Failed to create JAX PRNGKey: {e}")

    return {"seed": seed, "jax_key": key}


def split_key(key, num: int = 1):
    """
    Split a JAX PRNGKey into (new_key, [num keys]).

    Raises ImportError if JAX is not available.
    """
    if jax is None:
        raise ImportError("JAX not available for key splitting.")
    if num < 1:
        raise ValueError("num must be >= 1.")
    keys = jax.random.split(key, num + 1)
    return keys[0], keys[1:]
