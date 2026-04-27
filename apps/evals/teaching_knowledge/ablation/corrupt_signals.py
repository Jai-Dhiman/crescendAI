"""Three deterministic corruption functions for the white-noise ablation eval."""
from __future__ import annotations
import random
from typing import Any

Mode = str  # "shuffle" | "marginal" | "flip"


def corrupt(
    top_moments: list[dict[str, Any]],
    mode: Mode,
    seed: int,
    all_top_moments: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if mode == "shuffle":
        return _shuffle(top_moments, seed, all_top_moments)
    raise ValueError(f"unknown mode: {mode}")


def _shuffle(src, seed, corpus):
    others = [tm for tm in corpus if tm is not src]
    if not others:
        raise ValueError("cannot shuffle: corpus has no other sessions")
    rng = random.Random(seed)
    return rng.choice(others)
