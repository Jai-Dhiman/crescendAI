"""Three deterministic corruption functions for the white-noise ablation eval."""
from __future__ import annotations
import random
from typing import Any, Literal

Mode = Literal["shuffle", "marginal", "flip"]

SCALER_MEAN: dict[str, float] = {
    "dynamics": 0.545,
    "timing": 0.4848,
    "pedaling": 0.4594,
    "articulation": 0.5369,
    "phrasing": 0.5188,
    "interpretation": 0.5064,
}


def corrupt(
    top_moments: list[dict[str, Any]],
    mode: Mode,
    seed: int,
    all_top_moments: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if mode == "shuffle":
        return _shuffle(top_moments, seed, all_top_moments)
    if mode == "marginal":
        return _marginal(top_moments, seed, all_top_moments)
    if mode == "flip":
        return _flip(top_moments)
    raise ValueError(f"unknown mode: {mode}")


def _shuffle(src, seed, corpus):
    others = [tm for tm in corpus if tm is not src]
    if not others:
        raise ValueError("cannot shuffle: corpus has no other sessions")
    rng = random.Random(seed)
    return rng.choice(others)


def _marginal(src, seed, corpus):
    rng = random.Random(seed)
    pools: dict[str, list[float]] = {}
    for tm in corpus:
        for moment in tm:
            pools.setdefault(moment["dimension"], []).append(float(moment["score"]))
    out = []
    for moment in src:
        dim = moment["dimension"]
        pool = pools.get(dim, [moment["score"]])
        new_score = rng.choice(pool)
        out.append({
            "dimension": dim,
            "score": new_score,
            "deviation_from_mean": round(new_score - SCALER_MEAN.get(dim, 0.5), 3),
            "direction": "above_average" if new_score > SCALER_MEAN.get(dim, 0.5) else "below_average",
        })
    return out


def _flip(src):
    out = []
    for moment in src:
        new_score = round(1.0 - float(moment["score"]), 4)
        dim = moment["dimension"]
        out.append({
            "dimension": dim,
            "score": new_score,
            "deviation_from_mean": round(new_score - SCALER_MEAN.get(dim, 0.5), 3),
            "direction": "above_average" if new_score > SCALER_MEAN.get(dim, 0.5) else "below_average",
        })
    return out
