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


# --- Chunk-level corruption (DO-path ablation, #28) ---------------------------
# The legacy ablation corrupted Python-framed `top_moments`. The DO computes its
# OWN top moments from the per-chunk MuQ `predictions`, so to ablate the signal
# the real teacher sees we corrupt the chunk predictions BEFORE they are replayed
# as eval_chunk messages. midi_notes / pedal_events are preserved verbatim --
# only the MuQ score vector (the teaching-moment driver) is perturbed.


def corrupt_chunks(
    chunks: list[dict[str, Any]],
    mode: Mode,
    seed: int,
    all_chunks: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Corrupt a recording's chunk predictions for one ablation condition.

    `all_chunks` is the corpus of every recording's chunk list (used by shuffle to
    borrow another recording's signal and by marginal to draw from the per-dim pool).
    """
    if mode == "shuffle":
        return _shuffle_chunks(chunks, seed, all_chunks)
    if mode == "marginal":
        return _marginal_chunks(chunks, seed, all_chunks)
    if mode == "flip":
        return _flip_chunks(chunks)
    raise ValueError(f"unknown mode: {mode}")


def _chunk_with_predictions(
    chunk: dict[str, Any], predictions: dict[str, float]
) -> dict[str, Any]:
    """Copy a chunk, swapping only its `predictions` (other fields preserved)."""
    out = dict(chunk)
    out["predictions"] = predictions
    return out


def _shuffle_chunks(src, seed, corpus):
    others = [c for c in corpus if c is not src]
    if not others:
        raise ValueError("cannot shuffle: corpus has no other recordings")
    rng = random.Random(seed)
    return rng.choice(others)


def _marginal_chunks(src, seed, corpus):
    rng = random.Random(seed)
    pools: dict[str, list[float]] = {}
    for chunk_list in corpus:
        for chunk in chunk_list:
            for dim, val in (chunk.get("predictions") or {}).items():
                if val is not None:
                    pools.setdefault(dim, []).append(float(val))
    out = []
    for chunk in src:
        preds = chunk.get("predictions") or {}
        new_preds = {
            dim: rng.choice(pools.get(dim, [float(val)]))
            for dim, val in preds.items()
            if val is not None
        }
        out.append(_chunk_with_predictions(chunk, new_preds))
    return out


def _flip_chunks(src):
    out = []
    for chunk in src:
        preds = chunk.get("predictions") or {}
        new_preds = {
            dim: round(1.0 - float(val), 4)
            for dim, val in preds.items()
            if val is not None
        }
        out.append(_chunk_with_predictions(chunk, new_preds))
    return out
