"""Primary scalar + 5 guards + baseline-delta diff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkResult:
    kind: str  # "gold" | "amateur" | "silence" | "synthetic_practice" | "real_practice"
    error_frames: Optional[float]  # vs gold (gold only)
    cost: float
    abstain: bool
    bar_distance_from_forward: Optional[float] = None  # bars (amateur/real_practice)
    silence_loud_failure: Optional[bool] = None  # G3 phase-1
    stitch_error_frames: Optional[float] = None  # G4


@dataclass
class GuardSet:
    g1: float; g2: float; g3: float; g4: float; g5: float


@dataclass
class Baseline:
    primary: float
    guards: GuardSet


@dataclass
class Metrics:
    primary: float
    guards: GuardSet
    regressed: list[str]


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(1 for v in values if v) / max(1, len(values))


def aggregate(
    results: list[ChunkResult], baseline: Baseline,
    frame_rate_hz: float, tolerance_ms: float,
) -> Metrics:
    tol_frames = (tolerance_ms / 1000.0) * frame_rate_hz
    gold = [r for r in results if r.kind == "gold" and r.error_frames is not None]
    amateur = [r for r in results if r.kind == "amateur"]
    silence = [r for r in results if r.kind == "silence"]
    synth = [r for r in results if r.kind == "synthetic_practice" and r.stitch_error_frames is not None]
    real_practice = [r for r in results if r.kind == "real_practice"]

    primary = _pct([abs(r.error_frames) <= tol_frames for r in gold]) if gold else 0.0

    g1 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in amateur]) if amateur else 0.0
    if gold:
        labels = np.array([abs(r.error_frames) > tol_frames for r in gold], dtype=int)
        costs = np.array([r.cost for r in gold], dtype=float)
        g2 = _auc(costs, labels)
    else:
        g2 = 0.5
    g3 = _pct([(r.silence_loud_failure is True) for r in silence]) if silence else 0.0
    g4 = _pct([abs(r.stitch_error_frames) <= tol_frames for r in synth]) if synth else 0.0
    g5 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in real_practice]) if real_practice else 0.0

    guards = GuardSet(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5)
    regressed: list[str] = []
    if primary + 1e-9 < baseline.primary:
        regressed.append("primary")
    if g1 > baseline.guards.g1 + 1.0:
        regressed.append("g1")
    if g2 < baseline.guards.g2 - 0.02:
        regressed.append("g2")
    if g3 < baseline.guards.g3 - 1.0:
        regressed.append("g3")
    if g4 < baseline.guards.g4 - 1.0:
        regressed.append("g4")
    if g5 > baseline.guards.g5 + 1.0:
        regressed.append("g5")
    return Metrics(primary=primary, guards=guards, regressed=regressed)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.5
    order = np.argsort(-scores)
    labels = labels[order]
    pos = int(labels.sum())
    neg = len(labels) - pos
    cum_pos = 0
    auc = 0.0
    for y in labels:
        if y == 1:
            cum_pos += 1
        else:
            auc += cum_pos
    return float(auc / (pos * neg))
