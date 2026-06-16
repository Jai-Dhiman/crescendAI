"""Primary scalar (practice + AMT-pseudo-truth, seconds-tolerance) + 4 guards.

G1 teleport (amateur kind), G2 dead-reckon-residual-vs-error AUC (practice kind), G3
silence robustness (silence kind), G4 consecutive-chunk continuity
(practice kind), G5 self-consistency (real_practice kind).

G2 regression threshold scales by max(1.0, min(4.0, sqrt(50/n_chunks))).
G4 regression = drop > 5pp from baseline (higher is better, unlike g1/g3/g5).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkResult:
    kind: str  # "practice" | "amateur" | "silence" | "real_practice"
    piece: Optional[str] = None
    video_id: Optional[str] = None
    start_audio_sec: Optional[float] = None
    predicted_score_sec: Optional[float] = None
    error_seconds: Optional[float] = None
    cost: float = 0.0
    abstain: bool = False
    bar_distance_from_forward: Optional[float] = None
    silence_loud_failure: Optional[bool] = None
    # |predicted_score_sec - dead_reckon_prior_sec|: how far the DTW pulled the
    # endpoint from the tempo-model expectation. This is the g2 confidence signal
    # (large residual => the chunk disagrees with the prior => likely lost).
    dead_reckon_residual_sec: Optional[float] = None


@dataclass
class GuardSet:
    g1: float
    g2: float
    g3: float
    g4: float
    g5: float


@dataclass
class Baseline:
    primary: float
    guards: GuardSet


@dataclass
class Metrics:
    primary: float
    guards: GuardSet
    regressed: list[str]
    g2_threshold_scale: float


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(1 for v in values if v) / max(1, len(values))


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


_G4_TEMPO_MIN = 0.25
_G4_TEMPO_MAX = 1.5


def _g4_continuity(practice: list[ChunkResult]) -> float:
    """For each (piece, video_id), sort by start_audio_sec, count adjacent
    pairs whose IMPLIED LOCAL TEMPO (delta_predicted_score / delta_audio) is
    physically plausible -- in [_G4_TEMPO_MIN, _G4_TEMPO_MAX].
    Returns pct continuous (higher is better). Returns 100.0 if no pairs.

    This is a tempo-agnostic, truth-free teleport/stall detector: a backward
    jump (ratio < 0), a stall (ratio ~ 0, e.g. lock-to-origin), and an
    implausible forward leap (ratio > 1.5) are all flagged, while honest
    tracking of a slow performance (these recordings run ~0.5x notated tempo)
    is NOT. The previous form required |d_pred - d_audio| <= 5s, i.e. it
    assumed score advances at the audio rate (ratio 1.0) and so penalised
    accurate tracking of any sub-0.58x performance -- a mis-specification.
    """
    by_clip: dict[tuple[str, str], list[ChunkResult]] = defaultdict(list)
    for r in practice:
        if r.piece is None or r.video_id is None:
            continue
        if r.start_audio_sec is None or r.predicted_score_sec is None:
            continue
        by_clip[(r.piece, r.video_id)].append(r)
    total = 0
    ok = 0
    for chunks in by_clip.values():
        chunks_sorted = sorted(chunks, key=lambda c: c.start_audio_sec or 0.0)
        for i in range(len(chunks_sorted) - 1):
            a = chunks_sorted[i]
            b = chunks_sorted[i + 1]
            d_audio = (b.start_audio_sec or 0.0) - (a.start_audio_sec or 0.0)
            if d_audio <= 0.0:
                continue
            d_pred = (b.predicted_score_sec or 0.0) - (a.predicted_score_sec or 0.0)
            total += 1
            ratio = d_pred / d_audio
            if _G4_TEMPO_MIN <= ratio <= _G4_TEMPO_MAX:
                ok += 1
    if total == 0:
        return 100.0
    return 100.0 * ok / total


def aggregate(
    results: list[ChunkResult],
    baseline: Baseline,
    *,
    tolerance_s: float = 1.5,
) -> Metrics:
    practice = [r for r in results if r.kind == "practice" and r.error_seconds is not None]
    amateur = [r for r in results if r.kind == "amateur"]
    silence = [r for r in results if r.kind == "silence"]
    real_practice = [r for r in results if r.kind == "real_practice"]

    primary = (
        _pct([abs(r.error_seconds) <= tolerance_s for r in practice])  # type: ignore[arg-type]
        if practice else 0.0
    )

    g1 = (
        _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in amateur])
        if amateur else 0.0
    )
    if practice:
        labels = np.array(
            [abs(r.error_seconds) > tolerance_s for r in practice], dtype=int  # type: ignore[arg-type]
        )
        # g2 = AUC of the dead-reckon residual predicting error. The residual
        # (|pred - tempo-model prior|) is a genuine, per-piece-stable confidence
        # signal (it flags chunks where the DTW disagrees with the prior), unlike
        # raw path cost which is confounded by per-piece chroma difficulty and so
        # tracked piece identity rather than alignment error.
        residuals = np.array(
            [(r.dead_reckon_residual_sec or 0.0) for r in practice], dtype=float
        )
        g2 = _auc(residuals, labels)
    else:
        g2 = 0.5
    g3 = (
        _pct([(r.silence_loud_failure is True) for r in silence])
        if silence else 0.0
    )
    g4 = _g4_continuity(practice)
    g5 = (
        _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in real_practice])
        if real_practice else 0.0
    )

    g2_scale = max(1.0, min(4.0, math.sqrt(50.0 / max(len(practice), 1))))
    guards = GuardSet(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5)
    regressed: list[str] = []
    if primary + 1e-9 < baseline.primary:
        regressed.append("primary")
    if g1 > baseline.guards.g1 + 1.0:
        regressed.append("g1")
    if g2 < baseline.guards.g2 - 0.02 * g2_scale:
        regressed.append("g2")
    if g3 > baseline.guards.g3 + 1.0:
        regressed.append("g3")
    if g4 < baseline.guards.g4 - 5.0:
        regressed.append("g4")
    if g5 > baseline.guards.g5 + 1.0:
        regressed.append("g5")
    return Metrics(
        primary=primary, guards=guards, regressed=regressed,
        g2_threshold_scale=g2_scale,
    )
