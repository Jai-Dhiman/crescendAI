"""Pure G-C churn math (no torch / model / scipy dependency).

Splits the testable reduction out of the model-I/O shell (gc_dynamics_render.py).
Everything here is deterministic and unit-tested in test_gc_churn_metrics.py.

G-C question: how much does the G-B-validated dynamics statistic (whole_piece mean
AMT note-velocity) move when the SAME performance is re-captured under perceptually
neutral recording nuisances (sub-JND gain jitter + high-SNR additive noise)? aria-amt
decodes greedily (transcription.py:649 argmax, model.eval()), so re-transcribing the
identical WAV is a no-op; the churn we want is the substrate's response to nuisance-
equivalent captures, which is exactly what a real re-recording would exercise.

Two reductions:
  - statistic churn : std of whole_piece mean-velocity across the K nuisance variants,
    per clip, then pooled. This is the substrate 1-sigma ON THE STATISTIC.
  - per-note churn  : std of matched-note (variant - base) velocity deltas, pooled.
    Feeds the measurer's per-note substrate sigma, which keeps the /sqrt(N) averaging
    valid for bundles of any size.

The dead-band the frozen router uses (verdict_dispatch.py:94, abs(abs(d)-tau) <=
error_bar) must be >= the measured statistic 1-sigma, so the recommended per-note sigma
is set so that sigma_note / sqrt(N) covers the (possibly correlated) statistic churn.
"""
from __future__ import annotations

import math
from typing import Any


# ---- statistics (hand-rolled to keep this module dependency-free) -----------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sstd(xs: list[float]) -> float:
    """Sample standard deviation (ddof=1). NaN for < 2 points."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _pstd(xs: list[float]) -> float:
    """Population standard deviation (ddof=0)."""
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.9 * len(s)) - 1))]


def mean_velocity(vels: list[float]) -> float:
    """The G-B-validated whole_piece dynamics statistic substrate: mean note velocity."""
    return _mean(vels)


# ---- per-clip reductions ----------------------------------------------------

def clip_statistic_churn(variant_mean_vels: list[float]) -> float:
    """Sample std of whole_piece mean-velocity across nuisance variants (one clip).

    ``variant_mean_vels`` includes the unperturbed base plus each perturbed variant.
    This is the empirical substrate 1-sigma of the statistic for this clip and needs
    NO assumed-noise model -- it is measured directly from re-captured transcriptions.
    """
    return _sstd(variant_mean_vels)


def clip_per_note_churn(base_to_variant_deltas: list[float]) -> float:
    """Population std of matched-note (variant-base) velocity deltas (one clip).

    Each delta is one AMT note's velocity in a nuisance variant minus its velocity in
    the base transcription (matched pitch + nearest onset). Population std because we
    want the spread of the churn itself, not an estimate of a mean.
    """
    if len(base_to_variant_deltas) < 2:
        return float("nan")
    return _pstd(base_to_variant_deltas)


# ---- pooling + dead-band recommendation -------------------------------------

def pool(sigmas: list[float]) -> dict[str, float]:
    """Pool per-clip sigmas (dropping NaN) into median/mean/p90/max summaries."""
    xs = [s for s in sigmas if not math.isnan(s)]
    if not xs:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p90": float("nan"), "max": float("nan")}
    return {"n": len(xs), "median": _median(xs), "mean": _mean(xs),
            "p90": _p90(xs), "max": max(xs)}


def recommend_substrate_terms(
    statistic_churn: dict[str, float],
    per_note_churn: dict[str, float],
) -> dict[str, Any]:
    """Recommend the TWO substrate-error terms to wire into DynamicsMeasurer.

    The re-capture churn has two physically distinct parts, and the measured numbers
    show it is dominated by the correlated one:
      - per-note (independent) : quantization + local transcription noise; averages out
        as sigma_note / sqrt(N). Wired as the measured pooled p90 per-note churn.
      - correlated FLOOR       : a global gain-jitter shift moves every note together,
        so the statistic sigma does NOT shrink with N. The prior sigma_note/sqrt(N)-only
        model has no such term and under-covers at large N. Wired as the measured pooled
        p90 statistic churn -- a flat floor the error bar can never drop below.

    substrate_var = max(sigma_note**2 / N, floor**2). Returns both terms (p90 = the
    conservative "covers 90% of clips" pool; max is reported for context).
    """
    return {
        "sigma_note": per_note_churn["p90"],
        "statistic_floor": statistic_churn["p90"],
        "statistic_floor_max": statistic_churn["max"],
    }
