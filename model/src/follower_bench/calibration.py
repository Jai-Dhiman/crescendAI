"""Confidence-calibration measurement for the HMM follower (#119). Kept out of
metric.py so the position scorer stays follower-agnostic. Samples per-note
confidence onto the same uniform time grid metric.score_clip uses, then reports
Spearman rho(confidence, -|position error|) [the gate] and a risk-coverage curve
[the 'can I trust the cursor' diagnostic]. Confidence between matched notes is a
zero-order hold (last decoded confidence); position uses the follower's own
TrueTrajectory interpolation. Fails loud if any match lacks a confidence."""
from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass

from scipy.stats import spearmanr

from follower_bench.clip_generator import SynthClip
from follower_bench.follower import MatchedNote
from follower_bench.metric import SAMPLE_HZ, _sample_grid, trajectory_from_matches

DEFAULT_COVERAGE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class CalibrationStats:
    """spearman_rho: rank corr between confidence and -|error| (higher = better
    calibrated; the gate). overall_median_error: median |error| over all grid
    samples. risk_coverage: (coverage_fraction, median |error| among the top
    coverage_fraction most-confident samples) points, most-confident first."""
    n_samples: int
    spearman_rho: float
    overall_median_error: float
    risk_coverage: tuple[tuple[float, float], ...]


def _confidence_at(matches: tuple[MatchedNote, ...], t: float) -> float:
    """Zero-order hold: the confidence of the most recent match at or before t
    (or the first match's confidence before it)."""
    times = [m.perf_time for m in matches]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        i = 0
    return float(matches[i].confidence)


def calibration_stats(matches, clip, *, sample_hz: float = SAMPLE_HZ,
                      coverage_fractions=DEFAULT_COVERAGE_FRACTIONS) -> CalibrationStats:
    if not matches:
        raise ValueError("cannot compute calibration on empty matches")
    if any(m.confidence is None for m in matches):
        raise ValueError("every match must carry a confidence to compute calibration")
    est = trajectory_from_matches(matches)
    true = clip.true_trajectory
    t_min, t_max = true.anchors[0][0], true.anchors[-1][0]
    times = _sample_grid(t_min, t_max, sample_hz)
    errors = [abs(est.score_position_at(t) - true.score_position_at(t)) for t in times]
    confs = [_confidence_at(matches, t) for t in times]

    neg_err = [-e for e in errors]
    rho, _ = spearmanr(confs, neg_err)
    if math.isnan(rho):  # zero variance (e.g. all-equal confidence) -> uninformative
        rho = 0.0
    overall_median = statistics.median(errors)

    order = sorted(range(len(times)), key=lambda k: confs[k], reverse=True)
    rc: list[tuple[float, float]] = []
    for frac in sorted(coverage_fractions):
        k = max(1, round(frac * len(order)))
        head = [errors[order[t]] for t in range(k)]
        rc.append((k / len(order), statistics.median(head)))
    return CalibrationStats(
        n_samples=len(times),
        spearman_rho=float(rho),
        overall_median_error=float(overall_median),
        risk_coverage=tuple(rc),
    )
