"""Score a symbolic score-follower's estimated trajectory against a
SynthClip's exact ground truth (issue #113): position error, lock rate,
relock latency, and false-jump count, sampled on a common uniform time
grid over the clip's true-trajectory time span. Hides grid construction,
interpolation via TrueTrajectory.score_position_at, event-relative
relock search, and backward-move detection guarded by truth-
monotonicity. Does NOT import follower_bench.follower -- the metric
stays follower-agnostic; trajectory_from_matches is the one adapter
point a caller uses to bridge follow()'s output in.

NOTE ON UNITS: despite the "_beats" suffix on POSITION_TOL_BEATS,
FALSE_JUMP_BEATS, and the *_abs_error_beats fields (inherited naming
from issue #111/#115), the actual shared unit between TrueTrajectory
anchors and MatchedNote.score_position is score-MIDI SECONDS, not
musical beats -- see load_score_notes_from_midi's docstring in
score_notes.py. The metric is self-consistent regardless (both sides
of every comparison share this same actual unit), but a reader tuning
these constants "by ear" should know they are tuning score-MIDI-second
thresholds, not beat counts.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from follower_bench.clip_generator import SynthClip
from follower_bench.trajectory import TrueTrajectory

SAMPLE_HZ = 20.0
POSITION_TOL_BEATS = 0.5
FALSE_JUMP_BEATS = 1.0


@dataclass(frozen=True)
class TrajectoryScore:
    """One clip's measurement: position error (beats), lock rate,
    relock latency per position-changing event (seconds, math.inf if the
    estimate never re-locks before the clip ends), and false-jump
    count."""
    pathology_type: str
    median_abs_error_beats: float
    max_abs_error_beats: float
    lock_rate: float
    relock_latencies_s: tuple[float, ...]
    false_jump_count: int


def _sample_grid(t_min: float, t_max: float, sample_hz: float) -> list[float]:
    """Uniform time grid over [t_min, t_max] at ~sample_hz, always
    including both endpoints exactly."""
    duration = t_max - t_min
    n_steps = max(1, round(duration * sample_hz))
    return [t_min + i * duration / n_steps for i in range(n_steps + 1)]


def score_clip(
    estimated: TrueTrajectory,
    clip: SynthClip,
    *,
    sample_hz: float = SAMPLE_HZ,
    position_tol_beats: float = POSITION_TOL_BEATS,
    false_jump_beats: float = FALSE_JUMP_BEATS,
) -> TrajectoryScore:
    """Score estimated against clip.true_trajectory on a uniform grid
    over the true trajectory's own time span (the clip's real
    duration -- an estimate that starts/ends elsewhere is fairly
    penalized since score_position_at clamps outside its own anchors)."""
    true = clip.true_trajectory
    t_min = true.anchors[0][0]
    t_max = true.anchors[-1][0]
    times = _sample_grid(t_min, t_max, sample_hz)
    true_positions = [true.score_position_at(t) for t in times]
    est_positions = [estimated.score_position_at(t) for t in times]
    errors = [abs(e - t) for e, t in zip(est_positions, true_positions)]

    median_abs_error_beats = statistics.median(errors)
    max_abs_error_beats = max(errors)
    lock_rate = sum(1 for e in errors if e <= position_tol_beats) / len(errors)

    false_jump_count = 0
    for i in range(1, len(times)):
        true_non_decreasing = true_positions[i] >= true_positions[i - 1]
        backward_move = est_positions[i - 1] - est_positions[i]
        if true_non_decreasing and backward_move > false_jump_beats:
            false_jump_count += 1

    relock_latencies_s: list[float] = []
    for event in clip.event_labels:
        if event.from_score_position == event.to_score_position:
            continue
        latency = math.inf
        for t, e in zip(times, errors):
            if t >= event.perf_time and e <= position_tol_beats:
                latency = t - event.perf_time
                break
        relock_latencies_s.append(latency)

    return TrajectoryScore(
        pathology_type=clip.pathology_type,
        median_abs_error_beats=median_abs_error_beats,
        max_abs_error_beats=max_abs_error_beats,
        lock_rate=lock_rate,
        relock_latencies_s=tuple(relock_latencies_s),
        false_jump_count=false_jump_count,
    )


@dataclass(frozen=True)
class AggregateScore:
    """Per-pathology-type rollup of TrajectoryScores."""
    n_clips: int
    median_abs_error_beats: float
    mean_lock_rate: float
    relock_success_rate: float
    median_relock_latency_s: float
    total_false_jumps: int


def aggregate_by_pathology(scores: Iterable[TrajectoryScore]) -> dict[str, AggregateScore]:
    """Group scores by pathology_type and compute per-group stats.
    relock_success_rate is 1.0 for a group with zero position-changing
    events (vacuously perfect); median_relock_latency_s excludes
    math.inf entries, is 0.0 for a group with zero events, and is
    math.inf for a group that had events but none succeeded."""
    by_type: dict[str, list[TrajectoryScore]] = {}
    for score in scores:
        by_type.setdefault(score.pathology_type, []).append(score)

    result: dict[str, AggregateScore] = {}
    for pathology_type, group in by_type.items():
        all_latencies = [lat for s in group for lat in s.relock_latencies_s]
        finite_latencies = [lat for lat in all_latencies if math.isfinite(lat)]

        if not all_latencies:
            relock_success_rate = 1.0
            median_relock_latency_s = 0.0
        else:
            relock_success_rate = len(finite_latencies) / len(all_latencies)
            median_relock_latency_s = (
                statistics.median(finite_latencies) if finite_latencies else math.inf
            )

        result[pathology_type] = AggregateScore(
            n_clips=len(group),
            median_abs_error_beats=statistics.median(s.median_abs_error_beats for s in group),
            mean_lock_rate=statistics.mean(s.lock_rate for s in group),
            relock_success_rate=relock_success_rate,
            median_relock_latency_s=median_relock_latency_s,
            total_false_jumps=sum(s.false_jump_count for s in group),
        )
    return result
