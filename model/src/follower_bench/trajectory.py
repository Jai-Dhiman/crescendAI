# model/src/follower_bench/trajectory.py
"""Exact ground-truth mapping from performance time (seconds) to score
position (score beats). Piecewise-linear between explicit anchor points.
Anchors are (perf_time_seconds, score_beat_position) pairs, sorted
ascending by perf_time. A hard-splice discontinuity (repeat/jump/restart)
is represented as two anchors separated by a fixed, tiny time epsilon
with different score positions -- a near-instant transition rather than a
gradual ramp -- so a discontinuity always resolves within
DISCONTINUITY_EPS_S seconds of the injected event's perf_time.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from follower_bench.asap_alignment import ClipAlignment
from follower_bench.segments import Segment

DISCONTINUITY_EPS_S = 1e-6


@dataclass(frozen=True)
class TrueTrajectory:
    """Piecewise-linear score-position(perf_time) ground truth."""
    anchors: tuple[tuple[float, float], ...]

    def score_position_at(self, t: float) -> float:
        """Interpolate score position at perf-time t seconds. Clamps to
        the first/last anchor's score position outside the anchors' time
        range."""
        times = [a[0] for a in self.anchors]
        positions = [a[1] for a in self.anchors]
        if t <= times[0]:
            return positions[0]
        if t >= times[-1]:
            return positions[-1]
        i = bisect.bisect_right(times, t) - 1
        t0, p0 = times[i], positions[i]
        t1, p1 = times[i + 1], positions[i + 1]
        if t1 == t0:
            return p0
        frac = (t - t0) / (t1 - t0)
        return p0 + frac * (p1 - p0)

    def is_monotonic_non_decreasing(self) -> bool:
        """True iff score position never decreases as perf_time
        advances."""
        positions = [a[1] for a in self.anchors]
        return all(b >= a for a, b in zip(positions, positions[1:]))


def from_alignment(alignment: ClipAlignment) -> TrueTrajectory:
    """Build the clean (unmodified) trajectory directly from an ASAP
    beat alignment: performance_beats <-> midi_score_beats, zipped
    index-for-index."""
    anchors = tuple(zip(alignment.performance_beats, alignment.midi_score_beats))
    return TrueTrajectory(anchors=anchors)


def build_trajectory_from_segments(
    clean_traj: TrueTrajectory, segments: list[Segment]
) -> TrueTrajectory:
    """Build a spliced trajectory by replaying clean_traj's own anchors
    through each Segment's affine (perf_time -> dst_time) map, in
    destination order. Where consecutive segments are NOT contiguous in
    source time (a hard-splice jump), the later segment's start anchor is
    offset by DISCONTINUITY_EPS_S so the transition resolves as a sharp
    but well-defined ramp rather than an undefined vertical step.
    """
    anchors: list[tuple[float, float]] = []
    prev_seg: Segment | None = None
    for seg in segments:
        contiguous = prev_seg is None or abs(seg.src_start - prev_seg.src_end) < 1e-9
        start_dst = seg.dst_start if contiguous else seg.dst_start + DISCONTINUITY_EPS_S
        anchors.append((start_dst, clean_traj.score_position_at(seg.src_start)))
        for src_t, pos in clean_traj.anchors:
            if seg.src_start < src_t < seg.src_end:
                dst_t = seg.dst_start + (src_t - seg.src_start) * seg.time_scale
                anchors.append((dst_t, pos))
        anchors.append((seg.dst_end, clean_traj.score_position_at(seg.src_end)))
        prev_seg = seg
    anchors.sort(key=lambda a: a[0])
    return TrueTrajectory(anchors=tuple(anchors))
