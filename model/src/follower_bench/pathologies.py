# model/src/follower_bench/pathologies.py
"""Per-pathology-type construction of a ClipPlan: a deterministic (given
rng) Segment list that rearranges a clean ASAP performance's timeline,
plus the PathologyEvent labels describing what was injected and where.
wrong_note additionally carries a NoteMutation (pitch substitution, no
timeline rearrangement) instead of a rearranging Segment list.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from follower_bench.asap_alignment import ClipAlignment
from follower_bench.segments import NoteMutation, Segment
from follower_bench.trajectory import from_alignment

PATHOLOGY_TYPES = (
    "clean",
    "repeat",
    "jump",
    "restart",
    "hesitation",
    "wrong_note",
    "tempo_swing",
)

_HESITATION_PAUSE_MIN_S = 1.0
_HESITATION_PAUSE_MAX_S = 3.0
_WRONG_NOTE_PITCH_DELTAS = (-2, -1, 1, 2)
_TEMPO_SWING_SUBSEGMENTS = 4
_TEMPO_SWING_SCALE_START = 1.3
_TEMPO_SWING_SCALE_END = 0.8


@dataclass(frozen=True)
class PathologyEvent:
    """One injected pathology event. For pure timeline jumps (repeat,
    jump, restart), from_score_position != to_score_position. For
    pathologies that do not change score position (hesitation,
    wrong_note, tempo_swing), from_score_position == to_score_position."""
    type: str
    perf_time: float
    from_score_position: float
    to_score_position: float


@dataclass(frozen=True)
class ClipPlan:
    segments: tuple[Segment, ...]
    events: tuple[PathologyEvent, ...]
    note_mutations: tuple[NoteMutation, ...] = ()


def _bounds(alignment: ClipAlignment) -> tuple[float, float]:
    t_min = alignment.performance_beats[0]
    t_max = alignment.performance_beats[-1]
    if t_max <= t_min:
        raise ValueError(
            f"{alignment.asap_piece}: zero-duration performance ({t_min}..{t_max}), cannot splice"
        )
    return t_min, t_max


def _pick_two_points(alignment: ClipAlignment, rng: random.Random) -> tuple[float, float]:
    """Pick two ordered perf-time points inside the piece's beat-anchored
    range: a in [15%, 35%) of duration, b in [55%, 75%) of duration. The
    fixed, non-overlapping bands guarantee t_min < a < b < t_max for any
    duration > 0."""
    t_min, t_max = _bounds(alignment)
    duration = t_max - t_min
    a = t_min + rng.uniform(0.15, 0.35) * duration
    b = t_min + rng.uniform(0.55, 0.75) * duration
    return a, b


def build_plan(alignment: ClipAlignment, pathology_type: str, rng: random.Random) -> ClipPlan:
    """Build the ClipPlan for pathology_type.

    Raises:
        ValueError: pathology_type is not one of PATHOLOGY_TYPES, or the
            alignment's beat range is zero-duration.
    """
    if pathology_type not in PATHOLOGY_TYPES:
        raise ValueError(f"Unknown pathology_type {pathology_type!r}; must be one of {PATHOLOGY_TYPES}")

    t_min, t_max = _bounds(alignment)

    if pathology_type == "clean":
        return ClipPlan(segments=(Segment(t_min, t_max, t_min, 1.0),), events=())

    clean_traj = from_alignment(alignment)

    if pathology_type == "repeat":
        x, y = _pick_two_points(alignment, rng)
        seg1 = Segment(t_min, y, t_min, 1.0)
        seg2 = Segment(x, y, seg1.dst_end, 1.0)
        seg3 = Segment(y, t_max, seg2.dst_end, 1.0)
        event = PathologyEvent(
            type="repeat",
            perf_time=seg1.dst_end,
            from_score_position=clean_traj.score_position_at(y),
            to_score_position=clean_traj.score_position_at(x),
        )
        return ClipPlan(segments=(seg1, seg2, seg3), events=(event,))

    raise NotImplementedError(f"pathology_type {pathology_type!r} not yet implemented")
