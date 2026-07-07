# model/src/follower_bench/clip_generator.py
"""Public entrypoint for the synthetic score-follower benchmark (issue
#111): given an ASAP piece identifier and a pathology type, produce a
pathology-injected performance note stream together with its exact
ground-truth score-position trajectory and the labels of what was
injected. Composes asap_alignment (truth substrate) + pathologies
(splice plan) + segments (splice engine) + trajectory (ground truth)
behind one call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import partitura as pa

from follower_bench.asap_alignment import load_alignment
from follower_bench.pathologies import PathologyEvent, build_plan
from follower_bench.segments import PerfNote, apply_note_mutations, apply_segments
from follower_bench.trajectory import TrueTrajectory, build_trajectory_from_segments, from_alignment


@dataclass(frozen=True)
class SynthClip:
    """One generated benchmark clip: the pathology-injected note stream,
    its exact ground-truth score-position trajectory, and the injected
    pathology event labels. `notes` is an in-memory note stream
    (onset/offset/pitch/velocity in seconds); MIDI-file serialization is
    out of scope for #111 and is added by #112 when needed."""
    asap_piece: str
    pathology_type: str
    seed: int
    notes: tuple[PerfNote, ...]
    true_trajectory: TrueTrajectory
    event_labels: tuple[PathologyEvent, ...]


def _load_perf_notes(path: Path) -> list[PerfNote]:
    ppart = pa.load_performance_midi(str(path))
    note_array = ppart.note_array()
    return [
        PerfNote(
            onset=float(row["onset_sec"]),
            offset=float(row["onset_sec"] + row["duration_sec"]),
            pitch=int(row["pitch"]),
            velocity=int(row["velocity"]),
        )
        for row in note_array
    ]


def generate(asap_piece: str, pathology_type: str, seed: int) -> SynthClip:
    """Generate one pathology-injected clip for asap_piece.

    Raises:
        AsapAlignmentMissingError: asap_piece has no usable ASAP beat
            alignment (propagated from asap_alignment.load_alignment) --
            the caller (a batch driver) is expected to catch this and
            skip the piece with a logged reason, never fabricate a
            trajectory.
        FileNotFoundError: the resolved MIDI files are missing on disk.
        ValueError: pathology_type is not a known PATHOLOGY_TYPES member,
            or the piece's beat range is zero-duration.
    """
    alignment = load_alignment(asap_piece)
    rng = random.Random(seed)
    plan = build_plan(alignment, pathology_type, rng)
    clean_traj = from_alignment(alignment)

    notes = _load_perf_notes(alignment.performance_midi_path)
    spliced = apply_segments(notes, list(plan.segments))
    spliced = apply_note_mutations(spliced, list(plan.note_mutations))

    trajectory = build_trajectory_from_segments(clean_traj, list(plan.segments))

    return SynthClip(
        asap_piece=asap_piece,
        pathology_type=pathology_type,
        seed=seed,
        notes=tuple(spliced),
        true_trajectory=trajectory,
        event_labels=plan.events,
    )
