"""Python port of bar_analysis.rs Tier-1/2 statistics for the eval harness.

Mirrors apps/api/src/wasm/score-analysis/src/bar_analysis.rs. The Rust file is
the source of truth; if its formulas change, this file must change to match.
"""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any

DIMS_ORDER: list[str] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
]


def compute_tier2_dimensions(
    midi_notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return six DimensionAnalysis-shaped dicts from raw chunk MIDI + pedal."""
    if not midi_notes:
        return [{"dimension": d, "analysis": "No notes in chunk."} for d in DIMS_ORDER]

    velocities = [n["velocity"] for n in midi_notes]
    onsets = sorted(n["onset"] for n in midi_notes)
    iois = [b - a for a, b in zip(onsets, onsets[1:])] if len(onsets) >= 2 else []
    durations = [n["offset"] - n["onset"] for n in midi_notes]

    vel_mean = mean(velocities)
    vel_range = max(velocities) - min(velocities)
    ioi_mean = mean(iois) if iois else 0.0
    ioi_std = stdev(iois) if len(iois) >= 2 else 0.0
    dur_mean = mean(durations)
    pedal_count = len(pedal_events)

    return [
        {"dimension": "dynamics",
         "analysis": f"Mean velocity {vel_mean:.1f} (range {vel_range})."},
        {"dimension": "timing",
         "analysis": f"Mean inter-onset interval {ioi_mean:.3f}s (std {ioi_std:.3f}s)."},
        {"dimension": "pedaling",
         "analysis": f"{pedal_count} pedal events across chunk."},
        {"dimension": "articulation",
         "analysis": f"Mean note duration {dur_mean:.2f}s."},
        {"dimension": "phrasing",
         "analysis": f"{len(midi_notes)} notes; mean IOI {ioi_mean:.3f}s."},
        {"dimension": "interpretation",
         "analysis": f"Velocity range {vel_range}, IOI std {ioi_std:.3f}s."},
    ]


def select_worst_chunk(
    chunks: list[dict[str, Any]],
    baselines: dict[str, float],
) -> dict[str, Any] | None:
    """Return {chunk_index, dimension, chunk} for the chunk + dim with max |score-baseline|."""
    if not chunks:
        return None
    best: dict[str, Any] | None = None
    best_dev = -1.0
    for chunk in chunks:
        preds = chunk.get("predictions", {})
        for dim, score in preds.items():
            if dim not in baselines:
                continue
            dev = abs(float(score) - float(baselines[dim]))
            if dev > best_dev:
                best_dev = dev
                best = {
                    "chunk_index": chunk.get("chunk_index"),
                    "dimension": dim,
                    "chunk": chunk,
                }
    return best
