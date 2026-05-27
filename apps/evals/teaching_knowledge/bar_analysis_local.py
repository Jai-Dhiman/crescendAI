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


def compute_tier1_dimensions(
    midi_notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
    score_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tier-1: compute_tier2_dimensions + notated articulation comparison.

    NOTE: Dynamics is deliberately NOT enriched with a score comparison. Score
    JSONs under model/data/scores/ use the default MIDI export velocity 80 for
    every note (not a real notated dynamic), so a performance-vs-score velocity
    line carries zero signal and would dilute the prompt. The Rust Tier-1 in
    production has the same input limitation. Articulation IS enriched because
    notated note durations vary and the perf/score duration ratio is real signal.
    """
    base = compute_tier2_dimensions(midi_notes, pedal_events)
    if not midi_notes:
        return base

    score_notes: list[dict[str, Any]] = []
    for bar in score_json.get("bars", []):
        score_notes.extend(bar.get("notes", []))
    if not score_notes:
        return base

    perf_dur_mean = mean(n["offset"] - n["onset"] for n in midi_notes)
    score_dur_mean = mean(n.get("duration_seconds", 0.0) for n in score_notes) or 1e-6

    enriched = []
    for d in base:
        if d["dimension"] == "articulation":
            ratio = perf_dur_mean / score_dur_mean
            d = {
                **d,
                "analysis": (
                    f"{d['analysis']} Performance/score duration ratio {ratio:.2f}."
                ),
            }
        enriched.append(d)
    return enriched


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


DEVIATION_THRESHOLD = 0.15
CORRELATED_CAP = 2


def build_bar_analysis(
    chunks: list[dict[str, Any]],
    baselines: dict[str, float],
    score_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Public entry: pick worst chunk, compute Tier-1 or Tier-2, filter to facts."""
    worst = select_worst_chunk(chunks, baselines)
    if worst is None:
        return None
    chunk = worst["chunk"]
    selected_dim = worst["dimension"]
    midi = chunk.get("midi_notes", [])
    pedal = chunk.get("pedal_events", [])

    if score_json is not None:
        dims = compute_tier1_dimensions(midi, pedal, score_json)
        tier = 1
    else:
        dims = compute_tier2_dimensions(midi, pedal)
        tier = 2

    selected = next((d for d in dims if d["dimension"] == selected_dim), None)
    if selected is None:
        return None

    preds = chunk.get("predictions", {})
    candidates = []
    for d in dims:
        if d["dimension"] == selected_dim:
            continue
        if d["dimension"] not in baselines or d["dimension"] not in preds:
            continue
        dev = abs(float(preds[d["dimension"]]) - float(baselines[d["dimension"]]))
        if dev >= DEVIATION_THRESHOLD:
            candidates.append((dev, d))
    candidates.sort(key=lambda x: x[0], reverse=True)
    correlated = [d for _, d in candidates[:CORRELATED_CAP]]

    return {
        "tier": tier,
        "bar_range": None,
        "selected": selected,
        "correlated": correlated,
    }
