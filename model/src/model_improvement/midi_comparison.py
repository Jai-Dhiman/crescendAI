"""Structured MIDI comparison for Experiment 4 (MIDI-as-context feedback test).

Compares a performance MIDI against a score reference MIDI to produce
structured features that can augment teacher LLM feedback.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _match_notes(
    perf_notes: list[dict],
    score_notes: list[dict],
    onset_tolerance: float = 0.15,
    pitch_tolerance: int = 0,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Match performance notes to score notes by pitch + onset proximity.

    Returns:
        (matched_pairs, missed_score_notes, extra_perf_notes)
    """
    used_score = set()
    matched = []
    extra = []

    for pn in perf_notes:
        best_idx = -1
        best_dist = float("inf")
        for i, sn in enumerate(score_notes):
            if i in used_score:
                continue
            if abs(pn["pitch"] - sn["pitch"]) > pitch_tolerance:
                continue
            dist = abs(pn["onset"] - sn["onset"])
            if dist < onset_tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched.append((pn, score_notes[best_idx]))
            used_score.add(best_idx)
        else:
            extra.append(pn)

    missed = [sn for i, sn in enumerate(score_notes) if i not in used_score]
    return matched, missed, extra


def compare_velocity_curves(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare velocity profiles between performance and score.

    Args:
        perf_notes: [{pitch, velocity, onset, duration}, ...].
        score_notes: [{pitch, velocity, onset, duration}, ...].

    Returns:
        {velocity_mae, velocity_correlation, velocity_profile_summary}.
    """
    matched, _, _ = _match_notes(perf_notes, score_notes)
    if not matched:
        return {"velocity_mae": 0.0, "velocity_correlation": 0.0, "n_matched": 0}

    perf_vel = np.array([m[0]["velocity"] for m in matched], dtype=float)
    score_vel = np.array([m[1]["velocity"] for m in matched], dtype=float)

    mae = float(np.mean(np.abs(perf_vel - score_vel)))

    if len(matched) >= 3 and np.std(perf_vel) > 0 and np.std(score_vel) > 0:
        corr, _ = stats.pearsonr(perf_vel, score_vel)
    else:
        corr = 0.0

    return {
        "velocity_mae": mae,
        "velocity_correlation": float(corr),
        "n_matched": len(matched),
        "perf_velocity_mean": float(perf_vel.mean()),
        "perf_velocity_std": float(perf_vel.std()),
        "score_velocity_mean": float(score_vel.mean()),
    }


def compare_onset_timing(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare onset timing between performance and score.

    Returns:
        {mean_deviation_ms, max_deviation_ms, std_deviation_ms, timing_profile}.
    """
    matched, _, _ = _match_notes(perf_notes, score_notes)
    if not matched:
        return {"mean_deviation_ms": 0.0, "max_deviation_ms": 0.0, "std_deviation_ms": 0.0}

    deviations_ms = np.array([
        (m[0]["onset"] - m[1]["onset"]) * 1000.0 for m in matched
    ])

    return {
        "mean_deviation_ms": float(np.mean(np.abs(deviations_ms))),
        "max_deviation_ms": float(np.max(np.abs(deviations_ms))),
        "std_deviation_ms": float(np.std(deviations_ms)),
        "mean_signed_deviation_ms": float(np.mean(deviations_ms)),
        "n_matched": len(matched),
    }


def compare_note_accuracy(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Compare note-level accuracy between performance and score.

    Returns:
        {note_f1, precision, recall, missed_notes, extra_notes}.
    """
    matched, missed, extra = _match_notes(perf_notes, score_notes)

    n_matched = len(matched)
    n_missed = len(missed)
    n_extra = len(extra)

    precision = n_matched / (n_matched + n_extra) if (n_matched + n_extra) > 0 else 0.0
    recall = n_matched / (n_matched + n_missed) if (n_matched + n_missed) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "note_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "missed_notes": n_missed,
        "extra_notes": n_extra,
        "matched_notes": n_matched,
    }


def structured_midi_comparison(
    perf_notes: list[dict],
    score_notes: list[dict],
) -> dict:
    """Full structured comparison of performance vs score MIDI.

    Combines velocity, timing, and accuracy comparisons into a single
    structured output suitable for inclusion in teacher LLM prompts.

    Args:
        perf_notes: Performance note list [{pitch, velocity, onset, duration}].
        score_notes: Score reference note list.

    Returns:
        {velocity: {...}, timing: {...}, accuracy: {...}, summary: str}.
    """
    velocity = compare_velocity_curves(perf_notes, score_notes)
    timing = compare_onset_timing(perf_notes, score_notes)
    accuracy = compare_note_accuracy(perf_notes, score_notes)

    summary_parts = []
    summary_parts.append(
        f"Note accuracy: F1={accuracy['note_f1']:.2f} "
        f"({accuracy['missed_notes']} missed, {accuracy['extra_notes']} extra)"
    )
    summary_parts.append(
        f"Timing: mean deviation {timing['mean_deviation_ms']:.0f}ms, "
        f"max {timing['max_deviation_ms']:.0f}ms"
    )
    summary_parts.append(
        f"Dynamics: velocity MAE={velocity['velocity_mae']:.0f}, "
        f"correlation={velocity['velocity_correlation']:.2f}"
    )

    return {
        "velocity": velocity,
        "timing": timing,
        "accuracy": accuracy,
        "summary": "; ".join(summary_parts),
    }
