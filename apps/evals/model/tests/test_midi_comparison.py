"""Tests for structured MIDI comparison."""

import numpy as np
import pytest

from model_improvement.midi_comparison import (
    compare_velocity_curves,
    compare_onset_timing,
    compare_note_accuracy,
    structured_midi_comparison,
)


def _make_notes(pitches, velocities, onsets, durations):
    """Helper to create fake note lists as dicts."""
    return [
        {"pitch": p, "velocity": v, "onset": o, "duration": d}
        for p, v, o, d in zip(pitches, velocities, onsets, durations)
    ]


def test_compare_velocity_curves():
    """Velocity comparison returns MAE and correlation."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 90, 70], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_velocity_curves(perf_notes, score_notes)
    assert "velocity_mae" in result
    assert "velocity_correlation" in result
    assert result["velocity_mae"] > 0  # Different velocities


def test_compare_onset_timing():
    """Onset comparison returns mean and max deviation."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 80, 80], [0.02, 0.48, 1.05], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_onset_timing(perf_notes, score_notes)
    assert "mean_deviation_ms" in result
    assert "max_deviation_ms" in result
    assert result["mean_deviation_ms"] > 0


def test_compare_note_accuracy():
    """Note accuracy reports missed and extra notes."""
    perf_notes = _make_notes(
        [60, 62, 65], [80, 80, 80], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = compare_note_accuracy(perf_notes, score_notes)
    assert "note_f1" in result
    assert "missed_notes" in result
    assert "extra_notes" in result
    assert result["missed_notes"] == 1  # pitch 64 missing
    assert result["extra_notes"] == 1  # pitch 65 extra


def test_structured_midi_comparison_full():
    """structured_midi_comparison returns all comparison features."""
    perf_notes = _make_notes(
        [60, 62, 64], [80, 90, 70], [0.02, 0.48, 1.05], [0.4, 0.4, 0.4]
    )
    score_notes = _make_notes(
        [60, 62, 64], [64, 64, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]
    )
    result = structured_midi_comparison(perf_notes, score_notes)
    assert "velocity" in result
    assert "timing" in result
    assert "accuracy" in result
