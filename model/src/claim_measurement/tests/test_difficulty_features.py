"""Difficulty features for MIREX 2026 Music Performance Difficulty Prediction (#104).

These are the pitch/onset-based features RubricNet reports as strongest
(pitch-entropy + pitch-set Lempel-Ziv complexity, ~0.58 tau-c) plus polyphony
and pitch-range. All four are computed from AMT/MIDI note events using ONLY the
signals AMT recovers reliably -- pitch and onset (0.97 F1). Duration/offset-based
features are deliberately excluded (AMT offsets are fragile; see research-learnings.md
A5/A6). Every value here is hand-computable so the unit tests below pin exact numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from claim_measurement.difficulty.difficulty_features import (
    extract_difficulty_features,
    notes_to_pitch_sequence,
    pitch_entropy,
    pitch_lz_complexity,
    pitch_range,
    polyphony_per_onset,
)


def _note(pitch, onset, offset=None, velocity=64):
    """Minimal note dict matching the amt_fidelity note schema."""
    return {"pitch": int(pitch), "onset": float(onset),
            "offset": float(onset + 0.5 if offset is None else offset),
            "velocity": int(velocity)}


# ---- pitch_entropy (Shannon entropy of the pitch histogram, bits) -----------

def test_pitch_entropy_single_pitch_is_zero():
    assert pitch_entropy([60, 60, 60]) == pytest.approx(0.0, abs=1e-12)


def test_pitch_entropy_two_equal_pitches_is_one_bit():
    # {60:0.5, 64:0.5} -> H = -2 * (0.5 * log2 0.5) = 1.0 bit
    assert pitch_entropy([60, 60, 64, 64]) == pytest.approx(1.0, abs=1e-12)


def test_pitch_entropy_four_equal_pitches_is_two_bits():
    assert pitch_entropy([60, 62, 64, 65]) == pytest.approx(2.0, abs=1e-12)


def test_pitch_entropy_matches_manual_skewed_distribution():
    # {60:3, 64:1} over n=4 -> p=[0.75, 0.25]
    # H = -(0.75*log2 0.75 + 0.25*log2 0.25)
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert pitch_entropy([60, 60, 60, 64]) == pytest.approx(expected, abs=1e-12)


def test_pitch_entropy_empty_raises():
    with pytest.raises(ValueError):
        pitch_entropy([])


# ---- pitch_lz_complexity (LZ76 production complexity, integer count) ---------

def test_lz_empty_is_zero():
    assert pitch_lz_complexity([]) == 0


def test_lz_single_symbol_is_one():
    assert pitch_lz_complexity([60]) == 1


def test_lz_constant_sequence_is_two():
    # "aaaa" parses as a | aaa -> 2 phrases
    assert pitch_lz_complexity([60, 60, 60, 60]) == 2


def test_lz_all_distinct_equals_length():
    # every symbol novel -> one phrase each
    assert pitch_lz_complexity([60, 62, 64, 65]) == 4


def test_lz_more_repetitive_scores_lower_than_random():
    repetitive = [60, 62, 60, 62, 60, 62, 60, 62]
    varied = [60, 62, 64, 67, 71, 60, 65, 69]
    assert pitch_lz_complexity(repetitive) < pitch_lz_complexity(varied)


# ---- polyphony_per_onset (mean notes per onset cluster, onset-only) ----------

def test_polyphony_monophonic_is_one():
    onsets = [0.0, 1.0, 2.0, 3.0]
    assert polyphony_per_onset(onsets) == pytest.approx(1.0, abs=1e-12)


def test_polyphony_single_chord_is_chord_size():
    onsets = [0.0, 0.0, 0.0]
    assert polyphony_per_onset(onsets) == pytest.approx(3.0, abs=1e-12)


def test_polyphony_mixed_clusters_manual_mean():
    # clusters of size [3, 2, 1] -> mean 2.0
    onsets = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    assert polyphony_per_onset(onsets) == pytest.approx(2.0, abs=1e-12)


def test_polyphony_clusters_within_tolerance():
    # near-simultaneous onsets inside tol collapse to one chord
    onsets = [0.000, 0.005, 0.010, 1.000]
    assert polyphony_per_onset(onsets, onset_tol_s=0.03) == pytest.approx(2.0, abs=1e-12)


def test_polyphony_empty_raises():
    with pytest.raises(ValueError):
        polyphony_per_onset([])


# ---- pitch_range (max - min MIDI pitch, semitones) --------------------------

def test_pitch_range_octave_is_twelve():
    assert pitch_range([60, 72]) == 12


def test_pitch_range_single_pitch_is_zero():
    assert pitch_range([60]) == 0


def test_pitch_range_empty_raises():
    with pytest.raises(ValueError):
        pitch_range([])


# ---- notes_to_pitch_sequence (deterministic onset,pitch ordering) -----------

def test_pitch_sequence_orders_by_onset_then_pitch():
    notes = [_note(64, 1.0), _note(60, 0.0), _note(67, 0.0)]
    # onset 0.0 chord ordered low->high (60,67), then onset 1.0 (64)
    assert notes_to_pitch_sequence(notes) == [60, 67, 64]


# ---- extract_difficulty_features (the Phase-1 entry point) -------------------

def test_extract_returns_all_four_features():
    notes = [_note(60, 0.0), _note(64, 0.0), _note(67, 1.0), _note(72, 2.0)]
    feats = extract_difficulty_features(notes)
    assert set(feats) == {"pitch_entropy", "pitch_lz_complexity",
                          "polyphony", "pitch_range"}
    assert feats["pitch_range"] == 12  # 72 - 60
    assert feats["polyphony"] == pytest.approx(4 / 3, abs=1e-12)  # [2,1,1]
    for v in feats.values():
        assert np.isfinite(v)


def test_extract_empty_raises():
    with pytest.raises(ValueError):
        extract_difficulty_features([])
