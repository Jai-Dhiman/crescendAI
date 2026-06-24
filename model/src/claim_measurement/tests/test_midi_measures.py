"""MIDI-native measurement adapter for the M2 expert-anchor study (#66).

PercePiano is MIDI; the shipped verifier's dynamics measurer is audio/librosa-RMS.
These functions compute the same whole_piece d-statistics directly from MIDI so the
deterministic measurements can be correlated against PercePiano perceptual labels:
- timing: tempo CV% from inter-onset intervals (mirrors whole_piece timing CV%)
- dynamics: velocity dispersion (the velocity-based adapter the plan calls for)
- pedaling: sustain-on time fraction from CC64 (mirrors presence density)
"""
from __future__ import annotations

import numpy as np
import pytest

from claim_measurement.expert_anchor.midi_measures import (
    dynamics_mean_velocity,
    dynamics_velocity_dispersion,
    pedaling_on_fraction,
    timing_cv_percent,
    timing_ioi_cv,
)


def test_timing_cv_zero_for_constant_ioi():
    onsets = np.arange(0.0, 5.0, 0.5)  # perfectly regular -> CV 0
    assert timing_cv_percent(onsets) == pytest.approx(0.0, abs=1e-6)


def test_timing_cv_positive_for_irregular():
    onsets = np.array([0.0, 0.5, 0.7, 1.6, 1.8, 3.0])  # uneven
    assert timing_cv_percent(onsets) > 5.0


def test_timing_cv_collapses_chords():
    # Simultaneous onsets (a chord) must not create a zero IOI / inf tempo.
    onsets = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    cv = timing_cv_percent(onsets)
    assert np.isfinite(cv) and cv == pytest.approx(0.0, abs=1e-6)


def test_timing_cv_raises_when_too_few_onsets():
    with pytest.raises(ValueError):
        timing_cv_percent(np.array([1.0]))


def test_dynamics_dispersion_zero_for_flat():
    assert dynamics_velocity_dispersion(np.full(20, 64)) == pytest.approx(0.0, abs=1e-9)


def test_dynamics_dispersion_increases_with_spread():
    narrow = dynamics_velocity_dispersion(np.array([60, 62, 64, 66, 68]))
    wide = dynamics_velocity_dispersion(np.array([20, 50, 64, 90, 120]))
    assert wide > narrow > 0.0


def test_pedaling_on_fraction_half():
    # CC64 on at t=0..5 (>=64), off at 5..10 -> half the 10s span is pedaled.
    events = [(0.0, 127), (5.0, 0)]
    assert pedaling_on_fraction(events, total_dur=10.0) == pytest.approx(0.5, abs=1e-6)


def test_pedaling_on_fraction_none():
    assert pedaling_on_fraction([], total_dur=10.0) == 0.0


def test_timing_ioi_cv_zero_for_constant_and_positive_for_irregular():
    assert timing_ioi_cv(np.arange(0.0, 5.0, 0.5)) == pytest.approx(0.0, abs=1e-6)
    assert timing_ioi_cv(np.array([0.0, 0.5, 0.7, 1.6, 1.8, 3.0])) > 0.1


def test_dynamics_mean_velocity():
    assert dynamics_mean_velocity(np.array([40, 60, 80])) == pytest.approx(60.0)
    with pytest.raises(ValueError):
        dynamics_mean_velocity(np.array([]))
