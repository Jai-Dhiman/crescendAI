"""MAESTRO real-audio oracle harness (#101 FRONT 9) -- pure windowing + pedal integration."""
from __future__ import annotations

import pytest

from claim_measurement.dynamics_supply.render_maestro_bundles import (
    pedal_on_fraction,
    rel,
    window_notes,
)


def test_pedal_on_fraction_no_events_is_zero():
    assert pedal_on_fraction([], 0.0, 10.0) == 0.0


def test_pedal_down_whole_window_from_prior_event():
    # a down event before the window with nothing inside -> held down the whole window
    pedals = [{"time": -1.0, "value": 127}]
    assert pedal_on_fraction(pedals, 0.0, 10.0) == pytest.approx(1.0)


def test_pedal_down_half_the_window():
    pedals = [{"time": 0.0, "value": 127}, {"time": 5.0, "value": 0}]
    assert pedal_on_fraction(pedals, 0.0, 10.0) == pytest.approx(0.5)


def test_pedal_state_entering_window_from_before():
    # down since t=-1, released at t=3 inside the window -> down for [0,3] of [0,10]
    pedals = [{"time": -1.0, "value": 127}, {"time": 3.0, "value": 0}]
    assert pedal_on_fraction(pedals, 0.0, 10.0) == pytest.approx(0.3)


def test_pedal_threshold_is_64():
    # value 63 is UP, 64 is DOWN (half-pedal boundary)
    assert pedal_on_fraction([{"time": -1.0, "value": 63}], 0.0, 10.0) == 0.0
    assert pedal_on_fraction([{"time": -1.0, "value": 64}], 0.0, 10.0) == pytest.approx(1.0)


def test_window_notes_uses_onset_half_open_interval():
    notes = [{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 50},
             {"pitch": 62, "onset": 5.0, "offset": 6.0, "velocity": 60},
             {"pitch": 64, "onset": 10.0, "offset": 11.0, "velocity": 70}]  # onset==t1 excluded
    win = window_notes(notes, 0.0, 10.0)
    assert [n["pitch"] for n in win] == [60, 62]


def test_rel_shifts_times_to_window_origin():
    notes = [{"pitch": 60, "onset": 5.25, "offset": 6.5, "velocity": 50}]
    (r,) = rel(notes, 5.0)
    assert r["onset"] == pytest.approx(0.25)
    assert r["offset"] == pytest.approx(1.5)
    assert r["velocity"] == 50 and r["pitch"] == 60
