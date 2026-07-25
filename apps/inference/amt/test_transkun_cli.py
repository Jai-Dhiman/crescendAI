"""Behavior tests for the shared Transkun shell-out helper.

Run: cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile \
        --with pytest pytest test_transkun_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transkun_cli


def _write_midi(path: Path, notes, pedal_ccs) -> None:
    """notes: list of (pitch, start_s, end_s, velocity). pedal_ccs: list of (time_s, value)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    for t, v in pedal_ccs:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, time=t, value=v)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_notes_carry_pitch_onset_offset_velocity(tmp_path):
    midi_path = tmp_path / "n.mid"
    _write_midi(
        midi_path,
        notes=[(60, 0.5, 1.0, 90), (67, 0.10, 0.40, 55), (60, 0.10, 0.30, 70)],
        pedal_ccs=[],
    )
    notes, pedals = transkun_cli.midi_to_notes_and_pedals(midi_path)

    assert pedals == []
    # sorted by (onset, pitch): (60,0.10),(67,0.10),(60,0.50)
    assert [(n["pitch"], round(n["onset"], 2)) for n in notes] == [
        (60, 0.10), (67, 0.10), (60, 0.50)
    ]
    first = notes[0]
    assert set(first) == {"pitch", "onset", "offset", "velocity"}
    assert first["velocity"] == 70
    assert round(first["offset"], 2) == 0.30
    assert all(isinstance(n["velocity"], int) for n in notes)
