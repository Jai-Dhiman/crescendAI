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


def test_cc64_maps_to_pedal_on_off(tmp_path):
    midi_path = tmp_path / "p.mid"
    _write_midi(
        midi_path,
        notes=[(60, 0.0, 1.0, 80)],
        pedal_ccs=[(0.20, 100), (0.80, 10), (0.90, 64), (1.10, 63)],
    )
    _notes, pedals = transkun_cli.midi_to_notes_and_pedals(midi_path)

    assert [(round(p["time"], 2), p["value"]) for p in pedals] == [
        (0.20, 127),  # 100 >= 64 -> on
        (0.80, 0),    # 10  <  64 -> off
        (0.90, 127),  # 64  >= 64 -> on (boundary)
        (1.10, 0),    # 63  <  64 -> off (boundary)
    ]
    assert all(p["value"] in (0, 127) for p in pedals)


def test_transcribe_wav_missing_input_raises(tmp_path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(transkun_cli.TranskunError):
        transkun_cli.transcribe_wav(missing)
