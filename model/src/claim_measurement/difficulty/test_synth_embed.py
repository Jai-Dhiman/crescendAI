"""Offline tests for the #167 embedding pass. The 1.6GB backbone is never
loaded -- what is under test is which MIDI each arm reads and what the pass
refuses."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pretty_midi
import pytest

from claim_measurement.difficulty.synth_embed import source_midis

MANIFEST = [{"seg_id": "a", "key": "A", "midi_name": "mid/A.mid"},
            {"seg_id": "b", "key": "B", "midi_name": "mid/B.mid"}]


def _midi_bytes(n_notes=4, velocity=90, with_pedal=True) -> bytes:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_notes):
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=60 + i,
                                           start=0.5 * i, end=0.5 * i + 0.4))
    if with_pedal:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=0.0))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


@pytest.fixture
def mid_zip(tmp_path) -> Path:
    path = tmp_path / "mid.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mid/A.mid", _midi_bytes())
        zf.writestr("mid/B.mid", _midi_bytes(n_notes=7, velocity=40))
    return path


def test_source_midis_writes_one_file_per_seg_id(mid_zip, tmp_path):
    out = source_midis(["a", "b"], MANIFEST, mid_zip, tmp_path / "work")
    assert set(out) == {"a", "b"}
    assert all(p.is_file() for p in out.values())


def test_source_midis_preserves_notes_and_velocities(mid_zip, tmp_path):
    out = source_midis(["b"], MANIFEST, mid_zip, tmp_path / "work")
    pm = pretty_midi.PrettyMIDI(str(out["b"]))
    notes = [n for i in pm.instruments for n in i.notes]
    assert len(notes) == 7
    assert {n.velocity for n in notes} == {40}


def test_source_midis_goes_through_the_same_note_list_step_as_the_arms(mid_zip,
                                                                       tmp_path):
    """write_notes_midi drops pedal, and B0 must be written the same way the
    rendered arms' transcriptions are. Extracting the zip member verbatim would
    leave pedal in place for B0 only, so the control would differ from the arms
    in a second way beyond the one it isolates."""
    out = source_midis(["a"], MANIFEST, mid_zip, tmp_path / "work")
    pm = pretty_midi.PrettyMIDI(str(out["a"]))
    assert [c for i in pm.instruments for c in i.control_changes] == []
    assert len(pm.instruments) == 1
