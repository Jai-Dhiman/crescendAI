"""Offline tests for the #167 arm runner. No fluidsynth, no Transkun: both are
injected, which is the point of the seam."""
from __future__ import annotations

import io
from pathlib import Path

import pretty_midi
import pytest

from claim_measurement.difficulty.synth_arm import (
    ArmItem,
    append_row,
    pending,
    plan_items,
    read_rows,
    run_arm,
    run_item,
    transcription_path,
)


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
    import zipfile

    path = tmp_path / "mid.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mid/A.mid", _midi_bytes())
        zf.writestr("mid/B.mid", _midi_bytes(n_notes=6))
    return path


@pytest.fixture
def soundfont(tmp_path) -> Path:
    path = tmp_path / "sf.sf2"
    path.write_bytes(b"x")
    return path


def _fake_render(midi_path, sf2, out_wav, **kwargs):
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    Path(out_wav).write_bytes(b"RIFF" + b"\0" * 100)
    return Path(out_wav)


def _fake_transcribe(wav_path):
    return ([{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 70},
             {"pitch": 64, "onset": 1.0, "offset": 2.0, "velocity": 70}], [])


ITEM_A = ArmItem(seg_id="A", midi_name="mid/A.mid", fold=0)
ITEM_B = ArmItem(seg_id="B", midi_name="mid/B.mid", fold=1)


# --------------------------------------------------------------------------
# plan_items
# --------------------------------------------------------------------------


def test_plan_items_preserves_the_canonical_eval_row_order():
    rows = [{"seg_id": "B", "midi_name": "mid/B.mid"},
            {"seg_id": "A", "midi_name": "mid/A.mid"}]
    items = plan_items(["A", "B"], rows, {"A": 0, "B": 3})
    assert [i.seg_id for i in items] == ["A", "B"]
    assert [i.midi_name for i in items] == ["mid/A.mid", "mid/B.mid"]
    assert [i.fold for i in items] == [0, 3]


def test_plan_items_raises_rather_than_dropping_a_seg_id_with_no_manifest_row():
    """A silently shorter arm still produces a tau-c, and it would be a tau-c
    over a different population than the one it is compared against."""
    with pytest.raises(KeyError, match="no manifest row"):
        plan_items(["A", "Z"], [{"seg_id": "A", "midi_name": "mid/A.mid"}],
                   {"A": 0, "Z": 0})


def test_plan_items_raises_on_a_seg_id_assigned_to_no_fold():
    with pytest.raises(KeyError, match="no fold"):
        plan_items(["A"], [{"seg_id": "A", "midi_name": "mid/A.mid"}], {})


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_pending_is_everything_when_nothing_is_cached(tmp_path):
    assert pending([ITEM_A, ITEM_B], tmp_path) == [ITEM_A, ITEM_B]


def test_pending_skips_items_whose_transcription_exists(tmp_path):
    path = transcription_path(tmp_path, "A")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert pending([ITEM_A, ITEM_B], tmp_path) == [ITEM_B]


def test_a_completed_item_is_not_redone_on_a_second_run(tmp_path, mid_zip, soundfont):
    calls = []

    def counting_transcribe(wav_path):
        calls.append(wav_path)
        return _fake_transcribe(wav_path)

    for _ in range(2):
        run_arm([ITEM_A], "expressive", mid_zip, soundfont, tmp_path,
                counting_transcribe, render=_fake_render, log=lambda *a: None)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# run_item
# --------------------------------------------------------------------------


def test_run_item_writes_a_transcription_and_an_ok_row(tmp_path, mid_zip, soundfont):
    row = run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
                   _fake_transcribe, render=_fake_render)
    assert row["ok"] is True
    assert row["seg_id"] == "A" and row["fold"] == 0
    assert row["source_notes"] == 4 and row["source_pedals"] == 1
    assert row["transcribed_notes"] == 2
    assert transcription_path(tmp_path, "A").is_file()


def test_run_item_deadpan_arm_flattens_velocity_and_drops_pedal(tmp_path, mid_zip,
                                                                soundfont):
    """The arm transform must reach the rendered MIDI, not just exist in
    synth_render -- otherwise both arms would render identical audio and the
    deadpan delta would read as a clean zero."""
    seen = {}

    def capturing_render(midi_path, sf2, out_wav, **kwargs):
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        seen["velocities"] = {n.velocity for i in pm.instruments for n in i.notes}
        seen["pedals"] = [c for i in pm.instruments for c in i.control_changes]
        return _fake_render(midi_path, sf2, out_wav)

    row = run_item(ITEM_A, "deadpan", mid_zip, soundfont, tmp_path,
                   _fake_transcribe, render=capturing_render)
    assert row["ok"] is True
    assert seen["velocities"] == {64}
    assert seen["pedals"] == []
    assert row["source_pedals"] == 0


def test_run_item_expressive_arm_keeps_velocity_and_pedal(tmp_path, mid_zip, soundfont):
    seen = {}

    def capturing_render(midi_path, sf2, out_wav, **kwargs):
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        seen["velocities"] = {n.velocity for i in pm.instruments for n in i.notes}
        seen["pedals"] = [c for i in pm.instruments for c in i.control_changes]
        return _fake_render(midi_path, sf2, out_wav)

    run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
             _fake_transcribe, render=capturing_render)
    assert seen["velocities"] == {90}
    assert len(seen["pedals"]) == 1


def test_run_item_records_a_render_failure_without_raising(tmp_path, mid_zip,
                                                           soundfont):
    def failing_render(*a, **k):
        raise RuntimeError("fluidsynth exploded")

    row = run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
                   _fake_transcribe, render=failing_render)
    assert row["ok"] is False
    assert "fluidsynth exploded" in row["error"]
    assert "traceback" in row
    assert not transcription_path(tmp_path, "A").exists()


def test_run_item_treats_an_empty_transcription_as_a_failure(tmp_path, mid_zip,
                                                             soundfont):
    """A rendered WAV that transcribes to nothing would otherwise become a
    zero-note MIDI and a confident embedding of silence."""
    row = run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
                   lambda w: ([], []), render=_fake_render)
    assert row["ok"] is False
    assert "no notes" in row["error"]
    assert not transcription_path(tmp_path, "A").exists()


def test_run_item_deletes_the_intermediate_wav(tmp_path, mid_zip, soundfont):
    """900 renders of the mean 150s piece is ~24GB if these survive."""
    run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
             _fake_transcribe, render=_fake_render)
    assert list((tmp_path / "work").glob("*.wav")) == []
    assert list((tmp_path / "work").glob("*.mid")) == []


def test_run_item_cleans_up_intermediates_even_when_the_item_fails(tmp_path, mid_zip,
                                                                   soundfont):
    def failing_transcribe(wav_path):
        raise RuntimeError("transkun exploded")

    run_item(ITEM_A, "expressive", mid_zip, soundfont, tmp_path,
             failing_transcribe, render=_fake_render)
    assert list((tmp_path / "work").glob("*.wav")) == []


def test_run_item_rejects_an_unknown_arm(tmp_path, mid_zip, soundfont):
    with pytest.raises(ValueError, match="arm must be one of"):
        run_item(ITEM_A, "source", mid_zip, soundfont, tmp_path,
                 _fake_transcribe, render=_fake_render)


# --------------------------------------------------------------------------
# run_arm
# --------------------------------------------------------------------------


def test_run_arm_streams_a_row_per_item_and_returns_zero_failures(tmp_path, mid_zip,
                                                                  soundfont):
    failures = run_arm([ITEM_A, ITEM_B], "expressive", mid_zip, soundfont, tmp_path,
                       _fake_transcribe, workers=2, render=_fake_render,
                       log=lambda *a: None)
    assert failures == 0
    rows = read_rows(tmp_path)
    assert {r["seg_id"] for r in rows} == {"A", "B"}
    assert all(r["ok"] for r in rows)


def test_run_arm_counts_failures_and_keeps_going(tmp_path, mid_zip, soundfont):
    """16 hours of work must not be lost to one bad item, and the count must
    still make the run exit non-zero."""
    def flaky_transcribe(wav_path):
        if "A" in Path(wav_path).stem:
            raise RuntimeError("boom")
        return _fake_transcribe(wav_path)

    failures = run_arm([ITEM_A, ITEM_B], "expressive", mid_zip, soundfont, tmp_path,
                       flaky_transcribe, render=_fake_render, log=lambda *a: None)
    assert failures == 1
    rows = {r["seg_id"]: r for r in read_rows(tmp_path)}
    assert rows["A"]["ok"] is False and rows["B"]["ok"] is True
    assert transcription_path(tmp_path, "B").is_file()
    assert not transcription_path(tmp_path, "A").exists()


def test_run_arm_returns_zero_and_does_nothing_when_all_items_are_cached(
        tmp_path, mid_zip, soundfont):
    for seg_id in ("A", "B"):
        path = transcription_path(tmp_path, seg_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def exploding_transcribe(wav_path):
        raise AssertionError("must not be called")

    assert run_arm([ITEM_A, ITEM_B], "expressive", mid_zip, soundfont, tmp_path,
                   exploding_transcribe, render=_fake_render,
                   log=lambda *a: None) == 0


def test_append_row_and_read_rows_round_trip(tmp_path):
    append_row(tmp_path, {"seg_id": "A", "ok": True})
    append_row(tmp_path, {"seg_id": "B", "ok": False})
    assert [r["seg_id"] for r in read_rows(tmp_path)] == ["A", "B"]


def test_read_rows_is_empty_before_any_run(tmp_path):
    assert read_rows(tmp_path) == []
