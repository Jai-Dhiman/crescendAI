"""Offline tests for the #167 synthesis arm. Everything here runs without
fluidsynth except the two tests marked `real_fluidsynth`, which are skipped
when the binary or a soundfont is absent rather than silently passing."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path

import pretty_midi
import pytest

from claim_measurement.difficulty.bakeoff_paths import DEFAULT_DATA_ROOT
from claim_measurement.difficulty.synth_render import (
    SAMPLE_RATE,
    SUSTAIN_CC,
    Performance,
    deadpan,
    read_performance,
    render_wav,
    write_performance_midi,
)

# Worktrees carry their own (mostly unpopulated) model/data, so the soundfont
# usually lives in the primary checkout. Same override philosophy as
# run_bakeoff.py's --data-root: never a hardcoded machine path, and absent it
# the real-binary tests skip loudly rather than pass vacuously.
_DATA_ROOT = Path(os.environ.get("MIREX_DATA_ROOT", DEFAULT_DATA_ROOT))
SOUNDFONT = _DATA_ROOT / "weights" / "soundfonts" / "FluidR3_GM.sf2"


def _midi_bytes(notes, pedals=()) -> bytes:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, onset, offset, velocity in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch,
                                           start=onset, end=offset))
    for time, value in pedals:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=SUSTAIN_CC, value=value, time=time))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# read_performance
# --------------------------------------------------------------------------


def test_read_performance_returns_notes_onset_ordered_and_pedals_time_ordered():
    raw = _midi_bytes(
        notes=[(72, 1.0, 1.5, 90), (60, 0.0, 0.5, 40)],
        pedals=[(1.2, 0), (0.1, 127)],
    )
    perf = read_performance(raw)
    assert [n["pitch"] for n in perf.notes] == [60, 72]
    assert [n["velocity"] for n in perf.notes] == [40, 90]
    # approx, not equality: pretty_midi round-trips event times through MIDI
    # ticks, so 0.1 comes back as 0.09999999999999999. Ordering is the claim.
    assert [p["time"] for p in perf.pedals] == pytest.approx([0.1, 1.2], abs=1e-3)
    assert [p["value"] for p in perf.pedals] == [127, 0]


def test_read_performance_keeps_pedal_that_notes_from_midi_bytes_discards():
    """The whole reason this reader exists rather than reusing
    psyllabus.notes_from_midi_bytes: the pedal is one of the two things the
    expressive and deadpan arms differ in."""
    from claim_measurement.difficulty.psyllabus import notes_from_midi_bytes

    raw = _midi_bytes(notes=[(60, 0.0, 1.0, 80)], pedals=[(0.0, 127)])
    assert read_performance(raw).pedals == [{"time": 0.0, "value": 127}]
    assert notes_from_midi_bytes(raw) == read_performance(raw).notes


def test_read_performance_ignores_non_sustain_control_changes():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
    inst.control_changes.append(
        pretty_midi.ControlChange(number=67, value=127, time=0.0))
    inst.control_changes.append(
        pretty_midi.ControlChange(number=SUSTAIN_CC, value=64, time=0.5))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)

    perf = read_performance(buf.getvalue())
    assert perf.pedals == [{"time": 0.5, "value": 64}]


# --------------------------------------------------------------------------
# deadpan
# --------------------------------------------------------------------------


def test_deadpan_flattens_velocity_and_drops_pedal():
    perf = Performance(
        notes=[{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 30},
               {"pitch": 64, "onset": 0.5, "offset": 1.4, "velocity": 110}],
        pedals=[{"time": 0.0, "value": 127}],
    )
    out = deadpan(perf)
    assert {n["velocity"] for n in out.notes} == {64}
    assert out.pedals == []


def test_deadpan_preserves_timing_because_a_transcription_has_no_metrical_grid():
    """Pinned as a property, not an accident. Quantizing to these files'
    nominal 120bpm would fabricate structure; see the module docstring. Any
    deadpan delta is therefore a lower bound, and that only holds if timing is
    genuinely untouched."""
    perf = Performance(
        notes=[{"pitch": 60, "onset": 0.4336, "offset": 1.3828, "velocity": 30},
               {"pitch": 64, "onset": 1.4167, "offset": 1.9362, "velocity": 110}],
        pedals=[],
    )
    out = deadpan(perf)
    assert [(n["onset"], n["offset"]) for n in out.notes] == \
           [(0.4336, 1.3828), (1.4167, 1.9362)]
    assert [n["pitch"] for n in out.notes] == [60, 64]


def test_deadpan_does_not_mutate_its_input():
    perf = Performance(
        notes=[{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 30}],
        pedals=[{"time": 0.0, "value": 127}])
    deadpan(perf)
    assert perf.notes[0]["velocity"] == 30
    assert perf.pedals == [{"time": 0.0, "value": 127}]


@pytest.mark.parametrize("velocity", [0, -1, 128])
def test_deadpan_rejects_velocities_that_would_render_silence_or_be_invalid(velocity):
    perf = Performance(
        notes=[{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 30}],
        pedals=[])
    with pytest.raises(ValueError, match="velocity"):
        deadpan(perf, velocity=velocity)


# --------------------------------------------------------------------------
# write_performance_midi
# --------------------------------------------------------------------------


def test_write_performance_midi_round_trips_notes_and_pedal(tmp_path):
    perf = Performance(
        notes=[{"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 30},
               {"pitch": 64, "onset": 0.5, "offset": 1.4, "velocity": 110}],
        pedals=[{"time": 0.25, "value": 127}, {"time": 1.5, "value": 0}],
    )
    out = tmp_path / "nested" / "p.mid"
    write_performance_midi(perf, out)

    back = read_performance(out.read_bytes())
    assert [n["pitch"] for n in back.notes] == [60, 64]
    assert [n["velocity"] for n in back.notes] == [30, 110]
    assert [p["value"] for p in back.pedals] == [127, 0]
    assert back.notes[1]["onset"] == pytest.approx(0.5, abs=1e-3)


def test_write_performance_midi_refuses_an_empty_performance(tmp_path):
    with pytest.raises(ValueError, match="no notes"):
        write_performance_midi(Performance(notes=[], pedals=[]), tmp_path / "p.mid")


# --------------------------------------------------------------------------
# render_wav -- the subprocess boundary, faked
# --------------------------------------------------------------------------


def _fake_run_writing(out_index: int, payload: bytes = b"RIFFfake"):
    def run(argv, **kwargs):
        Path(argv[out_index]).write_bytes(payload)
        return subprocess.CompletedProcess(argv, 0, "", "")
    return run


def test_render_wav_passes_the_contract_sample_rate_and_the_soundfont(tmp_path):
    midi = tmp_path / "p.mid"
    midi.write_bytes(b"x")
    sf2 = tmp_path / "s.sf2"
    sf2.write_bytes(b"x")
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        Path(argv[3]).write_bytes(b"RIFFfake")
        return subprocess.CompletedProcess(argv, 0, "", "")

    render_wav(midi, sf2, tmp_path / "o.wav", run=run)
    argv = seen["argv"]
    assert argv[argv.index("-r") + 1] == str(SAMPLE_RATE)
    assert argv[-2:] == [str(sf2), str(midi)]


def test_render_wav_raises_when_fluidsynth_exits_nonzero(tmp_path):
    midi = tmp_path / "p.mid"
    midi.write_bytes(b"x")
    sf2 = tmp_path / "s.sf2"
    sf2.write_bytes(b"x")

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "boom")

    with pytest.raises(RuntimeError, match="exit 1"):
        render_wav(midi, sf2, tmp_path / "o.wav", run=run)


def test_render_wav_raises_when_fluidsynth_exits_zero_but_writes_nothing(tmp_path):
    """The failure mode that would otherwise become a silent empty
    transcription several hours into a 900-item arm."""
    midi = tmp_path / "p.mid"
    midi.write_bytes(b"x")
    sf2 = tmp_path / "s.sf2"
    sf2.write_bytes(b"x")

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="wrote no audio"):
        render_wav(midi, sf2, tmp_path / "o.wav", run=run)


def test_render_wav_raises_on_a_missing_soundfont(tmp_path):
    midi = tmp_path / "p.mid"
    midi.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="soundfont"):
        render_wav(midi, tmp_path / "absent.sf2", tmp_path / "o.wav",
                   run=_fake_run_writing(3))


# --------------------------------------------------------------------------
# render_wav -- the real binary
# --------------------------------------------------------------------------


real_fluidsynth = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or not SOUNDFONT.is_file(),
    reason="needs fluidsynth on PATH and data/weights/soundfonts/FluidR3_GM.sf2")


@real_fluidsynth
def test_real_fluidsynth_renders_audible_audio_at_the_contract_sample_rate(tmp_path):
    import numpy as np
    import soundfile as sf

    perf = Performance(
        notes=[{"pitch": 60 + 4 * i, "onset": 0.5 * i, "offset": 0.5 * i + 0.9,
                "velocity": 90} for i in range(6)],
        pedals=[{"time": 0.0, "value": 127}, {"time": 3.0, "value": 0}],
    )
    midi = tmp_path / "p.mid"
    write_performance_midi(perf, midi)
    wav = render_wav(midi, SOUNDFONT, tmp_path / "o.wav")

    audio, rate = sf.read(str(wav))
    assert rate == SAMPLE_RATE
    peak = float(np.abs(audio).max())
    # Audible, and not clipped -- a clipped render would be measuring the gain
    # setting rather than the soundfont.
    assert 0.01 < peak < 0.999


@real_fluidsynth
def test_real_fluidsynth_deadpan_arm_is_quieter_and_shorter(tmp_path):
    """The two arms must actually differ in the audio, not just in the MIDI.
    Dropping the pedal shortens the decay tail; flattening a loud performance
    to velocity 64 lowers the peak."""
    import numpy as np
    import soundfile as sf

    loud = Performance(
        notes=[{"pitch": 60 + 4 * i, "onset": 0.5 * i, "offset": 0.5 * i + 0.4,
                "velocity": 120} for i in range(6)],
        pedals=[{"time": 0.0, "value": 127}],
    )
    paths = {}
    for name, perf in (("expressive", loud), ("deadpan", deadpan(loud))):
        midi = tmp_path / f"{name}.mid"
        write_performance_midi(perf, midi)
        paths[name] = render_wav(midi, SOUNDFONT, tmp_path / f"{name}.wav")

    exp, _ = sf.read(str(paths["expressive"]))
    dead, _ = sf.read(str(paths["deadpan"]))
    assert float(np.abs(dead).max()) < float(np.abs(exp).max())
    assert len(dead) < len(exp)
