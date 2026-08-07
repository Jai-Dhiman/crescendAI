"""Behavior tests for the shared Transkun shell-out helper.

Run: cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile \
        --with pytest pytest test_transkun_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest
import soundfile as sf

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


def test_transcribe_pcm_on_real_sample_returns_notes_with_velocity():
    """Real Transkun on the committed piano fixture. Slow: downloads weights once.

    The fixture `apps/inference/amt/fixtures/piano_sample_5s_16k.wav` is a real
    ~5s mono 16kHz piano clip committed to the repo (force-added past the `*.wav`
    .gitignore rule), so it is GUARANTEED present in every fresh checkout/worktree.
    This test FAILS HARD (never pytest.skip) when the fixture is absent — a missing
    fixture is a real regression that must break the build, not silently pass.
    """
    wav = Path(__file__).resolve().parent / "fixtures" / "piano_sample_5s_16k.wav"
    if not wav.exists():
        raise AssertionError(
            f"required committed fixture missing: {wav} "
            "(force-add it: git add -f apps/inference/amt/fixtures/piano_sample_5s_16k.wav)"
        )
    y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != transkun_cli.SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), transkun_cli.SAMPLE_RATE)
        y = resample_poly(y, transkun_cli.SAMPLE_RATE // g, int(sr) // g).astype("float32")

    notes, pedals = transkun_cli.transcribe_pcm(y)

    assert len(notes) > 0
    assert all(set(n) == {"pitch", "onset", "offset", "velocity"} for n in notes)
    assert all(isinstance(n["velocity"], int) and n["velocity"] > 0 for n in notes)
    assert all(isinstance(p["value"], int) and p["value"] in (0, 127) for p in pedals)


# --------------------------------------------------------------------------
# Device plumbing (#166): transcription owns the per-second term of the MIREX
# 24h budget, and the subprocess boundary means the caller's torch device
# reaches Transkun only if it is threaded explicitly.
# --------------------------------------------------------------------------


def _capture_argv(monkeypatch):
    """Run _run_transkun against a fake subprocess and return the argv it built."""
    seen = {}

    class _Proc:
        returncode = 0
        stderr = b""

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        # _run_transkun asserts the MIDI exists after a clean exit.
        Path(cmd[-3]).write_bytes(b"")
        return _Proc()

    monkeypatch.setattr(transkun_cli.subprocess, "run", _fake_run)
    return seen


def test_device_defaults_to_cpu(tmp_path, monkeypatch):
    """Every non-MIREX caller must keep the previous behavior byte for byte."""
    seen = _capture_argv(monkeypatch)
    out_mid = tmp_path / "o.mid"
    (tmp_path / "in.wav").write_bytes(b"")

    transkun_cli._run_transkun(tmp_path / "in.wav", out_mid)

    cmd = seen["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_device_argument_reaches_the_transkun_subprocess(tmp_path, monkeypatch):
    seen = _capture_argv(monkeypatch)
    out_mid = tmp_path / "o.mid"
    (tmp_path / "in.wav").write_bytes(b"")

    transkun_cli._run_transkun(tmp_path / "in.wav", out_mid, device="cuda")

    cmd = seen["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cuda"


def test_transcribe_wav_forwards_device(tmp_path, monkeypatch):
    """The public entry point, not just the private helper -- what score_wav calls."""
    midi_path = tmp_path / "src.mid"
    _write_midi(midi_path, notes=[(60, 0.0, 1.0, 80)], pedal_ccs=[])
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"")
    seen = {}

    def _fake_run_transkun(in_wav, out_mid, device=transkun_cli.DEFAULT_DEVICE):
        seen["device"] = device
        Path(out_mid).write_bytes(midi_path.read_bytes())

    monkeypatch.setattr(transkun_cli, "_run_transkun", _fake_run_transkun)

    notes, _ = transkun_cli.transcribe_wav(wav, device="cuda")

    assert seen["device"] == "cuda"
    assert len(notes) == 1
