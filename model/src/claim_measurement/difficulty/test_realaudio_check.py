"""Tests for realaudio_check (#149 / #138 Phase 1) -- the real-audio second
gate: MIDI drift, resumable transcription, and per-fold audio scoring.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest

from claim_measurement.difficulty.realaudio_check import midi_drift


def test_midi_drift_computes_note_count_delta_and_onset_f1_with_tolerance_matching():
    reference = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
                 {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80}]

    identical = midi_drift(reference, reference, onset_tolerance=0.05)
    assert identical == {"note_count_delta": 0, "onset_f1": 1.0}

    candidate = [{"pitch": 60, "onset": 0.20, "offset": 0.5, "velocity": 80},  # onset shifted past tolerance
                 {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80},
                 {"pitch": 67, "onset": 2.0, "offset": 2.5, "velocity": 80}]  # extra note
    degraded = midi_drift(reference, candidate, onset_tolerance=0.05)
    assert degraded["note_count_delta"] == 1
    assert degraded["onset_f1"] == pytest.approx(2 / 5)  # tp=1, precision=1/3, recall=1/2


from claim_measurement.difficulty.realaudio_check import main


def test_transcribe_stage_skips_pieces_whose_cache_file_already_exists(tmp_path):
    wav_manifest = tmp_path / "wav_manifest.json"
    wav_manifest.write_text(json.dumps([
        {"seg_id": "already_done", "wav_path": str(tmp_path / "a.wav")},
        {"seg_id": "new_piece", "wav_path": str(tmp_path / "b.wav")},
    ]))
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    (out_dir / "already_done.json").write_text(json.dumps({"notes": [], "pedals": []}))

    calls = []

    def fake_transcriber(wav_path):
        calls.append(wav_path)
        return ([{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}], [])

    exit_code = main(["--wav-manifest", str(wav_manifest), "--out-dir", str(out_dir)],
                      transcriber=fake_transcriber)

    assert exit_code == 0
    # only the not-yet-cached piece was transcribed
    assert calls == [tmp_path / "b.wav"]
    cached = json.loads((out_dir / "new_piece.json").read_text())
    assert cached["notes"][0]["pitch"] == 60
