"""Behavior tests for the Transkun-backed EndpointHandler + helpers.

Run: cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi \
        --with fastapi --with pytest pytest test_transcription.py
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcription
import transkun_cli


def test_build_response_has_frozen_contract_shape():
    notes = [
        {"pitch": 60, "onset": 0.1, "offset": 0.5, "velocity": 70},
        {"pitch": 72, "onset": 0.2, "offset": 0.6, "velocity": 90},
    ]
    pedals = [{"time": 0.15, "value": 127}]
    resp = transcription.build_response(notes, pedals, chunk_duration_s=15.0, elapsed_ms=42)

    assert resp["midi_notes"] == notes
    assert resp["pedal_events"] == pedals
    info = resp["transcription_info"]
    assert info["note_count"] == 2
    assert info["pitch_range"] == [60, 72]
    assert info["pedal_event_count"] == 1
    assert info["transcription_time_ms"] == 42
    assert info["chunk_duration_s"] == 15.0


def test_build_response_empty_notes_pitch_range_zero():
    resp = transcription.build_response([], [], chunk_duration_s=0.0, elapsed_ms=1)
    assert resp["transcription_info"]["pitch_range"] == [0, 0]
    assert resp["midi_notes"] == []


def test_resolve_transcriber_refuses_when_no_path(monkeypatch):
    # Force the warm import to fail AND the CLI probe to fail.
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda _cmd: None)
    with pytest.raises(RuntimeError):
        transcription.resolve_transcriber()


def test_resolve_transcriber_falls_back_to_cli(monkeypatch):
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")
    fn = transcription.resolve_transcriber()
    assert fn is transkun_cli.transcribe_pcm


def test_handler_missing_chunk_returns_error_body(monkeypatch):
    # CLI-fallback resolution keeps __init__ fast (no model load).
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")
    handler = transcription.EndpointHandler(path="")
    out = handler({"inputs": {}})
    assert out["error"]["code"] == "MISSING_CHUNK_AUDIO"


def test_handler_transcribes_chunk_and_ignores_context(monkeypatch):
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")

    seen = {}
    def _fake_transcribe(pcm):
        seen["len"] = len(pcm)
        return ([{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}],
                [{"time": 0.1, "value": 127}])

    handler = transcription.EndpointHandler(path="")
    handler._transcribe_fn = _fake_transcribe
    monkeypatch.setattr(transcription, "decode_webm_to_pcm",
                        lambda b: np.zeros(transcription.SAMPLE_RATE, dtype=np.float32))

    out = handler({"inputs": {"chunk_audio": base64.b64encode(b"x").decode(),
                              "context_audio": base64.b64encode(b"y").decode()}})
    # context_audio must NOT be concatenated: transcriber sees exactly 1s of chunk PCM.
    assert seen["len"] == transcription.SAMPLE_RATE
    assert out["transcription_info"]["note_count"] == 1
    assert out["pedal_events"] == [{"time": 0.1, "value": 127}]
