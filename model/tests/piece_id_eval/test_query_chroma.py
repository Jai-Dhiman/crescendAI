"""Verify audio_to_chroma and window_chroma against known signal properties."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from piece_id_eval.query_chroma import audio_to_chroma, window_chroma


def _write_sine_wav(path: Path, freq_hz: float, duration_sec: float, sr: int = 16000) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    sf.write(path, y, sr)


def test_audio_to_chroma_shape(tmp_path: Path) -> None:
    # 4 seconds of audio at 16 kHz -> ~200 chroma frames at 50 Hz
    _write_sine_wav(tmp_path / "tone.wav", freq_hz=261.63, duration_sec=4.0)
    chroma, frame_rate_hz = audio_to_chroma(tmp_path / "tone.wav")
    assert chroma.shape[0] == 12
    assert chroma.shape[1] > 0
    assert chroma.dtype == np.float32
    assert 45.0 <= frame_rate_hz <= 55.0  # target ~50 Hz


def test_audio_to_chroma_c4_dominant_pitch_class(tmp_path: Path) -> None:
    # C4 = 261.63 Hz -> pitch class 0
    _write_sine_wav(tmp_path / "c4.wav", freq_hz=261.63, duration_sec=2.0)
    chroma, _ = audio_to_chroma(tmp_path / "c4.wav")
    # Most columns should have pitch-class 0 (C) as the maximum
    dominant_pcs = np.argmax(chroma, axis=0)
    # Allow some tolerance for CQT edge effects
    assert (dominant_pcs == 0).mean() > 0.5, f"expected C dominant, got {np.bincount(dominant_pcs)}"


def test_audio_to_chroma_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio not found"):
        audio_to_chroma(tmp_path / "nonexistent.wav")


def test_window_chroma_count(tmp_path: Path) -> None:
    # 10-second signal at 50 Hz = 500 frames
    # window=2s hop=1s -> windows start at 0,1,2,...,8 sec = 9 windows
    sr = 16000
    hop_target = max(1, round(sr / 50))
    frame_rate_hz = sr / hop_target
    n_frames = int(10.0 * frame_rate_hz)
    chroma = np.random.RandomState(0).rand(12, n_frames).astype(np.float32)
    windows = window_chroma(chroma, frame_rate_hz, window_seconds=2.0, hop_seconds=1.0)
    # windows start at 0,1,...,8 (last window at 8 ends at 10)
    assert len(windows) >= 8
    assert all(w.shape[0] == 12 for w in windows)


def test_window_chroma_each_window_correct_length(tmp_path: Path) -> None:
    sr = 16000
    hop_target = max(1, round(sr / 50))
    frame_rate_hz = sr / hop_target
    n_frames = int(8.0 * frame_rate_hz)
    chroma = np.ones((12, n_frames), dtype=np.float32)
    windows = window_chroma(chroma, frame_rate_hz, window_seconds=2.0, hop_seconds=2.0)
    expected_len = int(2.0 * frame_rate_hz)
    for w in windows:
        assert w.shape == (12, expected_len), f"unexpected window shape {w.shape}"
