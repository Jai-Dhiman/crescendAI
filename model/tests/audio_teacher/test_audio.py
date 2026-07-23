"""WAV validation behavior through validate_wav's public interface."""
from __future__ import annotations

import pytest

from audio_teacher.audio import validate_wav


def test_valid_mono_wav_passes_validation(wav_factory):
    path = wav_factory("clips/ok.wav", sample_rate=16000, seconds=2.0)
    info = validate_wav(path, expected_sample_rate=16000)
    assert info.sample_rate == 16000
    assert info.num_frames == 32000
    assert info.duration_seconds == pytest.approx(2.0)
