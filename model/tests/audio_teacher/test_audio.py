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


def _make_stereo(wav_factory):
    return wav_factory("clips/stereo.wav", channels=2)


def _make_wrong_rate(wav_factory):
    return wav_factory("clips/rate44k.wav", sample_rate=44100)


def _make_truncated(wav_factory):
    path = wav_factory("clips/trunc.wav", seconds=1.0)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 1000])
    return path


@pytest.mark.parametrize(
    "make_bad", [_make_stereo, _make_wrong_rate, _make_truncated],
    ids=["stereo", "wrong_rate", "truncated"],
)
def test_malformed_wav_aborts_naming_the_file(wav_factory, make_bad):
    from audio_teacher.audio import MalformedClipError

    path = make_bad(wav_factory)
    with pytest.raises(MalformedClipError) as excinfo:
        validate_wav(path, expected_sample_rate=16000)
    assert path.name in str(excinfo.value)
