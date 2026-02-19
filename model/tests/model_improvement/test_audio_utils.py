import numpy as np
import pytest
from pathlib import Path

from model_improvement.audio_utils import load_audio, segment_audio, save_audio


class TestLoadAudio:
    def test_loads_wav_as_mono_float32(self, tmp_path):
        # Create a synthetic stereo WAV file
        import soundfile as sf
        sr = 44100
        stereo = np.random.randn(sr * 2, 2).astype(np.float32)  # 2s stereo
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), stereo, sr)

        audio, out_sr = load_audio(wav_path, target_sr=24000)
        assert out_sr == 24000
        assert audio.ndim == 1  # mono
        assert audio.dtype == np.float32
        # 2s at 44100 -> 2s at 24000 = ~48000 samples
        assert abs(len(audio) - 48000) < 500

    def test_loads_mono_wav_without_conversion(self, tmp_path):
        import soundfile as sf
        sr = 24000
        mono = np.random.randn(sr).astype(np.float32)  # 1s mono
        wav_path = tmp_path / "mono.wav"
        sf.write(str(wav_path), mono, sr)

        audio, out_sr = load_audio(wav_path, target_sr=24000)
        assert out_sr == 24000
        assert audio.ndim == 1
        assert abs(len(audio) - sr) < 10

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_audio(Path("/nonexistent/file.wav"))


class TestSegmentAudio:
    def test_segments_into_expected_count(self):
        sr = 24000
        audio = np.random.randn(sr * 95).astype(np.float32)  # 95 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        # 95s -> 30, 30, 30, 5 = 4 segments (last one is 5s, meets min)
        assert len(segments) == 4

    def test_drops_short_tail(self):
        sr = 24000
        audio = np.random.randn(sr * 62).astype(np.float32)  # 62 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        # 62s -> 30, 30, 2 = 2 segments (2s tail dropped, below min_duration)
        assert len(segments) == 2

    def test_segment_metadata_correct(self):
        sr = 24000
        audio = np.random.randn(sr * 60).astype(np.float32)  # 60 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0)
        assert segments[0]["start_sec"] == 0.0
        assert segments[0]["end_sec"] == 30.0
        assert segments[1]["start_sec"] == 30.0
        assert segments[1]["end_sec"] == 60.0
        assert len(segments[0]["audio"]) == sr * 30

    def test_short_audio_returns_single_segment(self):
        sr = 24000
        audio = np.random.randn(sr * 10).astype(np.float32)  # 10 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        assert len(segments) == 1
        assert segments[0]["start_sec"] == 0.0
        assert abs(segments[0]["end_sec"] - 10.0) < 0.01


class TestSaveAudio:
    def test_save_and_reload(self, tmp_path):
        sr = 24000
        # Use values in [-0.5, 0.5] to stay within WAV dynamic range
        audio = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
        path = tmp_path / "out.wav"
        save_audio(audio, path, sr=sr)
        assert path.exists()
        assert path.stat().st_size > 0

        loaded, loaded_sr = load_audio(path, target_sr=sr)
        assert loaded_sr == sr
        np.testing.assert_allclose(loaded, audio, atol=1e-4)
