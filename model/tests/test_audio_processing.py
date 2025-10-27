"""
Unit tests for audio processing module.

Tests the production-ready audio pipeline using raw waveforms at 24kHz.
"""

import numpy as np
import pytest
import tempfile
import soundfile as sf
from pathlib import Path

from src.data.audio_processing import (
    load_audio,
    compute_cqt,
    segment_audio,
    normalize_audio,
    get_audio_duration,
    preprocess_audio_file,
)


# ==================== Helper Functions ====================


def create_test_audio_file(duration=2.0, sr=24000, freq=440.0):
    """Create a temporary audio file for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)

    return temp_file.name, audio, sr


# ==================== load_audio Tests ====================


def test_load_audio_default_sr():
    """Test that load_audio defaults to 24kHz (MERT requirement)."""
    audio_path, expected_audio, _ = create_test_audio_file(duration=1.0, sr=24000)

    try:
        audio, sr = load_audio(audio_path)

        # Check sample rate is 24kHz by default
        assert sr == 24000
        assert len(audio) == 24000  # 1 second at 24kHz
    finally:
        Path(audio_path).unlink()


def test_load_audio_custom_sr():
    """Test loading audio with custom sample rate."""
    audio_path, _, _ = create_test_audio_file(duration=1.0, sr=44100)

    try:
        # Load at 16kHz
        audio, sr = load_audio(audio_path, sr=16000)

        assert sr == 16000
        assert len(audio) == 16000  # 1 second at 16kHz
    finally:
        Path(audio_path).unlink()


def test_load_audio_duration():
    """Test loading only part of an audio file."""
    audio_path, _, _ = create_test_audio_file(duration=5.0, sr=24000)

    try:
        # Load only 2 seconds
        audio, sr = load_audio(audio_path, duration=2.0)

        assert sr == 24000
        assert len(audio) == pytest.approx(48000, rel=0.01)  # 2 seconds
    finally:
        Path(audio_path).unlink()


def test_load_audio_offset():
    """Test loading audio with offset."""
    audio_path, _, _ = create_test_audio_file(duration=5.0, sr=24000)

    try:
        # Start from 1 second, load 2 seconds
        audio, sr = load_audio(audio_path, duration=2.0, offset=1.0)

        assert sr == 24000
        assert len(audio) == pytest.approx(48000, rel=0.01)  # 2 seconds
    finally:
        Path(audio_path).unlink()


def test_load_audio_mono():
    """Test mono conversion."""
    # Create stereo audio
    audio_stereo = np.random.randn(2, 48000)  # 2 channels, 2 seconds
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio_stereo.T, 24000)

    try:
        audio, sr = load_audio(temp_file.name, mono=True)

        # Check it's mono (1D array)
        assert audio.ndim == 1
        assert sr == 24000
    finally:
        Path(temp_file.name).unlink()


# ==================== compute_cqt Tests ====================


def test_compute_cqt_default():
    """Test CQT computation with default parameters."""
    audio = np.random.randn(24000)  # 1 second at 24kHz

    cqt = compute_cqt(audio, sr=24000)

    # Check shape: [n_bins, time_frames]
    assert cqt.shape[0] == 168  # 7 octaves × 24 bins/octave
    assert cqt.shape[1] > 0  # Has time frames

    # Check it's in dB scale (negative values)
    assert cqt.max() <= 0


def test_compute_cqt_custom_bins():
    """Test CQT with custom frequency bins."""
    audio = np.random.randn(24000)

    cqt = compute_cqt(audio, sr=24000, n_bins=84, bins_per_octave=12)

    # Check shape
    assert cqt.shape[0] == 84  # Custom bins


def test_compute_cqt_not_nan():
    """Test that CQT doesn't produce NaN values."""
    audio = np.random.randn(48000)  # 2 seconds

    cqt = compute_cqt(audio, sr=24000)

    assert not np.isnan(cqt).any()
    assert not np.isinf(cqt).any()


# ==================== segment_audio Tests ====================


def test_segment_audio_default():
    """Test audio segmentation with default parameters."""
    audio = np.random.randn(240000)  # 10 seconds at 24kHz

    segments = segment_audio(audio, sr=24000, segment_len=10.0, overlap=0.5)

    # Should have 1 segment for exactly 10 seconds
    assert len(segments) == 1
    assert len(segments[0]) == 240000


def test_segment_audio_multiple():
    """Test segmentation with multiple segments."""
    audio = np.random.randn(720000)  # 30 seconds at 24kHz

    segments = segment_audio(audio, sr=24000, segment_len=10.0, overlap=0.0)

    # Should have 3 segments (no overlap)
    assert len(segments) == 3
    for seg in segments:
        assert len(seg) == 240000  # 10 seconds


def test_segment_audio_with_overlap():
    """Test segmentation with overlap."""
    audio = np.random.randn(480000)  # 20 seconds at 24kHz

    segments = segment_audio(audio, sr=24000, segment_len=10.0, overlap=0.5)

    # With 50% overlap: start at 0, 5, 10
    assert len(segments) >= 2


def test_segment_audio_padding():
    """Test that last segment is padded if needed."""
    audio = np.random.randn(366000)  # 15.25 seconds at 24kHz

    segments = segment_audio(audio, sr=24000, segment_len=10.0, overlap=0.0)

    # First segment: 0-10s, second would start at 10s
    # Remaining: 5.25s > 50% of 10s (5s), so it should be included and padded
    assert len(segments) == 2
    assert len(segments[0]) == 240000  # Full 10 seconds
    assert len(segments[1]) == 240000  # Padded to 10 seconds


# ==================== normalize_audio Tests ====================


def test_normalize_audio_default():
    """Test audio normalization to default -3dB."""
    audio = np.random.randn(24000) * 0.1  # Quiet audio

    normalized = normalize_audio(audio)

    # Peak should be close to -3dB (10^(-3/20) ≈ 0.708)
    peak = np.abs(normalized).max()
    target = 10 ** (-3.0 / 20.0)
    assert peak == pytest.approx(target, rel=0.01)


def test_normalize_audio_custom_db():
    """Test normalization to custom dB level."""
    audio = np.random.randn(24000) * 0.5

    normalized = normalize_audio(audio, target_db=-6.0)

    # Peak should be close to -6dB (10^(-6/20) ≈ 0.501)
    peak = np.abs(normalized).max()
    target = 10 ** (-6.0 / 20.0)
    assert peak == pytest.approx(target, rel=0.01)


def test_normalize_audio_silent():
    """Test normalization doesn't break on silent audio."""
    audio = np.zeros(24000)

    normalized = normalize_audio(audio)

    # Should remain silent
    assert np.allclose(normalized, 0.0)


def test_normalize_audio_preserves_shape():
    """Test that normalization preserves array shape."""
    audio = np.random.randn(48000)

    normalized = normalize_audio(audio)

    assert normalized.shape == audio.shape


# ==================== get_audio_duration Tests ====================


def test_get_audio_duration():
    """Test getting audio duration without loading."""
    audio_path, _, _ = create_test_audio_file(duration=3.5, sr=24000)

    try:
        duration = get_audio_duration(audio_path)

        assert duration == pytest.approx(3.5, rel=0.01)
    finally:
        Path(audio_path).unlink()


# ==================== preprocess_audio_file Tests ====================


def test_preprocess_audio_file_default():
    """Test complete preprocessing pipeline with defaults."""
    audio_path, _, _ = create_test_audio_file(duration=25.0, sr=24000)

    try:
        result = preprocess_audio_file(audio_path)

        # Check result structure
        assert 'audio' in result
        assert 'segments' in result
        assert 'sr' in result
        assert 'duration' in result

        # Check sample rate is 24kHz (production default)
        assert result['sr'] == 24000

        # Check duration
        assert result['duration'] == pytest.approx(25.0, rel=0.01)

        # Check segments exist
        assert len(result['segments']) > 0

        # CQT should NOT be computed by default (production mode)
        assert 'cqt_segments' not in result

    finally:
        Path(audio_path).unlink()


def test_preprocess_audio_file_with_cqt():
    """Test preprocessing with CQT computation enabled."""
    audio_path, _, _ = create_test_audio_file(duration=12.0, sr=24000)

    try:
        result = preprocess_audio_file(audio_path, compute_cqt_specs=True)

        # CQT should be computed when explicitly requested
        assert 'cqt_segments' in result
        assert len(result['cqt_segments']) == len(result['segments'])

        # Check CQT shape
        for cqt in result['cqt_segments']:
            assert cqt.shape[0] == 168  # Piano range bins

    finally:
        Path(audio_path).unlink()


def test_preprocess_audio_file_normalization():
    """Test preprocessing with normalization."""
    audio_path, original_audio, _ = create_test_audio_file(duration=2.0, sr=24000)

    try:
        result = preprocess_audio_file(audio_path, normalize=True)

        # Check audio is normalized
        peak = np.abs(result['audio']).max()
        target = 10 ** (-3.0 / 20.0)  # -3dB
        assert peak == pytest.approx(target, rel=0.01)

    finally:
        Path(audio_path).unlink()


def test_preprocess_audio_file_custom_sr():
    """Test preprocessing with custom sample rate."""
    audio_path, _, _ = create_test_audio_file(duration=2.0, sr=44100)

    try:
        result = preprocess_audio_file(audio_path, sr=16000)

        # Check resampling to 16kHz
        assert result['sr'] == 16000
        assert len(result['audio']) == 32000  # 2 seconds at 16kHz

    finally:
        Path(audio_path).unlink()


def test_preprocess_audio_file_segmentation():
    """Test preprocessing segmentation parameters."""
    audio_path, _, _ = create_test_audio_file(duration=30.0, sr=24000)

    try:
        result = preprocess_audio_file(
            audio_path,
            segment_len=5.0,
            overlap=0.0
        )

        # Should have 6 segments (30 / 5 = 6)
        assert len(result['segments']) == 6

        # Each segment should be 5 seconds = 120000 samples
        for seg in result['segments']:
            assert len(seg) == 120000

    finally:
        Path(audio_path).unlink()


# ==================== Integration Tests ====================


def test_full_pipeline_raw_waveforms():
    """Test that pipeline returns raw waveforms for MERT."""
    audio_path, _, _ = create_test_audio_file(duration=10.0, sr=24000)

    try:
        result = preprocess_audio_file(audio_path)

        # Verify we get raw waveforms
        assert isinstance(result['audio'], np.ndarray)
        assert result['audio'].ndim == 1  # 1D waveform
        assert result['sr'] == 24000  # MERT sample rate

        # Segments should also be raw waveforms
        for seg in result['segments']:
            assert isinstance(seg, np.ndarray)
            assert seg.ndim == 1

    finally:
        Path(audio_path).unlink()


def test_production_config():
    """Test that production configuration is correct."""
    audio_path, _, _ = create_test_audio_file(duration=5.0, sr=48000)

    try:
        # Use defaults (production mode)
        result = preprocess_audio_file(audio_path)

        # Check production defaults
        assert result['sr'] == 24000  # MERT requirement
        assert 'cqt_segments' not in result  # CQT disabled by default
        assert 'segments' in result  # Raw waveform segments present

    finally:
        Path(audio_path).unlink()
