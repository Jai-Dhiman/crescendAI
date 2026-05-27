# apps/inference/muq/test_chroma.py
"""Pytest suite for chroma_feature."""
import struct
import numpy as np
import pytest

from chroma import chroma_feature

SR = 22050
HOP = 441


def make_sine(freq_hz: float, duration_s: float = 1.0) -> np.ndarray:
    """Generate a mono float32 sine wave at 22050 Hz."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestChromaFeature:
    def test_output_is_tuple_of_bytes_and_int(self):
        y = make_sine(440.0)
        result = chroma_feature(y, SR)
        assert isinstance(result, tuple) and len(result) == 2
        b, n = result
        assert isinstance(b, (bytes, bytearray))
        assert isinstance(n, int)

    def test_frame_count_matches_bytes_length(self):
        y = make_sine(440.0)
        b, n = chroma_feature(y, SR)
        # bytes = 12 rows * n frames * 4 bytes per float32
        assert len(b) == 12 * n * 4

    def test_frame_count_matches_librosa_expectation(self):
        duration_s = 1.0
        y = make_sine(440.0, duration_s)
        _, n = chroma_feature(y, SR)
        expected_n = int(np.ceil(len(y) / HOP))
        # Allow +-1 frame tolerance for rounding
        assert abs(n - expected_n) <= 1

    def test_output_is_row_major_float32(self):
        y = make_sine(440.0)
        b, n = chroma_feature(y, SR)
        floats = struct.unpack(f"<{12 * n}f", b)
        # All values should be finite and in [0, 1] (L2-normalized columns)
        arr = np.array(floats, dtype=np.float32).reshape(12, n)
        assert np.all(np.isfinite(arr))
        # Column norms should be ~1.0 (L2-normalized)
        col_norms = np.linalg.norm(arr, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=0.01)

    def test_dominant_pitch_class_for_a440(self):
        # A440 = MIDI pitch 69, pitch class 9 (A)
        y = make_sine(440.0, 2.0)
        b, n = chroma_feature(y, SR)
        arr = np.frombuffer(b, dtype="<f4").reshape(12, n)
        dominant_pc = int(arr.mean(axis=1).argmax())
        assert dominant_pc == 9, f"Expected pitch class 9 (A), got {dominant_pc}"
