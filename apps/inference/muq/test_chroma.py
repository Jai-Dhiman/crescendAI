# apps/inference/muq/test_chroma.py
"""Pytest suite for chroma_feature."""
import struct
import numpy as np
import pytest

from chroma import chroma_feature

TARGET_SR = 24000
_22K_SR = 22050


def make_sine(freq_hz: float, duration_s: float = 1.0, sr: int = TARGET_SR) -> np.ndarray:
    """Generate a mono float32 sine wave at the given sample rate."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestChromaFeature:
    def test_output_is_tuple_of_three(self):
        y = make_sine(440.0)
        result = chroma_feature(y, TARGET_SR)
        assert isinstance(result, tuple) and len(result) == 3
        b, n, rate = result
        assert isinstance(b, (bytes, bytearray))
        assert isinstance(n, int)
        assert isinstance(rate, float)

    def test_frame_count_matches_bytes_length(self):
        y = make_sine(440.0)
        b, n, _ = chroma_feature(y, TARGET_SR)
        # bytes = 12 rows * n frames * 4 bytes per float32
        assert len(b) == 12 * n * 4

    def test_frame_count_matches_librosa_expectation(self):
        duration_s = 1.0
        y = make_sine(440.0, duration_s)
        hop = max(1, round(TARGET_SR / 50))
        _, n, _ = chroma_feature(y, TARGET_SR)
        expected_n = int(np.ceil(len(y) / hop))
        # Allow +-1 frame tolerance for rounding
        assert abs(n - expected_n) <= 1

    def test_output_is_row_major_float32(self):
        y = make_sine(440.0)
        b, n, _ = chroma_feature(y, TARGET_SR)
        floats = struct.unpack(f"<{12 * n}f", b)
        # All values should be finite
        arr = np.array(floats, dtype=np.float32).reshape(12, n)
        assert np.all(np.isfinite(arr))
        # Column norms should be ~1.0 (L2-normalized)
        col_norms = np.linalg.norm(arr, axis=0)
        np.testing.assert_allclose(col_norms, 1.0, atol=0.01)

    def test_dominant_pitch_class_for_a440(self):
        # A440 = MIDI pitch 69, pitch class 9 (A)
        y = make_sine(440.0, 2.0)
        b, n, _ = chroma_feature(y, TARGET_SR)
        arr = np.frombuffer(b, dtype="<f4").reshape(12, n)
        dominant_pc = int(arr.mean(axis=1).argmax())
        assert dominant_pc == 9, f"Expected pitch class 9 (A), got {dominant_pc}"

    def test_frame_rate_at_24khz(self):
        # At 24000 Hz: hop = round(24000 / 50) = 480, frame_rate = 24000/480 = 50.0
        y = make_sine(440.0, 1.0, sr=TARGET_SR)
        _, _, rate = chroma_feature(y, TARGET_SR)
        hop = max(1, round(TARGET_SR / 50))
        expected_rate = TARGET_SR / hop
        assert rate == pytest.approx(expected_rate), (
            f"Expected frame_rate_hz={expected_rate}, got {rate}"
        )

    def test_frame_rate_at_22khz(self):
        # At 22050 Hz: hop = round(22050 / 50) = 441, frame_rate = 22050/441 = 50.0
        y = make_sine(440.0, 1.0, sr=_22K_SR)
        _, _, rate = chroma_feature(y, _22K_SR)
        hop = max(1, round(_22K_SR / 50))
        expected_rate = _22K_SR / hop
        assert rate == pytest.approx(expected_rate)


class TestHandlerChromaIntegration:
    """Verify EndpointHandler.__call__ response includes chroma fields."""

    def test_handler_call_returns_chroma_fields(self, monkeypatch):
        """Invoke EndpointHandler().__call__ with a synthetic audio dict and assert
        that the returned response contains chroma_b64 (str), chroma_frames (int > 0),
        and chroma_frame_rate_hz at the value produced by TARGET_SR=24000.
        MuQ model loading is monkeypatched out so the test runs without GPU."""
        import base64
        import handler as handler_module
        from handler import EndpointHandler

        # Use TARGET_SR (24000 Hz) to match the production audio path
        y_synth = make_sine(440.0, 1.0, sr=TARGET_SR)

        h = EndpointHandler.__new__(EndpointHandler)

        # Minimal mock cache: muq_model must be truthy to pass the guard check;
        # muq_heads must be iterable for len() in model_info.
        class _FakeCache:
            muq_model = object()
            muq_heads = [None]

        h._cache = _FakeCache()

        def fake_load_audio(inputs, max_duration):
            return y_synth, 1.0

        stub_predictions = {
            "dynamics": 0.6,
            "timing": 0.7,
            "pedaling": 0.5,
            "articulation": 0.8,
            "phrasing": 0.65,
            "interpretation": 0.72,
        }

        stub_arr = np.array([0.6, 0.7, 0.5, 0.8, 0.65, 0.72], dtype=np.float32)

        def fake_extract_embeddings(audio, cache):
            return stub_arr

        def fake_predict_with_ensemble(embeddings, cache):
            return stub_arr

        def fake_predictions_to_dict(preds):
            return stub_predictions

        monkeypatch.setattr(h, "_load_audio", fake_load_audio)
        monkeypatch.setattr(h, "_predictions_to_dict", fake_predictions_to_dict)
        monkeypatch.setattr(handler_module, "extract_muq_embeddings", fake_extract_embeddings)
        monkeypatch.setattr(handler_module, "predict_with_ensemble", fake_predict_with_ensemble)

        inputs = {"audio": {"data": b"", "mime_type": "audio/wav"}}
        result = h(inputs)

        assert "chroma_b64" in result, f"handler response missing chroma_b64, got keys: {list(result.keys())}"
        assert "chroma_frames" in result, "handler response missing chroma_frames"
        assert "chroma_frame_rate_hz" in result, "handler response missing chroma_frame_rate_hz"
        assert isinstance(result["chroma_b64"], str), "chroma_b64 must be a str"
        assert result["chroma_frames"] > 0, "chroma_frames must be > 0"

        # At 24000 Hz: hop = round(24000/50) = 480, actual frame rate = 24000/480 = 50.0
        hop = max(1, round(TARGET_SR / 50))
        expected_rate = TARGET_SR / hop
        assert result["chroma_frame_rate_hz"] == pytest.approx(expected_rate), (
            f"chroma_frame_rate_hz expected {expected_rate}, got {result['chroma_frame_rate_hz']}"
        )

        n_frames = result["chroma_frames"]
        decoded = base64.b64decode(result["chroma_b64"])
        assert len(decoded) == 12 * n_frames * 4, (
            f"decoded chroma bytes length {len(decoded)} != 12 * {n_frames} * 4"
        )
