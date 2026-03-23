"""Tests for score fingerprinting (N-gram index + rerank features)."""

from __future__ import annotations

import pytest

from score_library.fingerprint import (
    compute_rerank_features,
    extract_pitch_trigrams,
)


class TestExtractPitchTrigrams:
    def test_extract_pitch_trigrams_basic(self) -> None:
        """5 pitches produce exactly 3 consecutive trigrams."""
        pitches = [60, 62, 64, 65, 67]
        result = extract_pitch_trigrams(pitches)
        assert result == [(60, 62, 64), (62, 64, 65), (64, 65, 67)]

    def test_extract_pitch_trigrams_too_short(self) -> None:
        """Fewer than 3 pitches produce an empty list."""
        assert extract_pitch_trigrams([]) == []
        assert extract_pitch_trigrams([60]) == []
        assert extract_pitch_trigrams([60, 62]) == []

    def test_extract_pitch_trigrams_exactly_three(self) -> None:
        """Exactly 3 pitches produce exactly 1 trigram."""
        result = extract_pitch_trigrams([60, 64, 67])
        assert result == [(60, 64, 67)]


class TestComputeRerankFeatures:
    def test_compute_rerank_features_shape(self) -> None:
        """Output is exactly 128 floats."""
        notes = [
            {"pitch": 60, "onset_seconds": 0.0, "velocity": 80},
            {"pitch": 64, "onset_seconds": 0.5, "velocity": 70},
            {"pitch": 67, "onset_seconds": 1.0, "velocity": 75},
        ]
        result = compute_rerank_features(notes)
        assert len(result) == 128
        assert all(isinstance(v, float) for v in result)

    def test_compute_rerank_features_pitch_class_histogram(self) -> None:
        """All-C notes: features[0] == 1.0, features[1:12] all 0.0."""
        # Pitch 60 = C4 (pitch class 0), pitch 72 = C5, pitch 48 = C3
        notes = [
            {"pitch": 60, "onset_seconds": 0.0, "velocity": 80},
            {"pitch": 72, "onset_seconds": 0.5, "velocity": 80},
            {"pitch": 48, "onset_seconds": 1.0, "velocity": 80},
        ]
        result = compute_rerank_features(notes)
        assert result[0] == pytest.approx(1.0)
        for i in range(1, 12):
            assert result[i] == pytest.approx(0.0)

    def test_compute_rerank_features_reserved_zeros(self) -> None:
        """Indices [82:128] are reserved and must be zero-padded."""
        notes = [
            {"pitch": 60, "onset_seconds": 0.0, "velocity": 64},
            {"pitch": 62, "onset_seconds": 0.25, "velocity": 64},
        ]
        result = compute_rerank_features(notes)
        for i in range(82, 128):
            assert result[i] == pytest.approx(0.0), f"Index {i} should be 0.0"

    def test_compute_rerank_features_single_note(self) -> None:
        """Single note: normalized histograms should still sum to 1 where applicable."""
        notes = [{"pitch": 69, "onset_seconds": 0.0, "velocity": 100}]
        result = compute_rerank_features(notes)
        assert len(result) == 128
        # pitch class 9 = A
        assert result[9] == pytest.approx(1.0)
        # all other pitch classes zero
        for i in range(12):
            if i != 9:
                assert result[i] == pytest.approx(0.0)
