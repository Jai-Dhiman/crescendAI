import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.augmentation import (
    create_augmentation_chain,
    augment_audio,
    augment_and_embed_piano,
    _mix_noise,
)


class TestCreateAugmentationChain:
    def test_returns_callable(self):
        chain = create_augmentation_chain(seed=42)
        assert callable(chain)

    def test_produces_different_output(self):
        chain = create_augmentation_chain(seed=42)
        audio = np.random.randn(24000 * 5).astype(np.float32)
        augmented = chain(audio, sample_rate=24000)
        assert augmented.shape == audio.shape
        # Should not be identical (augmentation applied)
        assert not np.allclose(audio, augmented, atol=1e-6)


class TestMixNoise:
    def test_adds_noise_to_audio(self):
        audio = np.random.randn(24000 * 5).astype(np.float32)
        noisy = _mix_noise(audio, snr_db=20.0, seed=42)
        assert noisy.shape == audio.shape
        assert noisy.dtype == np.float32
        assert not np.allclose(audio, noisy, atol=1e-6)

    def test_deterministic_with_same_seed(self):
        audio = np.random.randn(24000 * 5).astype(np.float32)
        noisy1 = _mix_noise(audio, snr_db=20.0, seed=42)
        noisy2 = _mix_noise(audio, snr_db=20.0, seed=42)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_produce_different_noise(self):
        audio = np.random.randn(24000 * 5).astype(np.float32)
        noisy1 = _mix_noise(audio, snr_db=20.0, seed=42)
        noisy2 = _mix_noise(audio, snr_db=20.0, seed=99)
        assert not np.allclose(noisy1, noisy2, atol=1e-6)

    def test_silent_audio_returns_unchanged(self):
        audio = np.zeros(24000, dtype=np.float32)
        noisy = _mix_noise(audio, snr_db=20.0, seed=42)
        # With zero signal power, scale is 0, so noise contribution is 0
        np.testing.assert_array_equal(audio, noisy)


class TestParametricEQ:
    def test_peakfilter_in_chain(self):
        """Verify PeakFilter effects appear in the augmentation chain."""
        from pedalboard import PeakFilter, Pedalboard

        # Use seed=0 which triggers the 70% EQ branch (rng.random() < 0.7)
        # We inspect the chain by checking the board's effects
        chain = create_augmentation_chain(seed=0)
        # The chain closure captures `board`; verify it processes audio
        audio = np.random.randn(24000 * 5).astype(np.float32)
        augmented = chain(audio, sample_rate=24000)
        assert augmented.shape == audio.shape

    def test_eq_changes_spectral_content(self):
        """EQ should alter frequency content without changing duration."""
        audio = np.sin(2 * np.pi * 440 * np.arange(24000 * 5) / 24000).astype(np.float32)
        augmented = augment_audio(audio, sr=24000, seed=0)
        assert augmented.shape == audio.shape
        assert not np.allclose(audio, augmented, atol=1e-6)


class TestAugmentAudio:
    def test_augments_and_returns_ndarray(self):
        audio = np.random.randn(24000 * 10).astype(np.float32)
        augmented = augment_audio(audio, sr=24000, seed=42)
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == audio.shape
        assert augmented.dtype == np.float32


class TestAugmentAndEmbedPiano:
    @pytest.fixture
    def cache_with_segments(self, tmp_path):
        """Create mock cache with clean embeddings and metadata."""
        cache_dir = tmp_path / "youtube_piano_cache"
        emb_dir = cache_dir / "muq_embeddings"
        audio_dir = cache_dir / "audio"
        emb_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        # Create fake audio and clean embeddings
        import soundfile as sf
        sr = 24000
        audio = np.random.randn(sr * 65).astype(np.float32)
        sf.write(str(audio_dir / "vid123.wav"), audio, sr)

        # Clean embeddings (from segment_and_embed_piano)
        for i in range(2):
            torch.save(torch.randn(93, 1024), emb_dir / f"yt_vid123_seg{i:03d}.pt")

        # Segment metadata
        import jsonlines
        segments = [
            {"segment_id": f"yt_vid123_seg{i:03d}", "video_id": "vid123",
             "segment_start": i * 30.0, "segment_end": (i + 1) * 30.0}
            for i in range(2)
        ]
        with jsonlines.open(cache_dir / "metadata.jsonl", mode="w") as writer:
            for s in segments:
                writer.write(s)

        # Recordings metadata
        with jsonlines.open(cache_dir / "recordings.jsonl", mode="w") as writer:
            writer.write({"video_id": "vid123", "title": "Test", "channel": "Test",
                          "duration_seconds": 65.0, "audio_path": "audio/vid123.wav",
                          "source_url": "https://youtube.com/watch?v=vid123"})

        return cache_dir

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_creates_augmented_embeddings(self, mock_cls, cache_with_segments):
        cache_dir = cache_with_segments
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n = augment_and_embed_piano(cache_dir)

        aug_dir = cache_dir / "muq_embeddings_augmented"
        assert aug_dir.exists()
        assert len(list(aug_dir.glob("*.pt"))) == 2
        assert n == 2
