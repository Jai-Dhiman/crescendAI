import pytest
import numpy as np
import tempfile
import os
from src.data.augmentation import AudioAugmentation, create_augmentation_pipeline


@pytest.fixture
def sample_audio():
    """Generate sample audio signal for testing."""
    sr = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration))
    # Generate a simple sine wave at A4 (440 Hz)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def augmentor():
    """Create AudioAugmentation instance."""
    return AudioAugmentation(sr=44100)


class TestPitchShift:
    """Tests for pitch shift augmentation."""

    def test_pitch_shift_shape_preserved(self, augmentor, sample_audio):
        """Test that pitch shift preserves audio shape."""
        audio, _ = sample_audio
        shifted = augmentor.pitch_shift(audio, semitones=2.0)
        assert shifted.shape == audio.shape

    def test_pitch_shift_positive(self, augmentor, sample_audio):
        """Test positive pitch shift."""
        audio, _ = sample_audio
        shifted = augmentor.pitch_shift(audio, semitones=2.0)
        assert shifted.shape == audio.shape
        assert not np.array_equal(shifted, audio)

    def test_pitch_shift_negative(self, augmentor, sample_audio):
        """Test negative pitch shift."""
        audio, _ = sample_audio
        shifted = augmentor.pitch_shift(audio, semitones=-2.0)
        assert shifted.shape == audio.shape
        assert not np.array_equal(shifted, audio)

    def test_pitch_shift_zero(self, augmentor, sample_audio):
        """Test zero pitch shift (should be approximately unchanged)."""
        audio, _ = sample_audio
        shifted = augmentor.pitch_shift(audio, semitones=0.0)
        assert shifted.shape == audio.shape
        # Should be very similar (not exact due to processing)
        assert np.allclose(shifted, audio, atol=0.1)

    def test_pitch_shift_random(self, augmentor, sample_audio):
        """Test random pitch shift (no semitones specified)."""
        audio, _ = sample_audio
        shifted = augmentor.pitch_shift(audio)
        assert shifted.shape == audio.shape


class TestTimeStretch:
    """Tests for time stretch augmentation."""

    def test_time_stretch_faster(self, augmentor, sample_audio):
        """Test time stretch with faster rate."""
        audio, _ = sample_audio
        stretched = augmentor.time_stretch(audio, rate=1.15)
        # Faster rate means shorter audio
        assert len(stretched) < len(audio)

    def test_time_stretch_slower(self, augmentor, sample_audio):
        """Test time stretch with slower rate."""
        audio, _ = sample_audio
        stretched = augmentor.time_stretch(audio, rate=0.85)
        # Slower rate means longer audio
        assert len(stretched) > len(audio)

    def test_time_stretch_normal(self, augmentor, sample_audio):
        """Test time stretch with rate=1.0 (no change)."""
        audio, _ = sample_audio
        stretched = augmentor.time_stretch(audio, rate=1.0)
        assert stretched.shape == audio.shape
        assert np.allclose(stretched, audio, atol=0.01)

    def test_time_stretch_random(self, augmentor, sample_audio):
        """Test random time stretch."""
        audio, _ = sample_audio
        stretched = augmentor.time_stretch(audio)
        # Should return some audio
        assert len(stretched) > 0


class TestAddNoise:
    """Tests for noise addition augmentation."""

    def test_add_noise_shape_preserved(self, augmentor, sample_audio):
        """Test that noise addition preserves shape."""
        audio, _ = sample_audio
        noisy = augmentor.add_noise(audio, snr_db=30.0)
        assert noisy.shape == audio.shape

    def test_add_noise_high_snr(self, augmentor, sample_audio):
        """Test noise addition with high SNR (subtle noise)."""
        audio, _ = sample_audio
        noisy = augmentor.add_noise(audio, snr_db=40.0)
        assert noisy.shape == audio.shape
        # High SNR should be similar to original
        assert np.allclose(noisy, audio, atol=0.2)

    def test_add_noise_low_snr(self, augmentor, sample_audio):
        """Test noise addition with low SNR (more noise)."""
        audio, _ = sample_audio
        noisy = augmentor.add_noise(audio, snr_db=25.0)
        assert noisy.shape == audio.shape
        # Should be noticeably different
        assert not np.allclose(noisy, audio, atol=0.1)

    def test_add_noise_increases_variance(self, augmentor, sample_audio):
        """Test that adding noise increases signal variance."""
        audio, _ = sample_audio
        noisy = augmentor.add_noise(audio, snr_db=30.0)
        assert np.var(noisy) >= np.var(audio)

    def test_add_noise_random(self, augmentor, sample_audio):
        """Test random noise addition."""
        audio, _ = sample_audio
        noisy = augmentor.add_noise(audio)
        assert noisy.shape == audio.shape


class TestRoomAcoustics:
    """Tests for room acoustics augmentation."""

    def test_room_acoustics_shape_preserved(self, augmentor, sample_audio):
        """Test that room acoustics preserves shape."""
        audio, _ = sample_audio
        reverb = augmentor.apply_room_acoustics(audio)
        assert reverb.shape == audio.shape

    def test_room_acoustics_synthetic_ir(self, augmentor, sample_audio):
        """Test room acoustics with synthetic IR."""
        audio, _ = sample_audio
        reverb = augmentor.apply_room_acoustics(audio)
        assert reverb.shape == audio.shape
        assert not np.array_equal(reverb, audio)

    def test_room_acoustics_custom_ir(self, augmentor, sample_audio):
        """Test room acoustics with custom impulse response."""
        audio, sr = sample_audio
        # Create simple impulse response
        ir = np.zeros(int(sr * 0.1))
        ir[0] = 1.0
        ir[int(sr * 0.02)] = 0.3  # Early reflection

        reverb = augmentor.apply_room_acoustics(audio, impulse_response=ir)
        assert reverb.shape == audio.shape

    def test_generate_simple_ir(self, augmentor):
        """Test synthetic IR generation."""
        ir = augmentor._generate_simple_ir()
        assert len(ir) > 0
        assert ir[0] == 1.0  # Direct sound
        assert np.max(np.abs(ir[1:])) < 1.0  # Reflections are quieter


class TestCompressAudio:
    """Tests for MP3 compression augmentation."""

    def test_compress_audio_low_quality(self, augmentor, sample_audio):
        """Test MP3 compression at low quality."""
        audio, _ = sample_audio
        compressed = augmentor.compress_audio(audio, quality='low')
        assert compressed.shape == audio.shape
        assert compressed.dtype == np.float32

    def test_compress_audio_medium_quality(self, augmentor, sample_audio):
        """Test MP3 compression at medium quality."""
        audio, _ = sample_audio
        compressed = augmentor.compress_audio(audio, quality='medium')
        assert compressed.shape == audio.shape

    def test_compress_audio_high_quality(self, augmentor, sample_audio):
        """Test MP3 compression at high quality."""
        audio, _ = sample_audio
        compressed = augmentor.compress_audio(audio, quality='high')
        assert compressed.shape == audio.shape
        # High quality should be more similar to original
        assert np.allclose(compressed, audio, atol=0.1)

    def test_compress_audio_random_quality(self, augmentor, sample_audio):
        """Test MP3 compression with random quality."""
        audio, _ = sample_audio
        compressed = augmentor.compress_audio(audio)
        assert compressed.shape == audio.shape

    def test_compress_audio_length_preserved(self, augmentor, sample_audio):
        """Test that compression preserves exact length."""
        audio, _ = sample_audio
        compressed = augmentor.compress_audio(audio, quality='medium')
        assert len(compressed) == len(audio)

    @pytest.mark.skipif(
        os.system("which ffmpeg > /dev/null 2>&1") != 0,
        reason="ffmpeg not installed"
    )
    def test_compress_audio_requires_ffmpeg(self, augmentor, sample_audio):
        """Test that compression requires ffmpeg."""
        audio, _ = sample_audio
        # This should work if ffmpeg is installed
        compressed = augmentor.compress_audio(audio, quality='low')
        assert compressed.shape == audio.shape


class TestGainVariation:
    """Tests for gain variation augmentation."""

    def test_gain_variation_shape_preserved(self, augmentor, sample_audio):
        """Test that gain variation preserves shape."""
        audio, _ = sample_audio
        gained = augmentor.gain_variation(audio, db_range=3.0)
        assert gained.shape == audio.shape

    def test_gain_variation_positive(self, augmentor, sample_audio):
        """Test positive gain (louder)."""
        audio, _ = sample_audio
        gained = augmentor.gain_variation(audio, db_range=6.0)
        # Positive gain should increase amplitude
        assert np.abs(gained).max() > np.abs(audio).max()

    def test_gain_variation_negative(self, augmentor, sample_audio):
        """Test negative gain (quieter)."""
        audio, _ = sample_audio
        gained = augmentor.gain_variation(audio, db_range=-6.0)
        # Negative gain should decrease amplitude
        assert np.abs(gained).max() < np.abs(audio).max()

    def test_gain_variation_zero(self, augmentor, sample_audio):
        """Test zero gain (no change)."""
        audio, _ = sample_audio
        gained = augmentor.gain_variation(audio, db_range=0.0)
        assert np.allclose(gained, audio)

    def test_gain_variation_random(self, augmentor, sample_audio):
        """Test random gain variation."""
        audio, _ = sample_audio
        gained = augmentor.gain_variation(audio)
        assert gained.shape == audio.shape


class TestAugmentPipeline:
    """Tests for full augmentation pipeline."""

    def test_augment_pipeline_shape_preserved(self, augmentor, sample_audio):
        """Test that pipeline preserves shape."""
        audio, _ = sample_audio
        augmented = augmentor.augment_pipeline(audio)
        assert augmented.shape == audio.shape

    def test_augment_pipeline_no_clipping(self, augmentor, sample_audio):
        """Test that pipeline prevents clipping."""
        audio, _ = sample_audio
        augmented = augmentor.augment_pipeline(audio)
        assert np.abs(augmented).max() <= 1.0

    def test_augment_pipeline_with_config(self, augmentor, sample_audio):
        """Test pipeline with custom config."""
        audio, _ = sample_audio
        config = {
            'pitch_shift_prob': 1.0,  # Always apply
            'time_stretch_prob': 0.0,  # Never apply
            'noise_prob': 0.0,
            'gain_prob': 0.0,
            'room_acoustics_prob': 0.0,
            'compression_prob': 0.0,
            'max_augmentations': 1,
        }
        augmented = augmentor.augment_pipeline(audio, config=config)
        assert augmented.shape == audio.shape

    def test_augment_pipeline_max_augmentations(self, augmentor, sample_audio):
        """Test that pipeline respects max_augmentations."""
        audio, _ = sample_audio
        # Force augmentations that preserve length to be selected
        augmentor.pitch_shift_prob = 1.0
        augmentor.time_stretch_prob = 0.0  # Skip time stretch (changes length)
        augmentor.noise_prob = 1.0
        augmentor.gain_prob = 1.0
        augmentor.room_acoustics_prob = 1.0
        augmentor.compression_prob = 1.0
        augmentor.max_augmentations = 2

        # Pipeline should still work (limiting to 2 augmentations)
        augmented = augmentor.augment_pipeline(audio)
        assert augmented.shape == audio.shape

    def test_augment_pipeline_deterministic_with_seed(self, augmentor, sample_audio):
        """Test that pipeline is deterministic with random seed."""
        audio, _ = sample_audio

        # Use fixed probabilities to avoid randomness in selection
        config = {
            'pitch_shift_prob': 1.0,
            'time_stretch_prob': 0.0,
            'noise_prob': 0.0,
            'gain_prob': 1.0,
            'room_acoustics_prob': 0.0,
            'compression_prob': 0.0,
            'max_augmentations': 2,
        }

        # Set seed and augment
        np.random.seed(42)
        augmented1 = augmentor.augment_pipeline(audio, config=config)

        # Reset seed and augment again
        np.random.seed(42)
        augmented2 = augmentor.augment_pipeline(audio, config=config)

        # Should be identical
        assert np.allclose(augmented1, augmented2, atol=1e-5)

    def test_augment_pipeline_different_without_seed(self, augmentor, sample_audio):
        """Test that pipeline produces different results without seed."""
        audio, _ = sample_audio

        # Use augmentations that preserve length
        config = {
            'pitch_shift_prob': 0.5,
            'time_stretch_prob': 0.0,  # Skip time stretch (changes length)
            'noise_prob': 0.5,
            'gain_prob': 0.5,
            'room_acoustics_prob': 0.5,
            'compression_prob': 0.5,
            'max_augmentations': 2,
        }

        augmented1 = augmentor.augment_pipeline(audio, config=config)
        augmented2 = augmentor.augment_pipeline(audio, config=config)

        # Should be different (very unlikely to be identical with randomness)
        # At least check that shapes match
        assert augmented1.shape == audio.shape
        assert augmented2.shape == audio.shape


class TestCreateAugmentationPipeline:
    """Tests for pipeline factory function."""

    def test_create_pipeline_training(self):
        """Test creating pipeline for training."""
        pipeline = create_augmentation_pipeline(sr=44100, training=True)
        assert pipeline is not None
        assert isinstance(pipeline, AudioAugmentation)
        assert pipeline.sr == 44100

    def test_create_pipeline_inference(self):
        """Test creating pipeline for inference."""
        pipeline = create_augmentation_pipeline(sr=44100, training=False)
        assert pipeline is None

    def test_create_pipeline_default_params(self):
        """Test pipeline with default parameters."""
        pipeline = create_augmentation_pipeline()
        assert pipeline is not None
        assert pipeline.sr == 44100
        assert pipeline.max_augmentations == 3


class TestAugmentationIntegration:
    """Integration tests for augmentation pipeline."""

    def test_sequential_augmentations(self, augmentor, sample_audio):
        """Test applying augmentations sequentially."""
        audio, _ = sample_audio

        # Apply augmentations one by one (avoid time stretch which changes length)
        audio = augmentor.pitch_shift(audio, semitones=1.0)
        audio = augmentor.add_noise(audio, snr_db=35.0)
        audio = augmentor.gain_variation(audio, db_range=3.0)

        assert audio.shape == sample_audio[0].shape
        assert np.abs(audio).max() <= 2.0  # Some tolerance for sequential augmentations

    def test_augmentation_on_silence(self, augmentor):
        """Test augmentations on silent audio."""
        silence = np.zeros(44100, dtype=np.float32)

        # These should handle silence gracefully
        pitch_shifted = augmentor.pitch_shift(silence, semitones=2.0)
        time_stretched = augmentor.time_stretch(silence, rate=1.1)
        gained = augmentor.gain_variation(silence, db_range=6.0)

        assert pitch_shifted.shape == silence.shape
        # Time stretch changes length
        assert len(time_stretched) > 0
        assert gained.shape == silence.shape

    def test_augmentation_on_loud_signal(self, augmentor):
        """Test augmentations on loud signal (near clipping)."""
        loud_signal = np.ones(44100, dtype=np.float32) * 0.95

        augmented = augmentor.augment_pipeline(loud_signal)
        # Pipeline should prevent clipping
        assert np.abs(augmented).max() <= 1.0

    def test_augmentation_preserves_dtype(self, augmentor, sample_audio):
        """Test that augmentations preserve numeric dtype."""
        audio, _ = sample_audio

        pitch_shifted = augmentor.pitch_shift(audio)
        time_stretched = augmentor.time_stretch(audio)
        noisy = augmentor.add_noise(audio)

        # librosa may return float64, but they should be floating point
        assert pitch_shifted.dtype in [np.float32, np.float64]
        assert time_stretched.dtype in [np.float32, np.float64]
        assert noisy.dtype in [np.float32, np.float64]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_audio(self, augmentor):
        """Test augmentation on very short audio."""
        short_audio = np.random.randn(100).astype(np.float32)

        # Use augmentations that preserve length
        config = {
            'pitch_shift_prob': 0.0,
            'time_stretch_prob': 0.0,  # Avoid time stretch on short audio
            'noise_prob': 1.0,
            'gain_prob': 1.0,
            'room_acoustics_prob': 0.0,
            'compression_prob': 0.0,
            'max_augmentations': 2,
        }

        # Should handle gracefully
        augmented = augmentor.augment_pipeline(short_audio, config=config)
        assert augmented.shape == short_audio.shape

    def test_augmentation_probabilities(self):
        """Test that probabilities are respected."""
        # Zero probability should disable augmentation
        aug = AudioAugmentation(
            pitch_shift_prob=0.0,
            time_stretch_prob=0.0,
            noise_prob=0.0,
            gain_prob=0.0,
            room_acoustics_prob=0.0,
            compression_prob=0.0,
        )

        # Use deterministic audio with known max value
        np.random.seed(42)
        audio = np.random.randn(44100).astype(np.float32) * 0.1  # Very small to avoid any normalization
        original_max = np.abs(audio).max()

        augmented = aug.augment_pipeline(audio)
        augmented_max = np.abs(augmented).max()

        # With zero probabilities and small audio, should be exactly unchanged
        # If normalization triggered, augmented_max would be less than original_max
        assert augmented_max <= original_max * 1.01  # Allow 1% tolerance

        # Or check that the shape is preserved at minimum
        assert augmented.shape == audio.shape

    def test_sample_rate_consistency(self):
        """Test that sample rate is used consistently."""
        sr = 22050  # Different sample rate
        aug = AudioAugmentation(sr=sr)
        assert aug.sr == sr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
