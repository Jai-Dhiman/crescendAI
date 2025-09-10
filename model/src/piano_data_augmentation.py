#!/usr/bin/env python3
"""
Piano-Specific Data Augmentation for Music Performance Analysis
Conservative augmentations that preserve musical meaning while increasing dataset size
"""

import numpy as np
import librosa
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, List
import random
from pathlib import Path


class PianoAudioAugmenter:
    """
    Piano-specific audio augmentation that preserves musical characteristics
    Focuses on realistic performance variations rather than aggressive transformations
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        print(f"🎹 Piano Audio Augmenter initialized (SR: {sr}Hz)")
        
    def time_stretch(self, y: np.ndarray, stretch_factor: float = 1.0) -> np.ndarray:
        """
        Time stretch audio while preserving pitch
        Conservative range: 0.9-1.1x (realistic tempo variations)
        """
        if stretch_factor == 1.0:
            return y
            
        # Use librosa's phase vocoder for high-quality time stretching
        return librosa.effects.time_stretch(y, rate=stretch_factor)
    
    def pitch_shift(self, y: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """
        Pitch shift audio while preserving timing
        Conservative range: ±1 semitone (realistic tuning variations)
        """
        if n_steps == 0.0:
            return y
            
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def add_realistic_noise(self, y: np.ndarray, noise_level: float = 0.001) -> np.ndarray:
        """
        Add very subtle noise to simulate slight recording variations
        Much more conservative than typical audio augmentation
        """
        if noise_level <= 0:
            return y
            
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise
    
    def dynamic_range_compression(self, y: np.ndarray, threshold: float = 0.8, ratio: float = 2.0) -> np.ndarray:
        """
        Subtle dynamic range compression to simulate different recording setups
        """
        # Simple soft compression
        compressed = np.copy(y)
        above_threshold = np.abs(compressed) > threshold
        
        sign = np.sign(compressed)
        abs_signal = np.abs(compressed)
        
        # Apply compression only above threshold
        compressed_abs = np.where(
            above_threshold,
            threshold + (abs_signal - threshold) / ratio,
            abs_signal
        )
        
        return sign * compressed_abs
    
    def subtle_eq_filter(self, y: np.ndarray, freq_center: float = 1000, gain_db: float = 0.0, q: float = 1.0) -> np.ndarray:
        """
        Apply subtle EQ changes to simulate different piano/room characteristics
        """
        if gain_db == 0.0:
            return y
            
        # Create simple shelving filter
        # This is a simplified implementation - for production, use more sophisticated filtering
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply frequency-dependent scaling (simplified)
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1/self.sr)
        
        # Simple bell curve around center frequency
        freq_response = 1 + (gain_linear - 1) * np.exp(-(freqs - freq_center)**2 / (2 * (freq_center/q)**2))
        
        # Apply filter
        filtered_fft = fft * freq_response
        filtered_y = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_y
    
    def augment_audio_conservative(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply conservative augmentation suitable for piano performance analysis
        Returns augmented audio and applied parameters for tracking
        """
        augmented = np.copy(y)
        params = {}
        
        # Random conservative transformations
        # Time stretch: 0.95-1.05 (5% variation)
        stretch_factor = random.uniform(0.95, 1.05)
        augmented = self.time_stretch(augmented, stretch_factor)
        params['time_stretch'] = stretch_factor
        
        # Pitch shift: ±0.5 semitones (very subtle)
        pitch_steps = random.uniform(-0.5, 0.5)
        augmented = self.pitch_shift(augmented, pitch_steps)
        params['pitch_shift'] = pitch_steps
        
        # Very subtle noise (recording variations)
        noise_level = random.uniform(0.0005, 0.002)
        augmented = self.add_realistic_noise(augmented, noise_level)
        params['noise_level'] = noise_level
        
        # Occasional subtle dynamic compression
        if random.random() < 0.3:  # 30% chance
            compression_ratio = random.uniform(1.5, 3.0)
            augmented = self.dynamic_range_compression(augmented, threshold=0.7, ratio=compression_ratio)
            params['compression_ratio'] = compression_ratio
        
        # Occasional subtle EQ
        if random.random() < 0.2:  # 20% chance
            eq_freq = random.uniform(500, 2000)  # Focus on important piano frequencies
            eq_gain = random.uniform(-1.0, 1.0)  # Very subtle
            augmented = self.subtle_eq_filter(augmented, eq_freq, eq_gain)
            params['eq_freq'] = eq_freq
            params['eq_gain'] = eq_gain
        
        # Normalize to prevent clipping
        if np.max(np.abs(augmented)) > 0:
            augmented = augmented / np.max(np.abs(augmented))
        
        return augmented, params


class PianoSpecAugment:
    """
    SpecAugment specifically tuned for piano spectrograms
    More conservative than typical speech/general audio augmentation
    """
    
    def __init__(self,
                 freq_mask_param: int = 8,      # Smaller than typical (27)
                 time_mask_param: int = 15,     # Smaller than typical (100)
                 num_freq_masks: int = 1,       # Conservative
                 num_time_masks: int = 1,       # Conservative
                 mask_value: float = 0.0):
        
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param  
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        
        print(f"🎼 Piano SpecAugment initialized:")
        print(f"   Freq mask: {freq_mask_param}, Time mask: {time_mask_param}")
        print(f"   Num masks: {num_freq_masks} freq, {num_time_masks} time")
    
    def freq_mask(self, spectrogram: np.ndarray, rng_key: Optional[jax.Array] = None) -> np.ndarray:
        """Apply frequency masking to spectrogram"""
        spec = np.copy(spectrogram)
        time_steps, freq_bins = spec.shape
        
        for _ in range(self.num_freq_masks):
            # Random mask width
            f = random.randint(0, self.freq_mask_param)
            if f == 0:
                continue
                
            # Random start position  
            f0 = random.randint(0, max(1, freq_bins - f))
            
            # Apply mask
            spec[:, f0:f0+f] = self.mask_value
            
        return spec
    
    def time_mask(self, spectrogram: np.ndarray, rng_key: Optional[jax.Array] = None) -> np.ndarray:
        """Apply time masking to spectrogram"""
        spec = np.copy(spectrogram)
        time_steps, freq_bins = spec.shape
        
        for _ in range(self.num_time_masks):
            # Random mask width
            t = random.randint(0, self.time_mask_param)
            if t == 0:
                continue
                
            # Random start position
            t0 = random.randint(0, max(1, time_steps - t))
            
            # Apply mask
            spec[t0:t0+t, :] = self.mask_value
            
        return spec
    
    def augment_spectrogram(self, spectrogram: np.ndarray, rng_key: Optional[jax.Array] = None) -> np.ndarray:
        """Apply full SpecAugment pipeline"""
        # Apply frequency masking
        spec = self.freq_mask(spectrogram, rng_key)
        
        # Apply time masking
        spec = self.time_mask(spec, rng_key)
        
        return spec


class PianoDataAugmentationPipeline:
    """
    Complete augmentation pipeline combining audio and spectrogram augmentations
    Designed for piano performance analysis with quality over quantity approach
    """
    
    def __init__(self, sr: int = 22050, target_length: int = 128):
        self.sr = sr
        self.target_length = target_length
        
        self.audio_augmenter = PianoAudioAugmenter(sr=sr)
        self.spec_augmenter = PianoSpecAugment(
            freq_mask_param=6,    # Conservative for piano
            time_mask_param=12,   # Conservative for piano
            num_freq_masks=1,
            num_time_masks=1
        )
        
        print(f"🎹 Piano Augmentation Pipeline initialized")
        print(f"   Target spectrogram length: {target_length}")
    
    def extract_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram from audio"""
        # Ensure normalization
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=2048, hop_length=512, n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target length
        if mel_spec_db.shape[1] > self.target_length:
            # Truncate from center
            start = (mel_spec_db.shape[1] - self.target_length) // 2
            mel_spec_db = mel_spec_db[:, start:start + self.target_length]
        elif mel_spec_db.shape[1] < self.target_length:
            # Pad with edge values
            pad_width = self.target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
        
        # Transpose to (time, freq) format
        return mel_spec_db.T
    
    def create_augmented_samples(self, y: np.ndarray, n_augmentations: int = 2) -> List[Tuple[np.ndarray, Dict]]:
        """
        Create augmented samples from original audio
        Returns list of (spectrogram, augmentation_params) tuples
        """
        augmented_samples = []
        
        # Original sample (no augmentation)
        original_spec = self.extract_spectrogram(y)
        augmented_samples.append((original_spec, {'augmentation_type': 'original'}))
        
        # Create augmented versions
        for i in range(n_augmentations):
            # Audio-level augmentation
            aug_audio, audio_params = self.audio_augmenter.augment_audio_conservative(y)
            
            # Extract spectrogram
            aug_spec = self.extract_spectrogram(aug_audio)
            
            # Spectrogram-level augmentation (50% chance)
            if random.random() < 0.5:
                aug_spec = self.spec_augmenter.augment_spectrogram(aug_spec)
                audio_params['spec_augment'] = True
            else:
                audio_params['spec_augment'] = False
            
            audio_params['augmentation_type'] = f'augmented_{i+1}'
            augmented_samples.append((aug_spec, audio_params))
        
        return augmented_samples
    
    def augment_dataset_batch(self, 
                             audio_files: List[Path], 
                             n_augmentations: int = 2) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Augment a batch of audio files
        Returns augmented spectrograms and their metadata
        """
        all_spectrograms = []
        all_metadata = []
        
        print(f"🎵 Augmenting {len(audio_files)} audio files with {n_augmentations} augmentations each...")
        
        for audio_file in audio_files:
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=self.sr, mono=True)
                
                # Create augmented samples
                augmented_samples = self.create_augmented_samples(y, n_augmentations)
                
                for spec, params in augmented_samples:
                    all_spectrograms.append(spec)
                    params['original_file'] = str(audio_file)
                    all_metadata.append(params)
                    
            except Exception as e:
                print(f"Warning: Failed to process {audio_file}: {e}")
                continue
        
        print(f"   Generated {len(all_spectrograms)} total samples from {len(audio_files)} originals")
        return all_spectrograms, all_metadata


def test_piano_augmentation():
    """Test piano-specific augmentation pipeline"""
    print("=== Piano Data Augmentation Test ===\n")
    
    # Create synthetic piano-like test signal
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Piano-like signal with multiple harmonics and decay
    fundamental = 261.63  # C4
    y = (0.8 * np.sin(2 * np.pi * fundamental * t) +
         0.4 * np.sin(2 * np.pi * fundamental * 2 * t) +
         0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +
         0.1 * np.sin(2 * np.pi * fundamental * 4 * t))
    
    # Add realistic envelope and subtle noise
    envelope = np.exp(-t * 1.5)  # Piano-like decay
    y = y * envelope + 0.005 * np.random.randn(len(t))
    
    # Initialize augmentation pipeline
    pipeline = PianoDataAugmentationPipeline(sr=sr)
    
    # Test audio augmentation
    print("Testing audio augmentation...")
    audio_augmenter = PianoAudioAugmenter(sr=sr)
    
    for i in range(3):
        aug_audio, params = audio_augmenter.augment_audio_conservative(y)
        print(f"  Augmentation {i+1}: {params}")
    
    # Test spectrogram augmentation
    print("\nTesting spectrogram augmentation...")
    original_spec = pipeline.extract_spectrogram(y)
    print(f"  Original spectrogram shape: {original_spec.shape}")
    
    spec_augmenter = PianoSpecAugment()
    aug_spec = spec_augmenter.augment_spectrogram(original_spec)
    print(f"  Augmented spectrogram shape: {aug_spec.shape}")
    
    # Test full pipeline
    print("\nTesting full augmentation pipeline...")
    augmented_samples = pipeline.create_augmented_samples(y, n_augmentations=3)
    
    print(f"  Generated {len(augmented_samples)} samples:")
    for i, (spec, metadata) in enumerate(augmented_samples):
        print(f"    Sample {i}: {spec.shape}, type: {metadata['augmentation_type']}")
        if 'time_stretch' in metadata:
            print(f"      Time stretch: {metadata['time_stretch']:.3f}")
        if 'pitch_shift' in metadata:
            print(f"      Pitch shift: {metadata['pitch_shift']:.3f} semitones")
    
    print("\n✅ Piano augmentation test complete!")
    
    # Calculate effective dataset multiplier
    multiplier = len(augmented_samples) / 1  # 1 original file
    print(f"\n🎯 Dataset expansion: {multiplier:.1f}x increase")
    print("   Conservative augmentations preserve musical meaning")
    print("   Quality over quantity approach for small dataset")
    
    return augmented_samples


if __name__ == "__main__":
    test_piano_augmentation()