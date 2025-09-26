#!/usr/bin/env python3
"""
CCMusic Piano Dataset Integration for Fine-tuning
Piano sound quality evaluation with perceptual labels
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any, Union
import librosa
import pandas as pd
import logging
from functools import partial
import pickle
import numpy.typing as npt
from datasets import load_dataset
from PIL import Image

from src.data.audio_io import mel_db_128x128, mel_db_time_major

# Optional JAX imports (will gracefully handle if not available)
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp  # Fallback to numpy
    HAS_JAX = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CCMusicPianoDataset:
    """
    CCMusic Piano dataset loader for fine-tuning
    Based on "A Holistic Evaluation of Piano Sound Quality" paper
    Handles loading, preprocessing, and batching of piano audio with quality labels
    """
    
    def __init__(
        self,
        cache_dir: str = "./__pycache__",
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_length: int = 128,
        input_size: int = 300,  # For mel spectrogram processing
        use_augmentation: bool = True,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # train, val, test
        random_seed: int = 42
    ):
        """
        Initialize CCMusic Piano dataset
        Args:
            cache_dir: Directory to cache the dataset
            target_sr: Target sample rate for audio processing
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            segment_length: Length of audio segments in time frames
            input_size: Size for mel spectrogram input
            use_augmentation: Whether to apply data augmentation
            split_ratios: (train, val, test) split ratios
            random_seed: Random seed for reproducibility
        """
        self.cache_dir = cache_dir
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.input_size = input_size
        self.use_augmentation = use_augmentation
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Load dataset
        print("🎹 Loading CCMusic Piano dataset...")
        self.dataset = self._load_dataset()
        
        # Get class names (piano brands) and other info
        self.classes = self._get_classes()
        self.pitch_classes = self._get_pitch_classes()
        
        # Use existing dataset splits
        self.train_data = self.dataset['train'] 
        self.val_data = self.dataset['validation']
        self.test_data = self.dataset['test']
        
        print(f"✅ CCMusic Piano Dataset initialized:")
        print(f"   Train samples: {len(self.train_data)}")
        print(f"   Validation samples: {len(self.val_data)}")
        print(f"   Test samples: {len(self.test_data)}")
        print(f"   Piano brands: {len(self.classes)} ({', '.join(self.classes)})")
        print(f"   Pitch classes: {len(self.pitch_classes)}")
        print(f"   Sample rate: {target_sr}Hz")
        print(f"   Mel bands: {n_mels}, Segment length: {segment_length}")
    
    def _load_dataset(self):
        """Load the CCMusic Piano dataset from Hugging Face"""
        try:
            dataset = load_dataset(
                "ccmusic-database/pianos",
                cache_dir=self.cache_dir
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load ccmusic-database/pianos dataset: {e}")
    
    def _get_classes(self) -> List[str]:
        """Extract piano brand classes from dataset"""
        try:
            features = self.dataset['train'].features
            if 'label' in features and hasattr(features['label'], 'names'):
                return features['label'].names
            else:
                # Default piano brands if extraction fails
                return ['PearlRiver', 'YoungChang', 'Steinway-T', 'Hsinghai', 'Kawai', 'Steinway', 'Kawai-G']
        except Exception as e:
            logger.warning(f"Could not extract classes: {e}, using defaults")
            return ['PearlRiver', 'YoungChang', 'Steinway-T', 'Hsinghai', 'Kawai', 'Steinway', 'Kawai-G']
    
    def _get_pitch_classes(self) -> List[str]:
        """Extract pitch classes from dataset"""
        try:
            features = self.dataset['train'].features
            if 'pitch' in features and hasattr(features['pitch'], 'names'):
                return features['pitch'].names
            else:
                return []
        except Exception as e:
            logger.warning(f"Could not extract pitch classes: {e}")
            return []
    
    
    def _process_mel_spectrogram(self, mel_image, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Process mel spectrogram image to numpy array and enforce time-major dB [-80,0].

        Note: When audio is available in the sample, we prefer recomputing mel from audio.
        This function is the fallback when only mel images are available.
        """
        try:
            # Handle different input types
            if isinstance(mel_image, Image.Image):
                mel_array = np.array(mel_image)
            elif hasattr(mel_image, 'convert'):
                mel_array = np.array(mel_image.convert('RGB'))
            else:
                mel_array = np.array(mel_image)
            
            # Convert to grayscale if RGB
            if len(mel_array.shape) == 3 and mel_array.shape[2] == 3:
                mel_array = np.mean(mel_array, axis=2)
            
            # Resize if target size specified (to [n_mels, segment_length])
            if target_size:
                from scipy.ndimage import zoom
                current_shape = mel_array.shape
                zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1])
                mel_array = zoom(mel_array, zoom_factors, order=1)
            
            # Map to dB-like range and clip
            mel_array = mel_array.astype(np.float32)
            if mel_array.size == 0:
                raise ValueError("Empty mel image array")
            mmin = float(mel_array.min())
            mmax = float(mel_array.max())
            if mmax > mmin:
                mel_array = (mel_array - mmin) / (mmax - mmin)
            else:
                mel_array = np.zeros_like(mel_array, dtype=np.float32)
            mel_array = mel_array * 80.0 - 80.0  # [-80,0]
            mel_array = np.clip(mel_array, -80.0, 0.0)

            # Current layout after resize is [n_mels, segment_length]; convert to time-major [time, mel]
            if mel_array.shape[0] == self.n_mels and mel_array.shape[1] == self.segment_length:
                mel_array = mel_array.T  # [time, mel]
            elif mel_array.shape == (self.segment_length, self.n_mels):
                # already time-major
                pass
            else:
                # Fallback: transpose if likely [freq, time]
                if mel_array.shape[0] == self.n_mels:
                    mel_array = mel_array.T

            # Ensure final shape [segment_length, n_mels]
            if mel_array.shape != (self.segment_length, self.n_mels):
                from numpy import pad
                t, f = mel_array.shape
                # Time crop/pad
                if t > self.segment_length:
                    start = (t - self.segment_length) // 2
                    mel_array = mel_array[start:start + self.segment_length, :]
                elif t < self.segment_length:
                    pad_t = self.segment_length - t
                    mel_array = np.pad(mel_array, ((0, pad_t), (0, 0)), mode="constant", constant_values=-80.0)
                # Freq crop/pad
                t2, f2 = mel_array.shape
                if f2 > self.n_mels:
                    mel_array = mel_array[:, : self.n_mels]
                elif f2 < self.n_mels:
                    pad_f = self.n_mels - f2
                    mel_array = np.pad(mel_array, ((0, 0), (0, pad_f)), mode="constant", constant_values=-80.0)

            return mel_array.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error processing mel spectrogram: {e}")
            # Return zeros if processing fails (time-major)
            return np.full((self.segment_length, self.n_mels), -80.0, dtype=np.float32)
    
    def _extract_audio_features(self, audio_data, sr: int = None) -> np.ndarray:
        """Extract additional audio features for hybrid approach"""
        if sr is None:
            sr = self.target_sr
        
        try:
            # Ensure audio is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Basic audio features extraction
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)
            features.extend([np.mean(rms), np.std(rms)])
            
            # Pad or truncate to fixed size (e.g., 20 features)
            target_size = 20
            if len(features) > target_size:
                features = features[:target_size]
            elif len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Could not extract audio features: {e}")
            return np.zeros(20, dtype=np.float32)  # Return zero features
    
    def _augment_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Apply conservative data augmentation to mel spectrogram"""
        if not self.use_augmentation:
            return mel_spec
        
        augmented = mel_spec.copy()
        
        # Time masking (very conservative)
        if np.random.random() < 0.2:
            time_mask_width = np.random.randint(1, min(8, mel_spec.shape[1] // 10))
            time_mask_start = np.random.randint(0, max(1, mel_spec.shape[1] - time_mask_width))
            augmented[:, time_mask_start:time_mask_start + time_mask_width] = augmented.min()
        
        # Frequency masking (very conservative)  
        if np.random.random() < 0.2:
            freq_mask_height = np.random.randint(1, min(6, mel_spec.shape[0] // 10))
            freq_mask_start = np.random.randint(0, max(1, mel_spec.shape[0] - freq_mask_height))
            augmented[freq_mask_start:freq_mask_start + freq_mask_height, :] = augmented.min()
        
        # Gaussian noise (very subtle)
        if np.random.random() < 0.1:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = augmented + noise
        
        return augmented
    
    def get_split_data(self, split: str = 'train') -> List:
        """Get data for specific split"""
        if split == 'train':
            return self.train_data
        elif split == 'val' or split == 'validation':
            return self.val_data
        elif split == 'test':
            return self.test_data
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")
    
    def get_data_iterator(
        self,
        split: str = 'train',
        batch_size: int = 16,
        shuffle: bool = True,
        infinite: bool = False
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Create data iterator for training/validation/testing
        Returns:
            Iterator yielding (mel_spectrograms, audio_features, quality_labels) batches
        """
        
        data = self.get_split_data(split)
        use_augmentation = self.use_augmentation and (split == 'train')
        
        def data_generator():
            while True:
                # Shuffle indices if requested
                indices = np.arange(len(data))
                if shuffle:
                    np.random.shuffle(indices)
                
                batch_spectrograms = []
                batch_audio_features = []
                batch_labels = []
                
                for idx in indices:
                    try:
                        sample = data[idx]
                        
                        # Prefer recomputing mel from raw audio if available
                        mel_spec = None
                        if 'audio' in sample and sample['audio'] is not None:
                            try:
                                audio = sample['audio']
                                # HuggingFace Audio feature typically provides dict with 'array' and 'sampling_rate'
                                if isinstance(audio, dict) and 'array' in audio:
                                    y = np.asarray(audio['array'], dtype=np.float32)
                                    sr = int(audio.get('sampling_rate', self.target_sr))
                                else:
                                    # If audio is a numpy array, assume current target_sr
                                    y = np.asarray(audio, dtype=np.float32)
                                    sr = self.target_sr
                                if y.ndim > 1:
                                    y = np.mean(y, axis=1).astype(np.float32)
                                if sr != self.target_sr and y.size > 0:
                                    y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                                # Compute a deterministic 128x128 window
                                mel_spec = mel_db_128x128(
                                    y,
                                    sr=self.target_sr,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    n_mels=self.n_mels,
                                )  # [time=128, mel=128]
                            except Exception as e:
                                logger.warning(f"Mel recompute from audio failed for sample {idx}: {e}")
                                mel_spec = None

                        # Fallback to provided mel image
                        if mel_spec is None:
                            if 'mel' in sample and sample['mel'] is not None:
                                mel_spec = self._process_mel_spectrogram(
                                    sample['mel'],
                                    target_size=(self.n_mels, self.segment_length)
                                )  # time-major [128,128]
                            else:
                                logger.warning(f"No mel or audio found in sample {idx}")
                                continue

                        # Apply augmentation if enabled (conservative)
                        if use_augmentation:
                            mel_spec = self._augment_mel_spectrogram(mel_spec)

                        # Extract audio features if audio is available (optional hybrid)
                        audio_features = np.zeros(20, dtype=np.float32)
                        if 'audio' in sample and sample['audio'] is not None:
                            try:
                                audio_features = self._extract_audio_features(sample['audio'])
                            except Exception as e:
                                logger.warning(f"Could not extract audio features: {e}")
                        
                        # Get quality label
                        if 'label' in sample:
                            label = sample['label']
                            if isinstance(label, str):
                                # Map string label to index
                                try:
                                    label = self.classes.index(label)
                                except ValueError:
                                    label = 0  # Default to first class
                        else:
                            label = 0  # Default label
                        
                        batch_spectrograms.append(mel_spec)
                        batch_audio_features.append(audio_features)
                        batch_labels.append(label)
                        
                        # Yield batch when full
                        if len(batch_spectrograms) == batch_size:
                            yield (
                                jnp.array(batch_spectrograms),
                                jnp.array(batch_audio_features),
                                jnp.array(batch_labels)
                            )
                            
                            # Reset batch
                            batch_spectrograms = []
                            batch_audio_features = []
                            batch_labels = []
                    
                    except Exception as e:
                        logger.warning(f"Error processing sample {idx}: {e}")
                        continue
                
                # Yield remaining batch if not empty
                if batch_spectrograms:
                    # Pad to batch_size if needed
                    while len(batch_spectrograms) < batch_size:
                        batch_spectrograms.append(batch_spectrograms[-1])
                        batch_audio_features.append(batch_audio_features[-1])
                        batch_labels.append(batch_labels[-1])
                    
                    yield (
                        jnp.array(batch_spectrograms),
                        jnp.array(batch_audio_features),
                        jnp.array(batch_labels)
                    )
                
                if not infinite:
                    break
        
        return data_generator()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics"""
        stats = {
            "total_samples": len(self.train_data) + len(self.val_data) + len(self.test_data),
            "train_samples": len(self.train_data),
            "val_samples": len(self.val_data),
            "test_samples": len(self.test_data),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "spectrogram_shape": (self.n_mels, self.segment_length),
            "audio_features_dim": 20,
            "sample_rate": self.target_sr
        }
        
        # Try to get label distribution
        try:
            train_labels = []
            for sample in self.train_data[:100]:  # Sample first 100 for efficiency
                if 'label' in sample:
                    label = sample['label']
                    if isinstance(label, str):
                        try:
                            label = self.classes.index(label)
                        except ValueError:
                            label = 0
                    train_labels.append(label)
            
            if train_labels:
                unique, counts = np.unique(train_labels, return_counts=True)
                stats['label_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
        except Exception as e:
            logger.warning(f"Could not compute label distribution: {e}")
        
        return stats


def create_quality_mapping() -> Dict[str, int]:
    """Create mapping from quality descriptions to perceptual dimensions"""
    # This maps piano quality aspects to indices for multi-dimensional evaluation
    # Based on the paper's methodology
    quality_mapping = {
        'timing_stability': 0,
        'articulation_clarity': 1,
        'dynamic_range': 2,
        'tonal_balance': 3,
        'expression_control': 4
    }
    return quality_mapping


if __name__ == "__main__":
    # Test CCMusic Piano dataset
    print("=== CCMusic Piano Dataset Test ===")
    
    try:
        dataset = CCMusicPianoDataset(
            cache_dir="./__pycache__/test_ccmusic",
            use_augmentation=True
        )
        
        # Print statistics
        stats = dataset.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test data iterator
        print(f"\nTesting data iterator...")
        iterator = dataset.get_data_iterator(
            split='train',
            batch_size=4,
            shuffle=True,
            infinite=False
        )
        
        # Get one batch
        try:
            mel_specs, audio_features, labels = next(iterator)
            print(f"✅ Batch loaded successfully:")
            print(f"  Mel spectrograms: {mel_specs.shape}")
            print(f"  Audio features: {audio_features.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Label range: {labels.min()} - {labels.max()}")
        except StopIteration:
            print("⚠️ No data available in iterator")
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
        
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
