#!/usr/bin/env python3
"""
PercePiano Dataset Integration for AST Fine-tuning
Piano performance analysis with perceptual dimension labels
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import librosa
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 19 perceptual dimensions from PercePiano paper
PERCEPTUAL_DIMENSIONS = [
    "Timing_Stable_Unstable",
    "Articulation_Short_Long", 
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark", 
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing"
]


class PercePianoDataset:
    """
    PercePiano dataset loader for fine-tuning
    Handles loading, preprocessing, and batching of piano audio with perceptual labels
    """
    
    def __init__(
        self,
        percepianos_root: str,
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_length: int = 128,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize PercePiano dataset
        Args:
            percepianos_root: Path to PercePiano dataset root
            target_sr: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            segment_length: Length of audio segments in time frames
            cache_dir: Directory to cache preprocessed spectrograms
        """
        self.percepianos_root = Path(percepianos_root)
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata and labels
        self.metadata, self.labels = self._load_data()
        
        print(f"🎹 PercePiano Dataset initialized:")
        print(f"   Total recordings: {len(self.metadata)}")
        print(f"   Sample rate: {target_sr}Hz")
        print(f"   Mel bands: {n_mels}, Segment length: {segment_length}")
        if cache_dir:
            print(f"   Cache directory: {cache_dir}")
    
    def _load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load PercePiano metadata and perceptual labels"""
        # Try to find label files
        possible_label_paths = [
            self.percepianos_root / "label_2round_mean_reg_19_with0_rm_highstd0.json",
            Path("archive/results/label_2round_std_reg_19_with0_rm_highstd0.json"),
            Path("data/label_2round_mean_reg_19_with0_rm_highstd0.json")
        ]
        
        labels_path = None
        for path in possible_label_paths:
            if path.exists():
                labels_path = path
                break
        
        if not labels_path:
            raise FileNotFoundError(f"PercePiano labels not found. Searched: {possible_label_paths}")
        
        # Load perceptual labels
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        print(f"Loaded labels from: {labels_path}")
        
        # Find audio files - look for common extensions
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.percepianos_root.glob(f"**/*{ext}"))
        
        if not audio_files:
            # Try alternative locations
            alt_paths = [
                Path("PercePiano"),
                Path("data/PercePiano"), 
                Path("archive/PercePiano")
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    for ext in audio_extensions:
                        audio_files.extend(alt_path.glob(f"**/*{ext}"))
                    if audio_files:
                        break
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.percepianos_root}")
        
        # Create metadata DataFrame
        metadata_records = []
        for audio_file in audio_files:
            # Extract performance ID from filename (assuming convention)
            stem = audio_file.stem
            # Try different naming conventions
            perf_id = stem
            
            # Check if we have labels for this performance
            if perf_id in labels:
                metadata_records.append({
                    'performance_id': perf_id,
                    'audio_path': audio_file,
                    'composer': self._extract_composer(stem),
                    'piece': self._extract_piece(stem)
                })
        
        metadata_df = pd.DataFrame(metadata_records)
        
        print(f"Found {len(metadata_df)} audio files with labels out of {len(audio_files)} total files")
        print(f"Sample performance IDs: {list(labels.keys())[:5]}")
        
        return metadata_df, labels
    
    def _extract_composer(self, filename: str) -> str:
        """Extract composer from filename"""
        if 'bach' in filename.lower():
            return 'Bach'
        elif 'mozart' in filename.lower():
            return 'Mozart'
        elif 'beethoven' in filename.lower():
            return 'Beethoven'
        elif 'chopin' in filename.lower():
            return 'Chopin'
        else:
            return 'Unknown'
    
    def _extract_piece(self, filename: str) -> str:
        """Extract piece name from filename"""
        return filename.replace('_', ' ').title()
    
    def _extract_mel_spectrogram(self, audio_path: Path) -> Optional[npt.NDArray[np.float64]]:
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (time, freq) format
            return log_mel.T  # Shape: (time_frames, n_mels)
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def _get_cache_path(self, audio_path: Path) -> Path:
        """Get cache path for preprocessed spectrogram"""
        if self.cache_dir is None:
            return None
        
        cache_name = f"{audio_path.stem}_{self.target_sr}_{self.n_mels}_{self.segment_length}.pkl"
        return self.cache_dir / cache_name
    
    def _load_or_create_spectrogram(self, audio_path: Path) -> np.ndarray:
        """Load cached spectrogram or create new one"""
        cache_path = self._get_cache_path(audio_path)
        
        # Try loading from cache
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed for {cache_path}: {e}")
        
        # Extract spectrogram
        spectrogram = self._extract_mel_spectrogram(audio_path)
        
        if spectrogram is None:
            return None
        
        # Save to cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(spectrogram, f)
            except Exception as e:
                logger.warning(f"Cache save failed for {cache_path}: {e}")
        
        return spectrogram
    
    def _create_segments(self, spectrogram: np.ndarray) -> List[np.ndarray]:
        """Split spectrogram into fixed-length segments"""
        time_frames, n_mels = spectrogram.shape
        
        if time_frames < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - time_frames
            spectrogram = np.pad(
                spectrogram, ((0, pad_length), (0, 0)), mode="constant"
            )
            return [spectrogram]
        
        # Create overlapping segments
        segments = []
        hop_size = self.segment_length // 4  # 75% overlap for more training data
        
        for start in range(0, time_frames - self.segment_length + 1, hop_size):
            end = start + self.segment_length
            segment = spectrogram[start:end]
            segments.append(segment)
        
        return segments
    
    def get_data_iterator(
        self, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        infinite: bool = True
    ) -> Iterator[Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
        """
        Create data iterator for training
        Returns:
            Iterator yielding (spectrograms, labels) batches
        """
        
        def data_generator():
            while True:
                # Shuffle indices if requested
                indices = np.arange(len(self.metadata))
                if shuffle:
                    np.random.shuffle(indices)
                
                batch_spectrograms = []
                batch_labels = {dim: [] for dim in PERCEPTUAL_DIMENSIONS}
                
                for idx in indices:
                    row = self.metadata.iloc[idx]
                    perf_id = row['performance_id']
                    
                    # Get spectrogram
                    spectrogram = self._load_or_create_spectrogram(row['audio_path'])
                    if spectrogram is None:
                        continue
                    
                    # Get labels for this performance
                    perf_labels = self.labels[perf_id]
                    
                    # Create segments
                    segments = self._create_segments(spectrogram)
                    
                    for segment in segments:
                        batch_spectrograms.append(segment)
                        
                        # Replicate labels for each segment
                        for dim in PERCEPTUAL_DIMENSIONS:
                            if dim in perf_labels:
                                batch_labels[dim].append(perf_labels[dim])
                            else:
                                batch_labels[dim].append(0.0)  # Default value
                        
                        if len(batch_spectrograms) == batch_size:
                            # Convert to JAX arrays
                            spectrograms = jnp.array(batch_spectrograms)
                            labels = {dim: jnp.array(batch_labels[dim]) for dim in PERCEPTUAL_DIMENSIONS}
                            
                            yield spectrograms, labels
                            
                            # Reset batch
                            batch_spectrograms = []
                            batch_labels = {dim: [] for dim in PERCEPTUAL_DIMENSIONS}
                
                # Yield remaining batch if not empty
                if batch_spectrograms:
                    # Pad to batch_size if needed
                    while len(batch_spectrograms) < batch_size:
                        batch_spectrograms.append(batch_spectrograms[0])
                        for dim in PERCEPTUAL_DIMENSIONS:
                            batch_labels[dim].append(batch_labels[dim][0])
                    
                    spectrograms = jnp.array(batch_spectrograms)
                    labels = {dim: jnp.array(batch_labels[dim]) for dim in PERCEPTUAL_DIMENSIONS}
                    
                    yield spectrograms, labels
                
                if not infinite:
                    break
        
        return data_generator()
    
    def get_statistics(self) -> Dict:
        """Compute dataset statistics"""
        print("Computing PercePiano dataset statistics...")
        
        all_values = []
        total_segments = 0
        
        # Sample subset for statistics
        sample_size = min(10, len(self.metadata))
        
        for idx in range(sample_size):
            row = self.metadata.iloc[idx]
            spectrogram = self._load_or_create_spectrogram(row['audio_path'])
            
            if spectrogram is not None:
                segments = self._create_segments(spectrogram)
                total_segments += len(segments)
                
                for segment in segments:
                    all_values.extend(segment.flatten())
        
        if not all_values:
            return {"error": "No valid spectrograms found"}
        
        all_values = np.array(all_values)
        
        stats = {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "total_performances": len(self.metadata),
            "total_segments": total_segments,
            "segment_shape": (self.segment_length, self.n_mels),
            "perceptual_dimensions": len(PERCEPTUAL_DIMENSIONS)
        }
        
        print(f"Dataset statistics:")
        print(f"  Performances: {stats['total_performances']}")
        print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Total segments (sampled): {stats['total_segments']}")
        
        return stats


if __name__ == "__main__":
    # Test PercePiano dataset
    print("=== PercePiano Dataset Test ===")
    
    # Try to find dataset
    possible_paths = [
        "data/PercePiano",
        "PercePiano", 
        "archive/PercePiano"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ PercePiano dataset not found")
        print("Please provide the path to PercePiano dataset")
    else:
        dataset = PercePianoDataset(
            percepianos_root=dataset_path,
            cache_dir="cache/percepiano_preprocessed"
        )
        
        stats = dataset.get_statistics()
        
        # Test iterator
        print("\n🔄 Testing data iterator...")
        data_iter = dataset.get_data_iterator(batch_size=2, infinite=False)
        
        for i, (spectrograms, labels) in enumerate(data_iter):
            print(f"Batch {i+1}: Spectrograms {spectrograms.shape}")
            print(f"  Sample labels: {list(labels.keys())[:3]}")
            if i >= 2:  # Test first 3 batches
                break
        
        print("✅ PercePiano dataset test complete!")