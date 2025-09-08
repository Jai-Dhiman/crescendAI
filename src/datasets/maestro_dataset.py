#!/usr/bin/env python3
"""
MAESTRO Dataset Integration for SSAST Pre-training
Self-supervised training on large-scale piano audio data
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any, Union
import librosa
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy.typing as npt
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Track failed files
_failed_files = []
_failures_log_path = Path("logs/maestro_failures.csv")


class MAESTRODataset:
    """
    MAESTRO dataset loader for self-supervised pre-training
    Handles loading, preprocessing, and batching of piano audio data
    """

    def __init__(
        self,
        maestro_root: str,
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_length: int = 128,  # Time frames per segment
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize MAESTRO dataset
        Args:
            maestro_root: Path to MAESTRO dataset root directory
            target_sr: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            segment_length: Length of audio segments in time frames
            cache_dir: Directory to cache preprocessed spectrograms
        """
        self.maestro_root = Path(maestro_root)
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Load MAESTRO metadata
        self.metadata = self._load_metadata()

        print(f"🎹 MAESTRO Dataset initialized:")
        print(f"   Total files: {len(self.metadata)}")
        print(f"   Sample rate: {target_sr}Hz")
        print(f"   Mel bands: {n_mels}, Segment length: {segment_length}")
        if cache_dir:
            print(f"   Cache directory: {cache_dir}")

    def _load_metadata(self) -> pd.DataFrame:
        """Load MAESTRO metadata CSV"""
        metadata_path = self.maestro_root / "maestro-v3.0.0.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"MAESTRO metadata not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Filter for training split for pre-training
        train_df = df[df["split"] == "train"].copy()

        # Add full audio paths
        train_df["audio_path"] = train_df["audio_filename"].apply(
            lambda x: self.maestro_root / x
        )

        # Filter for existing files
        existing_files = train_df[train_df["audio_path"].apply(lambda x: x.exists())]

        print(f"Found {len(existing_files)} training audio files")

        return existing_files.reset_index(drop=True)

    def _extract_mel_spectrogram(
        self, audio_path: Path
    ) -> Optional[npt.NDArray[np.float64]]:
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
            # Log the error with stack trace
            logger.error(
                f"Error processing {audio_path}: {str(e)}\n{traceback.format_exc()}"
            )

            # Track the failure
            failure_info = {
                "audio_path": str(audio_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            _failed_files.append(failure_info)

            # Ensure the logs directory exists
            _failures_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to CSV log file
            if not _failures_log_path.exists():
                with open(_failures_log_path, "w") as f:
                    f.write("timestamp,audio_path,error\n")

            with open(_failures_log_path, "a") as f:
                f.write(
                    f'{failure_info["timestamp"]},{failure_info["audio_path"]},{failure_info["error"]}\n'
                )

            return None

    def _get_cache_path(self, audio_path: Path) -> Path:
        """Get cache path for preprocessed spectrogram"""
        if self.cache_dir is None:
            return None

        # Create cache filename based on audio path and parameters
        cache_name = f"{audio_path.stem}_{self.target_sr}_{self.n_mels}_{self.segment_length}.pkl"
        return self.cache_dir / cache_name

    def _load_or_create_spectrogram(self, audio_path: Path) -> np.ndarray:
        """Load cached spectrogram or create new one"""
        cache_path = self._get_cache_path(audio_path)

        # Try loading from cache
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed for {cache_path}: {e}")

        # Extract spectrogram
        spectrogram = self._extract_mel_spectrogram(audio_path)

        if spectrogram is None:
            return None

        # Save to cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(spectrogram, f)
            except Exception as e:
                print(f"Cache save failed for {cache_path}: {e}")

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
        hop_size = self.segment_length // 2  # 50% overlap

        for start in range(0, time_frames - self.segment_length + 1, hop_size):
            end = start + self.segment_length
            segment = spectrogram[start:end]
            segments.append(segment)

        return segments

    def preprocess_dataset(self, max_workers: int = 4) -> None:
        """Preprocess entire dataset and cache spectrograms"""
        print(f"Preprocessing MAESTRO dataset with {max_workers} workers...")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        def process_file(row):
            audio_path = row["audio_path"]
            spectrogram = self._load_or_create_spectrogram(audio_path)
            return audio_path.name, spectrogram is not None

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(process_file, [row for _, row in self.metadata.iterrows()])
            )

        successful = sum(1 for _, success in results if success)
        print(f"Successfully preprocessed {successful}/{len(results)} files")

    def get_data_iterator(
        self, batch_size: int = 32, shuffle: bool = True, infinite: bool = True
    ) -> Iterator[jnp.ndarray]:
        """
        Create data iterator for training
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            infinite: Whether to create infinite iterator
        Returns:
            Iterator yielding batches of spectrograms
        """

        def data_generator():
            while True:
                # Shuffle files if requested
                indices = np.arange(len(self.metadata))
                if shuffle:
                    np.random.shuffle(indices)

                batch = []

                for idx in indices:
                    row = self.metadata.iloc[idx]
                    spectrogram = self._load_or_create_spectrogram(row["audio_path"])

                    if spectrogram is None:
                        continue

                    # Create segments from this spectrogram
                    segments = self._create_segments(spectrogram)

                    for segment in segments:
                        batch.append(segment)

                        if len(batch) == batch_size:
                            # Yield batch as JAX array
                            yield jnp.array(batch)
                            batch = []

                # Yield remaining batch if not empty
                if batch and len(batch) > 0:
                    # Pad to batch_size if needed
                    while len(batch) < batch_size:
                        batch.append(batch[0])  # Repeat first sample
                    yield jnp.array(batch)

                if not infinite:
                    break

        return data_generator()

    def get_statistics(self) -> Dict:
        """Compute dataset statistics for normalization"""
        print("Computing dataset statistics...")

        all_values = []
        total_segments = 0

        for idx in range(min(50, len(self.metadata))):  # Sample first 50 files
            row = self.metadata.iloc[idx]
            spectrogram = self._load_or_create_spectrogram(row["audio_path"])

            if spectrogram is not None:
                segments = self._create_segments(spectrogram)
                total_segments += len(segments)

                for segment in segments:
                    all_values.extend(segment.flatten())

        all_values = np.array(all_values)

        stats = {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "total_segments": total_segments,
            "segment_shape": (self.segment_length, self.n_mels),
        }

        print(f"Dataset statistics:")
        print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Total segments (sampled): {stats['total_segments']}")

        return stats


def download_maestro_instructions():
    """Print instructions for downloading MAESTRO dataset"""
    print("\n📥 MAESTRO DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 50)
    print("1. Visit: https://magenta.tensorflow.org/datasets/maestro")
    print("2. Download MAESTRO v3.0.0 (approximately 200 GB)")
    print("3. Extract to your desired directory")
    print("4. Update the maestro_root path in your configuration")
    print("\nFor Google Colab:")
    print("- Consider downloading to Google Drive")
    print("- Use selective download for faster experimentation")
    print("- MAESTRO provides ~200 hours of piano performance data")


if __name__ == "__main__":
    # Test MAESTRO dataset integration
    print("=== MAESTRO Dataset Integration Test ===\n")

    # Check if MAESTRO dataset is available
    maestro_path = "data/maestro-v3.0.0"  # Adjust path as needed

    if not Path(maestro_path).exists():
        download_maestro_instructions()
        print(f"\n❌ MAESTRO dataset not found at: {maestro_path}")
        print("Please download and extract MAESTRO dataset first.")
        exit(1)

    # Initialize dataset
    dataset = MAESTRODataset(
        maestro_root=maestro_path,
        segment_length=128,
        cache_dir="cache/maestro_preprocessed",
    )

    # Get statistics
    stats = dataset.get_statistics()

    # Test data iterator
    print("\n🔄 Testing data iterator...")
    data_iter = dataset.get_data_iterator(batch_size=4, infinite=False)

    batch_count = 0
    for batch in data_iter:
        print(f"Batch {batch_count + 1}: {batch.shape}")
        batch_count += 1

        if batch_count >= 3:  # Test first 3 batches
            break

    print(f"\n✅ MAESTRO dataset integration complete!")
    print(f"Ready for SSAST pre-training with {len(dataset.metadata)} training files")
