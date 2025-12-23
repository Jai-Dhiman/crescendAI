"""
PyTorch Dataset for PercePiano with VirtuosoNet features.

This module provides a Dataset class that loads preprocessed VirtuosoNet
features (83-dim: 78 base + 5 unnorm) for training the PercePiano replica model.
This matches the SOTA configuration (R2 = 0.397).

Feature layout:
- Indices 0-77: Base VirtuosoNet features (z-score normalized where applicable)
- Index 78: midi_pitch_unnorm (raw MIDI pitch 21-108, used for key augmentation)
- Index 79: duration_unnorm
- Index 80: beat_importance_unnorm
- Index 81: measure_length_unnorm
- Index 82: following_rest_unnorm
"""

import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# No global flags - they don't work correctly with multiprocessing workers


class PercePianoVNetDataset(Dataset):
    """
    PyTorch Dataset for PercePiano with VirtuosoNet 83-dim features (SOTA config).

    Loads preprocessed pickle files containing:
    - input: (num_notes, 83) VirtuosoNet features (78 base + 5 unnorm)
    - note_location: dict with 'beat', 'measure', 'voice' arrays
    - labels: (19,) PercePiano scores

    The unnorm features (indices 78-82) preserve raw values before normalization,
    which is critical for key augmentation (midi_pitch_unnorm gives raw MIDI pitch 21-108).

    Args:
        data_dir: Path to preprocessed data directory (e.g., data/processed/percepiano_vnet/train)
        max_notes: Maximum number of notes per sample (for padding/truncation)
        transform: Optional transform to apply to features
        augment: Whether to apply key augmentation (random pitch shifts)
        pitch_std: Standard deviation used for midi_pitch normalization (for augmentation).
                   If None, will attempt to load from stat.pkl in parent directory.
        stats_file: Optional path to stats file. If None and pitch_std is None,
                    will look for stat.pkl in parent directory.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_notes: int = 1024,
        transform: Optional[callable] = None,
        augment: bool = False,
        pitch_std: Optional[float] = None,
        stats_file: Optional[Union[str, Path]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.max_notes = max_notes
        self.transform = transform
        self.augment = augment

        # Load pitch_std from stats file if not provided
        if pitch_std is None:
            self.pitch_std = self._load_pitch_std_from_stats(stats_file)
        else:
            self.pitch_std = pitch_std

        # Find all pickle files
        self.pkl_files = sorted(self.data_dir.glob("*.pkl"))
        if not self.pkl_files:
            raise ValueError(f"No pickle files found in {self.data_dir}")

        print(f"Found {len(self.pkl_files)} samples in {self.data_dir}")
        if augment:
            print(f"  Key augmentation ENABLED (pitch_std={self.pitch_std})")

    def _load_pitch_std_from_stats(
        self, stats_file: Optional[Union[str, Path]] = None
    ) -> float:
        """Load pitch_std from stats file."""
        if stats_file is None:
            # Look for stat.pkl in parent directory (data_dir is train/val/test subdirectory)
            stats_file = self.data_dir.parent / "stat.pkl"

        stats_path = Path(stats_file)
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
            if "std" in stats and "midi_pitch" in stats["std"]:
                loaded_std = stats["std"]["midi_pitch"]
                print(f"  Loaded pitch_std={loaded_std:.4f} from {stats_path}")
                return loaded_std

        # Fallback to default if stats file not found or doesn't have midi_pitch
        print(f"  Warning: Could not load pitch_std from stats, using default 12.0")
        return 12.0

    def __len__(self) -> int:
        return len(self.pkl_files)

    def _validate_sample(
        self,
        idx: int,
        input_features: np.ndarray,
        beat_indices: np.ndarray,
        labels: np.ndarray,
        num_notes: int,
    ) -> None:
        """
        Validate a sample for common issues that cause training to fail.

        Logs warnings (not errors) to help diagnose poor training performance.
        Only logs detailed diagnostics for the first sample in main process to avoid spam.
        """
        # Only log from main process (worker_info is None) to avoid spam from workers
        worker_info = torch.utils.data.get_worker_info()
        is_main_process = worker_info is None

        sample_name = self.pkl_files[idx].stem

        # 1. Check for NaN/Inf in features (always check, only warn from main)
        nan_count = np.isnan(input_features).sum()
        inf_count = np.isinf(input_features).sum()
        if nan_count > 0 and is_main_process:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has {nan_count} NaN values in features! "
                "This will cause training to produce NaN losses.",
                RuntimeWarning,
            )
        if inf_count > 0 and is_main_process:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has {inf_count} Inf values in features! "
                "This will cause training to produce Inf losses.",
                RuntimeWarning,
            )

        # 2. Check label range (should be 0-1 for sigmoid output)
        if (labels.min() < 0 or labels.max() > 1) and is_main_process:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has labels outside [0,1]: "
                f"[{labels.min():.3f}, {labels.max():.3f}]. "
                "Model uses sigmoid output (0-1), causing MSE mismatch!",
                RuntimeWarning,
            )

        # Beat index gap check removed - gaps are common and don't break training

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample.

        Returns:
            Dictionary with:
            - input_features: (max_notes, 84) float tensor
            - note_locations_beat: (max_notes,) long tensor
            - note_locations_measure: (max_notes,) long tensor
            - note_locations_voice: (max_notes,) long tensor
            - scores: (19,) float tensor
            - num_notes: int - actual number of notes before padding
            - attention_mask: (max_notes,) bool tensor - True for valid notes
        """
        # Load pickle file
        with open(self.pkl_files[idx], "rb") as f:
            data = pickle.load(f)

        # Get input features - REQUIRED
        if "input" not in data:
            raise ValueError(f"Missing 'input' features in {self.pkl_files[idx]}")
        input_features = data["input"]  # (num_notes, 78)
        num_notes = min(input_features.shape[0], self.max_notes)

        # Get note locations - beat and measure are REQUIRED, voice can default to 1
        if "note_location" not in data:
            raise ValueError(f"Missing 'note_location' in {self.pkl_files[idx]}")
        note_loc = data["note_location"]

        if "beat" not in note_loc:
            raise ValueError(
                f"Missing 'beat' in note_location for {self.pkl_files[idx]}"
            )
        if "measure" not in note_loc:
            raise ValueError(
                f"Missing 'measure' in note_location for {self.pkl_files[idx]}"
            )

        beat_indices = np.array(note_loc["beat"], dtype=np.int64)
        measure_indices = np.array(note_loc["measure"], dtype=np.int64)
        # Voice can default to 1 (single voice) if not present
        voice_indices = np.array(
            note_loc.get("voice", np.ones(num_notes)), dtype=np.int64
        )

        # CRITICAL FIX: Convert sparse indices to dense sequential indices
        # Original data has non-sequential values like [0, 1, 4, 11, 15] from the score
        # But hierarchy_utils expects sequential indices like [0, 1, 2, 3, 4]
        # This densification ensures proper beat->measure and measure aggregation
        beat_indices = self._densify_indices(beat_indices)
        measure_indices = self._densify_indices(measure_indices)
        # Voice indices are typically already dense, but apply anyway for safety
        voice_indices = self._densify_indices(voice_indices)

        # CRITICAL FIX: hierarchy_utils expects indices to start from 1, not 0
        # It uses torch.nonzero() which treats 0 as padding
        # Shift all indices to start from 1
        if beat_indices.min() == 0:
            beat_indices = beat_indices + 1
        if measure_indices.min() == 0:
            measure_indices = measure_indices + 1
        if voice_indices.min() == 0:
            voice_indices = voice_indices + 1

        # Get labels - REQUIRED
        if "labels" not in data:
            raise ValueError(f"Missing 'labels' in {self.pkl_files[idx]}")
        labels = np.array(data["labels"], dtype=np.float32)

        # VALIDATION: Check for common issues that cause training to fail
        # Note: beat_indices are already shifted to start from 1 at this point
        self._validate_sample(idx, input_features, beat_indices, labels, num_notes)

        # Key augmentation (random pitch shift) - training only
        if self.augment:
            input_features = self._apply_key_augmentation(
                input_features.copy(), num_notes
            )

        # Pad or truncate to max_notes
        # Only use first 78 features for model input (SOTA configuration)
        # The 5 unnorm features (indices 78-82) are only for key augmentation, not model input
        BASE_FEATURE_DIM = 78
        model_features = input_features[
            :, :BASE_FEATURE_DIM
        ]  # Only normalized features

        padded_features = np.zeros((self.max_notes, BASE_FEATURE_DIM), dtype=np.float32)
        padded_beat = np.zeros(self.max_notes, dtype=np.int64)
        padded_measure = np.zeros(self.max_notes, dtype=np.int64)
        padded_voice = np.zeros(self.max_notes, dtype=np.int64)
        attention_mask = np.zeros(self.max_notes, dtype=bool)

        # Fill with actual data
        actual_notes = min(num_notes, self.max_notes)
        padded_features[:actual_notes] = model_features[:actual_notes]
        padded_beat[:actual_notes] = beat_indices[:actual_notes]
        padded_measure[:actual_notes] = measure_indices[:actual_notes]
        padded_voice[:actual_notes] = voice_indices[:actual_notes]
        attention_mask[:actual_notes] = True

        # Convert to tensors
        result = {
            "input_features": torch.from_numpy(padded_features),
            "note_locations_beat": torch.from_numpy(padded_beat),
            "note_locations_measure": torch.from_numpy(padded_measure),
            "note_locations_voice": torch.from_numpy(padded_voice),
            "scores": torch.from_numpy(labels),
            "num_notes": actual_notes,
            "attention_mask": torch.from_numpy(attention_mask),
        }

        # Apply transform if provided
        if self.transform is not None:
            result = self.transform(result)

        return result

    def _densify_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert sparse non-sequential indices to dense sequential indices.

        The original PercePiano data has beat/measure indices from the score which
        may be non-sequential (e.g., [0, 0, 1, 1, 4, 4, 11, 11, 15, 15] for beats).
        The hierarchy_utils functions expect sequential indices (0, 1, 2, 3, 4...).

        This converts sparse to dense by assigning sequential indices based on
        unique values in order of first appearance.

        Example:
            Input:  [0, 0, 0, 1, 1, 4, 4, 11, 11, 15, 15]
            Output: [0, 0, 0, 1, 1, 2, 2, 3,  3,  4,  4]

        Args:
            indices: Array of non-sequential indices

        Returns:
            Array of sequential indices (0, 1, 2, ..., n_unique-1)
        """
        # Get unique values in order of first appearance
        _, unique_indices = np.unique(indices, return_inverse=True)
        return unique_indices

    def _apply_key_augmentation(
        self, input_features: np.ndarray, num_notes: int
    ) -> np.ndarray:
        """
        Apply random key augmentation (pitch shift) to features.

        Matches original PercePiano training augmentation for better generalization.
        Uses midi_pitch_unnorm (raw MIDI pitch 21-108) for calculating valid shift range.

        Args:
            input_features: (num_notes, 83) feature array (78 base + 5 unnorm)
            num_notes: Actual number of notes (before padding)

        Returns:
            Augmented feature array
        """
        # Feature indices based on VNET_INPUT_KEYS order (SOTA 78-feature config):
        # midi_pitch (normalized) is at index 0
        # pitch vector is at index 13: [octave (1), pitch_class_one_hot (12)]
        # midi_pitch_unnorm (raw MIDI 21-108) is at index 78
        MIDI_PITCH_IDX = 0
        PITCH_VEC_START = 13  # After 13 scalar features (section_tempo removed)
        PITCH_CLASS_START = 14  # octave at 13, pitch class starts at 14
        PITCH_CLASS_END = 26  # 12 pitch classes (indices 14-25)
        MIDI_PITCH_UNNORM_IDX = 78  # Raw MIDI pitch (21-108)

        # Get current pitch range from midi_pitch_unnorm (raw MIDI values)
        # This is much more accurate than estimating from normalized values
        raw_pitches = input_features[:num_notes, MIDI_PITCH_UNNORM_IDX]

        # For piano: MIDI 21 (A0) to 108 (C8)
        # Calculate safe shift range to stay within piano range
        current_max = raw_pitches.max()
        current_min = raw_pitches.min()

        max_up = min(108 - current_max, 7)  # Max 7 semitones up
        max_down = min(current_min - 21, 5)  # Max 5 semitones down

        # Clamp to valid range
        max_up = max(0, int(max_up))
        max_down = max(0, int(max_down))

        if max_up == 0 and max_down == 0:
            return input_features

        # Random key shift
        key_shift = random.randint(-max_down, max_up)

        if key_shift == 0:
            return input_features

        # Apply shift to midi_pitch (normalized, so shift by key_shift/std)
        input_features[:num_notes, MIDI_PITCH_IDX] += key_shift / self.pitch_std

        # Apply shift to midi_pitch_unnorm (raw MIDI values)
        input_features[:num_notes, MIDI_PITCH_UNNORM_IDX] += key_shift

        # Roll the pitch class one-hot encoding
        pitch_class = input_features[
            :num_notes, PITCH_CLASS_START:PITCH_CLASS_END
        ].copy()
        pitch_class_shifted = np.roll(pitch_class, key_shift, axis=1)
        input_features[:num_notes, PITCH_CLASS_START:PITCH_CLASS_END] = (
            pitch_class_shifted
        )

        # Adjust octave if shift crosses octave boundary
        # Octave is normalized around octave 4, each octave = 0.25
        octave_adjustment = key_shift // 12
        if octave_adjustment != 0:
            input_features[:num_notes, PITCH_VEC_START] += octave_adjustment * 0.25

        return input_features


class PercePianoVNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for PercePiano with VirtuosoNet features.

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size for dataloaders (SOTA: 8)
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data (SOTA: False)
        pitch_std: Standard deviation for pitch normalization (for augmentation).
                   If None (default), auto-loads from stat.pkl in data_dir.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,  # SOTA uses batch_size=8
        max_notes: int = 1024,
        num_workers: int = 4,
        augment_train: bool = False,  # SOTA doesn't use augmentation
        pitch_std: Optional[float] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_notes = max_notes
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.pitch_std = pitch_std  # None = auto-load from stats
        self._stats_file = self.data_dir / "stat.pkl"

        self.train_dataset: Optional[PercePianoVNetDataset] = None
        self.val_dataset: Optional[PercePianoVNetDataset] = None
        self.test_dataset: Optional[PercePianoVNetDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"

            if train_dir.exists():
                # Training dataset with augmentation enabled
                self.train_dataset = PercePianoVNetDataset(
                    train_dir,
                    max_notes=self.max_notes,
                    augment=self.augment_train,
                    pitch_std=self.pitch_std,
                    stats_file=self._stats_file,
                )
            if val_dir.exists():
                # Validation dataset without augmentation
                self.val_dataset = PercePianoVNetDataset(
                    val_dir,
                    max_notes=self.max_notes,
                    stats_file=self._stats_file,
                )

        if stage == "test" or stage is None:
            test_dir = self.data_dir / "test"
            if test_dir.exists():
                self.test_dataset = PercePianoVNetDataset(
                    test_dir,
                    max_notes=self.max_notes,
                    stats_file=self._stats_file,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_vnet_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 8,  # SOTA uses batch_size=8
    max_notes: int = 1024,
    num_workers: int = 4,
    augment_train: bool = False,  # SOTA doesn't use augmentation
    pitch_std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for VirtuosoNet features.

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size (SOTA: 8)
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data (SOTA: False)
        pitch_std: Standard deviation for pitch normalization (for augmentation).
                   If None (default), auto-loads from stat.pkl in data_dir.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    stats_file = data_dir / "stat.pkl"

    # Training dataset with augmentation enabled by default
    train_dataset = PercePianoVNetDataset(
        data_dir / "train",
        max_notes=max_notes,
        augment=augment_train,
        pitch_std=pitch_std,
        stats_file=stats_file,
    )
    # Validation and test datasets without augmentation
    val_dataset = PercePianoVNetDataset(
        data_dir / "val",
        max_notes=max_notes,
        stats_file=stats_file,
    )
    test_dataset = PercePianoVNetDataset(
        data_dir / "test",
        max_notes=max_notes,
        stats_file=stats_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class PercePianoKFoldDataset(Dataset):
    """
    K-Fold Dataset for PercePiano with VirtuosoNet features.

    Unlike PercePianoVNetDataset which uses fixed train/val/test directories,
    this dataset uses fold assignments to dynamically select samples based on
    the current fold and mode.

    For fold_id=0, mode='val': uses samples from fold 0 as validation
    For fold_id=0, mode='train': uses samples from folds 1,2,3 as training

    Normalization stats are computed from training folds only (not val fold).

    Args:
        data_dir: Path to data directory containing all .pkl files (or train/val/test subdirs)
        fold_assignments: Dictionary mapping sample_name -> {'fold': 0-3 or 'test', 'piece_id': str}
        fold_id: Which fold to use as validation (0 to n_folds-1)
        mode: 'train' or 'val'
        max_notes: Maximum number of notes per sample
        augment: Whether to apply key augmentation (only for training)
        normalization_stats: Optional pre-computed stats dict. If None, will be computed.
    """

    # Class-level cache for loaded sample data
    _sample_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        data_dir: Union[str, Path],
        fold_assignments: Dict[str, Dict[str, Union[int, str]]],
        fold_id: int,
        mode: str,
        max_notes: int = 1024,
        augment: bool = False,
        normalization_stats: Optional[Dict[str, Any]] = None,
    ):
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got: {mode}")

        self.data_dir = Path(data_dir)
        self.fold_assignments = fold_assignments
        self.fold_id = fold_id
        self.mode = mode
        self.max_notes = max_notes
        self.augment = augment and (mode == "train")  # Only augment training

        # Get sample list for this fold/mode
        self.sample_files = self._get_sample_files()
        if not self.sample_files:
            raise ValueError(f"No samples found for fold {fold_id}, mode '{mode}'")

        # Load or compute normalization stats
        if normalization_stats is not None:
            self.stats = normalization_stats
        else:
            self.stats = self._compute_normalization_stats()

        self.pitch_std = self.stats.get("std", {}).get("midi_pitch", 12.0)

        print(
            f"KFold Dataset: fold={fold_id}, mode={mode}, samples={len(self.sample_files)}"
        )
        if self.augment:
            print(f"  Key augmentation ENABLED (pitch_std={self.pitch_std:.4f})")

    def _get_sample_files(self) -> List[Path]:
        """Get list of sample files for this fold/mode."""
        sample_names = []

        for sample_name, info in self.fold_assignments.items():
            sample_fold = info["fold"]

            # Skip test samples
            if sample_fold == "test":
                continue

            if self.mode == "val":
                # Validation: use only samples from this fold
                if sample_fold == self.fold_id:
                    sample_names.append(sample_name)
            else:
                # Training: use samples from all OTHER folds
                if sample_fold != self.fold_id:
                    sample_names.append(sample_name)

        # Find actual file paths
        sample_files = []
        for sample_name in sample_names:
            # Try to find the file in data_dir or its subdirectories
            file_path = self._find_sample_file(sample_name)
            if file_path is not None:
                sample_files.append(file_path)

        return sorted(sample_files)

    def _find_sample_file(self, sample_name: str) -> Optional[Path]:
        """Find the file path for a sample name."""
        # Add .pkl extension if not present
        if not sample_name.endswith(".pkl"):
            sample_name = f"{sample_name}.pkl"

        # Check direct path
        direct_path = self.data_dir / sample_name
        if direct_path.exists():
            return direct_path

        # Check subdirectories (train/val/test)
        for subdir in ["train", "val", "test"]:
            subdir_path = self.data_dir / subdir / sample_name
            if subdir_path.exists():
                return subdir_path

        return None

    def _compute_normalization_stats(self) -> Dict[str, Any]:
        """
        Compute normalization statistics from training samples only.

        Returns dict with 'mean' and 'std' for each feature.
        """
        print(
            f"  Computing normalization stats from {len(self.sample_files)} training samples..."
        )

        # Collect all features from training samples
        all_features = []

        for file_path in self.sample_files:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            # Only use base 78 features for stats (not unnormalized) - SOTA config
            features = data["input"][:, :78]
            all_features.append(features)

        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)

        # Compute mean and std per feature
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)

        # Avoid division by zero
        std = np.where(std < 1e-6, 1.0, std)

        # Create stats dict in same format as stat.pkl
        # Note: These are per-dimension stats, but for key augmentation
        # we specifically need midi_pitch std
        stats = {
            "mean": {"midi_pitch": float(mean[0])},
            "std": {"midi_pitch": float(std[0])},
            "_raw_mean": mean,  # Keep raw arrays for potential future use
            "_raw_std": std,
        }

        print(f"  Computed stats: pitch_std={stats['std']['midi_pitch']:.4f}")
        return stats

    def get_normalization_stats(self) -> Dict[str, Any]:
        """Return normalization stats for sharing with validation dataset."""
        return self.stats

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a single sample."""
        file_path = self.sample_files[idx]

        # Load pickle file
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Get input features
        input_features = data["input"]
        num_notes = min(input_features.shape[0], self.max_notes)

        # Get note locations
        note_loc = data["note_location"]
        beat_indices = np.array(note_loc["beat"], dtype=np.int64)
        measure_indices = np.array(note_loc["measure"], dtype=np.int64)
        voice_indices = np.array(
            note_loc.get("voice", np.ones(num_notes)), dtype=np.int64
        )

        # Shift indices to start from 1 (hierarchy_utils expects this)
        if beat_indices.min() == 0:
            beat_indices = beat_indices + 1
        if measure_indices.min() == 0:
            measure_indices = measure_indices + 1
        if voice_indices.min() == 0:
            voice_indices = voice_indices + 1

        # Get labels
        labels = np.array(data["labels"], dtype=np.float32)

        # Key augmentation
        if self.augment:
            input_features = self._apply_key_augmentation(
                input_features.copy(), num_notes
            )

        # Only use first 78 features for model input (SOTA configuration)
        BASE_FEATURE_DIM = 78
        model_features = input_features[:, :BASE_FEATURE_DIM]

        # Pad or truncate
        padded_features = np.zeros((self.max_notes, BASE_FEATURE_DIM), dtype=np.float32)
        padded_beat = np.zeros(self.max_notes, dtype=np.int64)
        padded_measure = np.zeros(self.max_notes, dtype=np.int64)
        padded_voice = np.zeros(self.max_notes, dtype=np.int64)
        attention_mask = np.zeros(self.max_notes, dtype=bool)

        actual_notes = min(num_notes, self.max_notes)
        padded_features[:actual_notes] = model_features[:actual_notes]
        padded_beat[:actual_notes] = beat_indices[:actual_notes]
        padded_measure[:actual_notes] = measure_indices[:actual_notes]
        padded_voice[:actual_notes] = voice_indices[:actual_notes]
        attention_mask[:actual_notes] = True

        return {
            "input_features": torch.from_numpy(padded_features),
            "note_locations_beat": torch.from_numpy(padded_beat),
            "note_locations_measure": torch.from_numpy(padded_measure),
            "note_locations_voice": torch.from_numpy(padded_voice),
            "scores": torch.from_numpy(labels),
            "num_notes": actual_notes,
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def _apply_key_augmentation(
        self, input_features: np.ndarray, num_notes: int
    ) -> np.ndarray:
        """Apply random key augmentation (pitch shift)."""
        # SOTA 78-feature configuration indices
        MIDI_PITCH_IDX = 0
        PITCH_VEC_START = 13  # After 13 scalar features
        PITCH_CLASS_START = 14  # octave at 13, pitch class starts at 14
        PITCH_CLASS_END = 26  # 12 pitch classes (indices 14-25)
        MIDI_PITCH_UNNORM_IDX = 78

        raw_pitches = input_features[:num_notes, MIDI_PITCH_UNNORM_IDX]
        current_max = raw_pitches.max()
        current_min = raw_pitches.min()

        max_up = min(108 - current_max, 7)
        max_down = min(current_min - 21, 5)
        max_up = max(0, int(max_up))
        max_down = max(0, int(max_down))

        if max_up == 0 and max_down == 0:
            return input_features

        key_shift = random.randint(-max_down, max_up)
        if key_shift == 0:
            return input_features

        input_features[:num_notes, MIDI_PITCH_IDX] += key_shift / self.pitch_std
        input_features[:num_notes, MIDI_PITCH_UNNORM_IDX] += key_shift

        pitch_class = input_features[
            :num_notes, PITCH_CLASS_START:PITCH_CLASS_END
        ].copy()
        pitch_class_shifted = np.roll(pitch_class, key_shift, axis=1)
        input_features[:num_notes, PITCH_CLASS_START:PITCH_CLASS_END] = (
            pitch_class_shifted
        )

        octave_adjustment = key_shift // 12
        if octave_adjustment != 0:
            input_features[:num_notes, PITCH_VEC_START] += octave_adjustment * 0.25

        return input_features


class PercePianoKFoldDataModule(pl.LightningDataModule):
    """
    K-Fold DataModule for PercePiano with VirtuosoNet features (SOTA config).

    Creates train/val dataloaders for a specific fold of k-fold cross-validation.
    Normalization stats are computed from training folds only.

    Args:
        data_dir: Path to data directory
        fold_assignments: Dictionary mapping sample_name -> {'fold': 0-3 or 'test', 'piece_id': str}
        fold_id: Which fold to use as validation (0 to n_folds-1)
        batch_size: Batch size for dataloaders (SOTA: 8)
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data (SOTA: False)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        fold_assignments: Dict[str, Dict[str, Union[int, str]]],
        fold_id: int,
        batch_size: int = 8,  # SOTA uses batch_size=8
        max_notes: int = 1024,
        num_workers: int = 4,
        augment_train: bool = False,  # SOTA doesn't use augmentation
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.fold_assignments = fold_assignments
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.max_notes = max_notes
        self.num_workers = num_workers
        self.augment_train = augment_train

        self.train_dataset: Optional[PercePianoKFoldDataset] = None
        self.val_dataset: Optional[PercePianoKFoldDataset] = None
        self._normalization_stats: Optional[Dict[str, Any]] = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for this fold."""
        if stage == "fit" or stage is None:
            # Create training dataset first to compute normalization stats
            self.train_dataset = PercePianoKFoldDataset(
                data_dir=self.data_dir,
                fold_assignments=self.fold_assignments,
                fold_id=self.fold_id,
                mode="train",
                max_notes=self.max_notes,
                augment=self.augment_train,
                normalization_stats=None,  # Will compute from training data
            )

            # Get stats from training dataset
            self._normalization_stats = self.train_dataset.get_normalization_stats()

            # Create validation dataset with same stats
            self.val_dataset = PercePianoKFoldDataset(
                data_dir=self.data_dir,
                fold_assignments=self.fold_assignments,
                fold_id=self.fold_id,
                mode="val",
                max_notes=self.max_notes,
                augment=False,  # Never augment validation
                normalization_stats=self._normalization_stats,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PercePianoTestDataset(Dataset):
    """
    Test Dataset for PercePiano with VirtuosoNet features.

    Loads only the test samples (fold='test') from fold assignments.

    Args:
        data_dir: Path to data directory
        fold_assignments: Dictionary mapping sample_name -> {'fold': 0-3 or 'test', 'piece_id': str}
        max_notes: Maximum number of notes per sample
        normalization_stats: Optional pre-computed stats dict
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        fold_assignments: Dict[str, Dict[str, Union[int, str]]],
        max_notes: int = 1024,
        normalization_stats: Optional[Dict[str, Any]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.fold_assignments = fold_assignments
        self.max_notes = max_notes

        # Get test samples
        self.sample_files = self._get_test_files()
        if not self.sample_files:
            raise ValueError("No test samples found in fold assignments")

        self.stats = normalization_stats or {}
        self.pitch_std = self.stats.get("std", {}).get("midi_pitch", 12.0)

        print(f"Test Dataset: {len(self.sample_files)} samples")

    def _get_test_files(self) -> List[Path]:
        """Get list of test sample files."""
        sample_names = [
            name
            for name, info in self.fold_assignments.items()
            if info["fold"] == "test"
        ]

        sample_files = []
        for sample_name in sample_names:
            file_path = self._find_sample_file(sample_name)
            if file_path is not None:
                sample_files.append(file_path)

        return sorted(sample_files)

    def _find_sample_file(self, sample_name: str) -> Optional[Path]:
        """Find the file path for a sample name."""
        if not sample_name.endswith(".pkl"):
            sample_name = f"{sample_name}.pkl"

        direct_path = self.data_dir / sample_name
        if direct_path.exists():
            return direct_path

        for subdir in ["train", "val", "test"]:
            subdir_path = self.data_dir / subdir / sample_name
            if subdir_path.exists():
                return subdir_path

        return None

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a single sample."""
        file_path = self.sample_files[idx]

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        input_features = data["input"]
        num_notes = min(input_features.shape[0], self.max_notes)

        note_loc = data["note_location"]
        beat_indices = np.array(note_loc["beat"], dtype=np.int64)
        measure_indices = np.array(note_loc["measure"], dtype=np.int64)
        voice_indices = np.array(
            note_loc.get("voice", np.ones(num_notes)), dtype=np.int64
        )

        if beat_indices.min() == 0:
            beat_indices = beat_indices + 1
        if measure_indices.min() == 0:
            measure_indices = measure_indices + 1
        if voice_indices.min() == 0:
            voice_indices = voice_indices + 1

        labels = np.array(data["labels"], dtype=np.float32)

        # SOTA configuration uses 78 base features
        BASE_FEATURE_DIM = 78
        model_features = input_features[:, :BASE_FEATURE_DIM]

        padded_features = np.zeros((self.max_notes, BASE_FEATURE_DIM), dtype=np.float32)
        padded_beat = np.zeros(self.max_notes, dtype=np.int64)
        padded_measure = np.zeros(self.max_notes, dtype=np.int64)
        padded_voice = np.zeros(self.max_notes, dtype=np.int64)
        attention_mask = np.zeros(self.max_notes, dtype=bool)

        actual_notes = min(num_notes, self.max_notes)
        padded_features[:actual_notes] = model_features[:actual_notes]
        padded_beat[:actual_notes] = beat_indices[:actual_notes]
        padded_measure[:actual_notes] = measure_indices[:actual_notes]
        padded_voice[:actual_notes] = voice_indices[:actual_notes]
        attention_mask[:actual_notes] = True

        return {
            "input_features": torch.from_numpy(padded_features),
            "note_locations_beat": torch.from_numpy(padded_beat),
            "note_locations_measure": torch.from_numpy(padded_measure),
            "note_locations_voice": torch.from_numpy(padded_voice),
            "scores": torch.from_numpy(labels),
            "num_notes": actual_notes,
            "attention_mask": torch.from_numpy(attention_mask),
        }
