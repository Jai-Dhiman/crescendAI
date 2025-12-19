"""
PyTorch Dataset for PercePiano with VirtuosoNet features.

This module provides a Dataset class that loads preprocessed VirtuosoNet
features (84-dim: 79 base + 5 unnorm) for training the PercePiano replica model.

Feature layout:
- Indices 0-78: Base VirtuosoNet features (z-score normalized where applicable)
- Index 79: midi_pitch_unnorm (raw MIDI pitch 21-108, used for key augmentation)
- Index 80: duration_unnorm
- Index 81: beat_importance_unnorm
- Index 82: measure_length_unnorm
- Index 83: following_rest_unnorm
"""

import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Global flag to track if we've logged diagnostics (to avoid spam)
_FIRST_SAMPLE_LOGGED = False


class PercePianoVNetDataset(Dataset):
    """
    PyTorch Dataset for PercePiano with VirtuosoNet 84-dim features.

    Loads preprocessed pickle files containing:
    - input: (num_notes, 84) VirtuosoNet features (79 base + 5 unnorm)
    - note_location: dict with 'beat', 'measure', 'voice' arrays
    - labels: (19,) PercePiano scores

    The unnorm features (indices 79-83) preserve raw values before normalization,
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

    def _load_pitch_std_from_stats(self, stats_file: Optional[Union[str, Path]] = None) -> float:
        """Load pitch_std from stats file."""
        if stats_file is None:
            # Look for stat.pkl in parent directory (data_dir is train/val/test subdirectory)
            stats_file = self.data_dir.parent / 'stat.pkl'

        stats_path = Path(stats_file)
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            if 'std' in stats and 'midi_pitch' in stats['std']:
                loaded_std = stats['std']['midi_pitch']
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
        Only logs detailed diagnostics for the first sample to avoid spam.
        """
        global _FIRST_SAMPLE_LOGGED
        sample_name = self.pkl_files[idx].stem

        # 1. Check for NaN/Inf in features
        nan_count = np.isnan(input_features).sum()
        inf_count = np.isinf(input_features).sum()
        if nan_count > 0:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has {nan_count} NaN values in features! "
                "This will cause training to produce NaN losses.",
                RuntimeWarning
            )
        if inf_count > 0:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has {inf_count} Inf values in features! "
                "This will cause training to produce Inf losses.",
                RuntimeWarning
            )

        # 2. Check label range (should be 0-1 for sigmoid output)
        if labels.min() < 0 or labels.max() > 1:
            warnings.warn(
                f"[DATA] Sample '{sample_name}' has labels outside [0,1]: "
                f"[{labels.min():.3f}, {labels.max():.3f}]. "
                "Model uses sigmoid output (0-1), causing MSE mismatch!",
                RuntimeWarning
            )

        # 3. Check beat indices for gaps (critical for hierarchy_utils)
        # hierarchy_utils.find_boundaries_batch expects diff == 1 at boundaries
        valid_beats = beat_indices[:num_notes]
        if len(valid_beats) > 1:
            diffs = np.diff(valid_beats)
            # Beat indices should either stay same (within beat) or increase by 1 (new beat)
            invalid_diffs = np.sum((diffs != 0) & (diffs != 1))
            if invalid_diffs > 0:
                warnings.warn(
                    f"[DATA] Sample '{sample_name}' has {invalid_diffs} beat index gaps > 1. "
                    "hierarchy_utils expects consecutive beat indices. "
                    f"Beat range: [{valid_beats.min()}, {valid_beats.max()}]",
                    RuntimeWarning
                )

        # 4. Log detailed diagnostics for first sample only
        if not _FIRST_SAMPLE_LOGGED:
            _FIRST_SAMPLE_LOGGED = True
            print(f"\n[DATASET DIAGNOSTICS] First sample loaded: {sample_name}")
            print(f"  Feature shape: {input_features.shape}")
            print(f"  Feature stats: mean={input_features.mean():.4f}, std={input_features.std():.4f}")
            print(f"  Feature range: [{input_features.min():.4f}, {input_features.max():.4f}]")
            print(f"  NaN/Inf count: {nan_count}/{inf_count}")
            print(f"  Labels: mean={labels.mean():.4f}, range=[{labels.min():.4f}, {labels.max():.4f}]")
            print(f"  Beat indices: range=[{beat_indices[:num_notes].min()}, {beat_indices[:num_notes].max()}]")
            print(f"  Num notes: {num_notes}")

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
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)

        # Get input features - REQUIRED
        if 'input' not in data:
            raise ValueError(f"Missing 'input' features in {self.pkl_files[idx]}")
        input_features = data['input']  # (num_notes, 78)
        num_notes = min(input_features.shape[0], self.max_notes)

        # Get note locations - beat and measure are REQUIRED, voice can default to 1
        if 'note_location' not in data:
            raise ValueError(f"Missing 'note_location' in {self.pkl_files[idx]}")
        note_loc = data['note_location']

        if 'beat' not in note_loc:
            raise ValueError(f"Missing 'beat' in note_location for {self.pkl_files[idx]}")
        if 'measure' not in note_loc:
            raise ValueError(f"Missing 'measure' in note_location for {self.pkl_files[idx]}")

        beat_indices = np.array(note_loc['beat'], dtype=np.int64)
        measure_indices = np.array(note_loc['measure'], dtype=np.int64)
        # Voice can default to 1 (single voice) if not present
        voice_indices = np.array(note_loc.get('voice', np.ones(num_notes)), dtype=np.int64)

        # Get labels - REQUIRED
        if 'labels' not in data:
            raise ValueError(f"Missing 'labels' in {self.pkl_files[idx]}")
        labels = np.array(data['labels'], dtype=np.float32)

        # VALIDATION: Check for common issues that cause training to fail
        self._validate_sample(idx, input_features, beat_indices, labels, num_notes)

        # Key augmentation (random pitch shift) - training only
        if self.augment:
            input_features = self._apply_key_augmentation(input_features.copy(), num_notes)

        # Pad or truncate to max_notes
        padded_features = np.zeros((self.max_notes, input_features.shape[1]), dtype=np.float32)
        padded_beat = np.zeros(self.max_notes, dtype=np.int64)
        padded_measure = np.zeros(self.max_notes, dtype=np.int64)
        padded_voice = np.zeros(self.max_notes, dtype=np.int64)
        attention_mask = np.zeros(self.max_notes, dtype=bool)

        # Fill with actual data
        actual_notes = min(num_notes, self.max_notes)
        padded_features[:actual_notes] = input_features[:actual_notes]
        padded_beat[:actual_notes] = beat_indices[:actual_notes]
        padded_measure[:actual_notes] = measure_indices[:actual_notes]
        padded_voice[:actual_notes] = voice_indices[:actual_notes]
        attention_mask[:actual_notes] = True

        # Convert to tensors
        result = {
            'input_features': torch.from_numpy(padded_features),
            'note_locations_beat': torch.from_numpy(padded_beat),
            'note_locations_measure': torch.from_numpy(padded_measure),
            'note_locations_voice': torch.from_numpy(padded_voice),
            'scores': torch.from_numpy(labels),
            'num_notes': actual_notes,
            'attention_mask': torch.from_numpy(attention_mask),
        }

        # Apply transform if provided
        if self.transform is not None:
            result = self.transform(result)

        return result

    def _apply_key_augmentation(
        self, input_features: np.ndarray, num_notes: int
    ) -> np.ndarray:
        """
        Apply random key augmentation (pitch shift) to features.

        Matches original PercePiano training augmentation for better generalization.
        Uses midi_pitch_unnorm (raw MIDI pitch 21-108) for calculating valid shift range.

        Args:
            input_features: (num_notes, 84) feature array
            num_notes: Actual number of notes (before padding)

        Returns:
            Augmented feature array
        """
        # Feature indices based on VNET_INPUT_KEYS order:
        # midi_pitch (normalized) is at index 0
        # pitch vector is at index 14: [octave (1), pitch_class_one_hot (12)]
        # midi_pitch_unnorm (raw MIDI 21-108) is at index 79
        MIDI_PITCH_IDX = 0
        PITCH_VEC_START = 14  # After 14 scalar features
        PITCH_CLASS_START = 15  # octave at 14, pitch class starts at 15
        PITCH_CLASS_END = 27  # 12 pitch classes
        MIDI_PITCH_UNNORM_IDX = 79  # Raw MIDI pitch (21-108)

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
        pitch_class = input_features[:num_notes, PITCH_CLASS_START:PITCH_CLASS_END].copy()
        pitch_class_shifted = np.roll(pitch_class, key_shift, axis=1)
        input_features[:num_notes, PITCH_CLASS_START:PITCH_CLASS_END] = pitch_class_shifted

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
        batch_size: Batch size for dataloaders
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data
        pitch_std: Standard deviation for pitch normalization (for augmentation).
                   If None (default), auto-loads from stat.pkl in data_dir.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,  # Original PercePiano uses batch_size=32 (parser.py:107)
        max_notes: int = 1024,
        num_workers: int = 4,
        augment_train: bool = True,
        pitch_std: Optional[float] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_notes = max_notes
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.pitch_std = pitch_std  # None = auto-load from stats
        self._stats_file = self.data_dir / 'stat.pkl'

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
    batch_size: int = 32,  # Original PercePiano uses batch_size=32 (parser.py:107)
    max_notes: int = 1024,
    num_workers: int = 4,
    augment_train: bool = True,
    pitch_std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for VirtuosoNet features.

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data (default: True)
        pitch_std: Standard deviation for pitch normalization (for augmentation).
                   If None (default), auto-loads from stat.pkl in data_dir.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    stats_file = data_dir / 'stat.pkl'

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
