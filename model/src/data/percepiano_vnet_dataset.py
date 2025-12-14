"""
PyTorch Dataset for PercePiano with VirtuosoNet features.

This module provides a Dataset class that loads preprocessed VirtuosoNet
features (78-dim) for training the PercePiano replica model.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class PercePianoVNetDataset(Dataset):
    """
    PyTorch Dataset for PercePiano with VirtuosoNet 78-dim features.

    Loads preprocessed pickle files containing:
    - input: (num_notes, 78) VirtuosoNet features
    - note_location: dict with 'beat', 'measure', 'voice' arrays
    - labels: (19,) PercePiano scores

    Args:
        data_dir: Path to preprocessed data directory (e.g., data/processed/percepiano_vnet/train)
        max_notes: Maximum number of notes per sample (for padding/truncation)
        transform: Optional transform to apply to features
        augment: Whether to apply key augmentation (random pitch shifts)
        pitch_std: Standard deviation used for midi_pitch normalization (for augmentation)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_notes: int = 1024,
        transform: Optional[callable] = None,
        augment: bool = False,
        pitch_std: float = 12.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_notes = max_notes
        self.transform = transform
        self.augment = augment
        self.pitch_std = pitch_std

        # Find all pickle files
        self.pkl_files = sorted(self.data_dir.glob("*.pkl"))
        if not self.pkl_files:
            raise ValueError(f"No pickle files found in {self.data_dir}")

        print(f"Found {len(self.pkl_files)} samples in {self.data_dir}")
        if augment:
            print(f"  Key augmentation ENABLED (pitch_std={pitch_std})")

    def __len__(self) -> int:
        return len(self.pkl_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample.

        Returns:
            Dictionary with:
            - input_features: (max_notes, 78) float tensor
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

        Args:
            input_features: (num_notes, 78) feature array
            num_notes: Actual number of notes (before padding)

        Returns:
            Augmented feature array
        """
        # Feature indices based on VNET_INPUT_KEYS order:
        # midi_pitch is at index 0 (normalized scalar)
        # pitch vector is at index 14: [octave (1), pitch_class_one_hot (12)]
        MIDI_PITCH_IDX = 0
        PITCH_VEC_START = 14  # After 14 scalar features
        PITCH_CLASS_START = 15  # octave at 14, pitch class starts at 15
        PITCH_CLASS_END = 27  # 12 pitch classes

        # Get current pitch range from midi_pitch feature
        # Note: midi_pitch is z-score normalized, so we need to work with relative shifts
        pitches_normalized = input_features[:num_notes, MIDI_PITCH_IDX]

        # Estimate original pitch range (assuming std=12 from normalization)
        # For piano: MIDI 21 (A0) to 108 (C8)
        # We limit shifts to stay within piano range
        pitch_mean = pitches_normalized.mean() * self.pitch_std
        estimated_max = pitch_mean + pitches_normalized.max() * self.pitch_std
        estimated_min = pitch_mean + pitches_normalized.min() * self.pitch_std

        # Calculate safe shift range
        max_up = min(108 - estimated_max, 7)  # Max 7 semitones up
        max_down = min(estimated_min - 21, 5)  # Max 5 semitones down

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
        pitch_std: Standard deviation for pitch normalization (for augmentation)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        max_notes: int = 1024,
        num_workers: int = 4,
        augment_train: bool = True,
        pitch_std: float = 12.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_notes = max_notes
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.pitch_std = pitch_std

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
                )
            if val_dir.exists():
                # Validation dataset without augmentation
                self.val_dataset = PercePianoVNetDataset(
                    val_dir, max_notes=self.max_notes
                )

        if stage == "test" or stage is None:
            test_dir = self.data_dir / "test"
            if test_dir.exists():
                self.test_dataset = PercePianoVNetDataset(
                    test_dir, max_notes=self.max_notes
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
    batch_size: int = 8,
    max_notes: int = 1024,
    num_workers: int = 4,
    augment_train: bool = True,
    pitch_std: float = 12.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for VirtuosoNet features.

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size
        max_notes: Maximum notes per sample
        num_workers: Number of dataloader workers
        augment_train: Whether to apply key augmentation to training data (default: True)
        pitch_std: Standard deviation for pitch normalization (for augmentation)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Training dataset with augmentation enabled by default
    train_dataset = PercePianoVNetDataset(
        data_dir / "train",
        max_notes=max_notes,
        augment=augment_train,
        pitch_std=pitch_std,
    )
    # Validation and test datasets without augmentation
    val_dataset = PercePianoVNetDataset(data_dir / "val", max_notes=max_notes)
    test_dataset = PercePianoVNetDataset(data_dir / "test", max_notes=max_notes)

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
