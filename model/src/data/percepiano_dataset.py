"""
PercePiano Dataset for MIDI-only piano performance evaluation.

Loads PercePiano MIDI files with expert annotations for training
the MIDI-only scoring model.
"""

import json
import numpy as np
import pretty_midi
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from .midi_processing import OctupleMIDITokenizer


# All 19 PercePiano dimensions (matching reference implementation)
DIMENSIONS = [
    "timing",              # 0: Stable <-> Unstable
    "articulation_length", # 1: Short <-> Long
    "articulation_touch",  # 2: Soft/Cushioned <-> Hard/Solid
    "pedal_amount",        # 3: Sparse/Dry <-> Saturated/Wet
    "pedal_clarity",       # 4: Clean <-> Blurred
    "timbre_variety",      # 5: Even <-> Colorful
    "timbre_depth",        # 6: Shallow <-> Rich
    "timbre_brightness",   # 7: Bright <-> Dark
    "timbre_loudness",     # 8: Soft <-> Loud
    "dynamic_range",       # 9: Little Range <-> Large Range
    "tempo",               # 10: Fast-paced <-> Slow-paced
    "space",               # 11: Flat <-> Spacious
    "balance",             # 12: Disproportioned <-> Balanced
    "drama",               # 13: Pure <-> Dramatic
    "mood_valence",        # 14: Optimistic <-> Dark
    "mood_energy",         # 15: Low Energy <-> High Energy
    "mood_imagination",    # 16: Honest <-> Imaginative
    "sophistication",      # 17: Sophisticated/Mellow <-> Raw/Crude
    "interpretation",      # 18: Unsatisfactory <-> Convincing
]


class PercePianoDataset(Dataset):
    """
    PyTorch Dataset for PercePiano MIDI performance evaluation.

    Loads MIDI files and their corresponding expert annotations,
    tokenizes MIDI using OctupleMIDI format, and returns tensors
    suitable for training.
    """

    def __init__(
        self,
        data_file: Path,
        max_seq_length: int = 1024,
        segment_seconds: float = 30.0,
        augment: bool = False,
        cache_midi: bool = True,
    ):
        """
        Args:
            data_file: Path to JSON file with sample list (from prepare_percepiano.py)
            max_seq_length: Maximum number of MIDI tokens per sample
            segment_seconds: Target segment length in seconds
            augment: Whether to apply data augmentation
            cache_midi: Whether to cache loaded MIDI in memory
        """
        self.data_file = Path(data_file)
        self.max_seq_length = max_seq_length
        self.segment_seconds = segment_seconds
        self.augment = augment
        self.cache_midi = cache_midi

        # Load sample list
        with open(self.data_file, "r") as f:
            self.samples = json.load(f)

        # Initialize tokenizer
        self.tokenizer = OctupleMIDITokenizer()

        # Cache for loaded MIDI
        self._midi_cache: Dict[str, np.ndarray] = {}

        # Dimension names for score ordering
        self.dimensions = DIMENSIONS

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load and tokenize MIDI
        midi_tokens = self._load_midi(sample["midi_path"])

        # Truncate or pad to max_seq_length
        midi_tokens = self._prepare_tokens(midi_tokens)

        # Get scores as tensor (ordered by DIMENSIONS)
        scores = torch.tensor(
            [sample["scores"][dim] for dim in self.dimensions],
            dtype=torch.float32,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        # Padding is all zeros, real tokens have non-zero pitch (column 3) or velocity (column 5)
        attention_mask = ((midi_tokens[:, 3] != 0) | (midi_tokens[:, 5] != 0)).float()

        return {
            "midi_tokens": midi_tokens,
            "attention_mask": attention_mask,
            "scores": scores,
            "name": sample["name"],
        }

    def _load_midi(self, midi_path: str) -> np.ndarray:
        """Load and tokenize a MIDI file."""
        # Check cache first
        if self.cache_midi and midi_path in self._midi_cache:
            tokens = self._midi_cache[midi_path].copy()
        else:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Tokenize
            tokens = self.tokenizer.encode(midi_data)

            # Cache if enabled
            if self.cache_midi:
                self._midi_cache[midi_path] = tokens.copy()

        # Apply augmentation if enabled
        if self.augment:
            tokens = self._augment_midi(tokens)

        return tokens

    def _prepare_tokens(self, tokens: np.ndarray) -> torch.Tensor:
        """Prepare tokens: truncate or pad to max_seq_length."""
        seq_len = len(tokens)

        if seq_len > self.max_seq_length:
            # Random crop for training, center crop for eval
            if self.augment:
                start = np.random.randint(0, seq_len - self.max_seq_length)
            else:
                start = (seq_len - self.max_seq_length) // 2
            tokens = tokens[start : start + self.max_seq_length]
        elif seq_len < self.max_seq_length:
            # Pad with zeros
            padding = np.zeros((self.max_seq_length - seq_len, 8), dtype=np.int32)
            tokens = np.concatenate([tokens, padding], axis=0)

        return torch.tensor(tokens, dtype=torch.long)

    def _augment_midi(self, tokens: np.ndarray) -> np.ndarray:
        """Apply data augmentation to MIDI tokens."""
        tokens = tokens.copy()

        # Pitch shift (+/- 2 semitones)
        if np.random.random() < 0.5:
            pitch_shift = np.random.randint(-2, 3)
            tokens[:, 3] = np.clip(tokens[:, 3] + pitch_shift, 0, 87)

        # Velocity scaling (0.8x to 1.2x)
        if np.random.random() < 0.5:
            vel_scale = np.random.uniform(0.8, 1.2)
            tokens[:, 5] = np.clip(tokens[:, 5] * vel_scale, 0, 127).astype(np.int32)

        # Time jitter (small random offset to timing)
        if np.random.random() < 0.3:
            time_jitter = np.random.randint(-1, 2, size=len(tokens))
            tokens[:, 2] = np.clip(tokens[:, 2] + time_jitter, 0, 15)

        return tokens

    def get_dimension_names(self) -> List[str]:
        """Return ordered list of dimension names."""
        return self.dimensions.copy()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute dataset statistics for each dimension."""
        stats = {}
        for dim in self.dimensions:
            values = [s["scores"][dim] for s in self.samples]
            stats[dim] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        return stats


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    max_seq_length: int = 1024,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing processed PercePiano JSON files
        batch_size: Batch size for training
        max_seq_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PercePianoDataset(
        data_file=data_dir / "percepiano_train.json",
        max_seq_length=max_seq_length,
        augment=True,
        cache_midi=True,
    )

    val_dataset = PercePianoDataset(
        data_file=data_dir / "percepiano_val.json",
        max_seq_length=max_seq_length,
        augment=False,
        cache_midi=True,
    )

    test_dataset = PercePianoDataset(
        data_file=data_dir / "percepiano_test.json",
        max_seq_length=max_seq_length,
        augment=False,
        cache_midi=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = PercePianoDataset(
        data_file=args.data_dir / "percepiano_train.json",
        max_seq_length=512,
        augment=False,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Dimensions: {dataset.get_dimension_names()}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"MIDI tokens shape: {sample['midi_tokens'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Scores shape: {sample['scores'].shape}")
    print(f"Scores: {sample['scores']}")

    # Print statistics
    print("\nDataset statistics:")
    stats = dataset.get_stats()
    for dim, dim_stats in stats.items():
        print(f"  {dim}: mean={dim_stats['mean']:.1f}, std={dim_stats['std']:.1f}")
