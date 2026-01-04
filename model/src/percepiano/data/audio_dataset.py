"""
Audio Dataset for PercePiano Evaluation.

Loads pre-extracted MERT embeddings and PercePiano labels for audio-based
piano performance evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 19 PercePiano dimensions (same order as symbolic baseline)
PERCEPIANO_DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamic_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "sophistication",
    "interpretation",
]

# Dimension categories for analysis
DIMENSION_CATEGORIES = {
    "timing": ["timing"],
    "articulation": ["articulation_length", "articulation_touch"],
    "pedal": ["pedal_amount", "pedal_clarity"],
    "timbre": ["timbre_variety", "timbre_depth", "timbre_brightness", "timbre_loudness"],
    "dynamics": ["dynamic_range"],
    "tempo_space": ["tempo", "space", "balance", "drama"],
    "emotion": ["mood_valence", "mood_energy", "mood_imagination"],
    "interpretation": ["sophistication", "interpretation"],
}


class AudioPercePianoDataset(Dataset):
    """
    Dataset for audio-based PercePiano evaluation.

    Loads pre-extracted MERT embeddings and PercePiano labels.
    Supports k-fold cross-validation with the same splits as symbolic baseline.

    Attributes:
        mert_cache_dir: Directory containing cached MERT embeddings
        max_frames: Maximum frames to use (truncates longer sequences)
        samples: List of (key, labels) tuples
    """

    def __init__(
        self,
        mert_cache_dir: Path,
        label_file: Path,
        fold_assignments: Optional[Dict] = None,
        fold_id: int = 0,
        mode: str = "train",
        max_frames: int = 1000,
    ):
        """
        Initialize dataset.

        Args:
            mert_cache_dir: Directory containing cached .pt embeddings
            label_file: Path to PercePiano label JSON file
            fold_assignments: Optional dict with fold assignments
                Format: {"test": [...], "fold_0": [...], ...}
            fold_id: Fold ID for train/val split (0-3)
            mode: "train", "val", or "test"
            max_frames: Maximum MERT frames per segment (truncate if longer)
        """
        self.mert_cache_dir = Path(mert_cache_dir)
        self.max_frames = max_frames

        # Load labels
        with open(label_file) as f:
            all_labels = json.load(f)

        # Filter to segments with cached embeddings
        available_keys = {p.stem for p in self.mert_cache_dir.glob("*.pt")}
        self.samples: List[tuple] = []

        for key, label_values in all_labels.items():
            if key in available_keys:
                # Labels are [19 values, 0] - last value is unused
                labels = torch.tensor(label_values[:19], dtype=torch.float32)
                self.samples.append((key, labels))

        print(f"Total samples with embeddings: {len(self.samples)}")

        # Apply fold filtering if provided
        if fold_assignments is not None:
            self._apply_fold_filter(fold_assignments, fold_id, mode)

    def _apply_fold_filter(
        self, fold_assignments: Dict, fold_id: int, mode: str
    ) -> None:
        """Filter samples based on fold assignment."""
        if mode == "test":
            valid_keys = set(fold_assignments.get("test", []))
        elif mode == "val":
            valid_keys = set(fold_assignments.get(f"fold_{fold_id}", []))
        else:  # train
            valid_keys = set()
            for i in range(4):  # 4 folds
                if i != fold_id:
                    valid_keys.update(fold_assignments.get(f"fold_{i}", []))

        self.samples = [(k, l) for k, l in self.samples if k in valid_keys]
        print(f"{mode} samples (fold {fold_id}): {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key, labels = self.samples[idx]

        # Load cached embeddings
        embed_path = self.mert_cache_dir / f"{key}.pt"
        embeddings = torch.load(embed_path, weights_only=True)  # [T, 1024]

        # Truncate if too long
        if embeddings.shape[0] > self.max_frames:
            embeddings = embeddings[: self.max_frames]

        return {
            "embeddings": embeddings,
            "labels": labels,
            "key": key,
            "length": embeddings.shape[0],
        }


def audio_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length MERT embeddings.

    Pads embeddings to max length in batch and creates attention mask.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with padded embeddings and attention mask
    """
    embeddings = [item["embeddings"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])
    keys = [item["key"] for item in batch]

    # Pad embeddings
    padded_embeddings = pad_sequence(embeddings, batch_first=True)  # [B, T, H]

    # Create attention mask (1 for valid, 0 for padding)
    max_len = padded_embeddings.shape[1]
    attention_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "embeddings": padded_embeddings,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
        "keys": keys,
    }


class AudioPercePianoDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for audio-based PercePiano evaluation.

    Handles train/val/test data loading with k-fold cross-validation support.
    """

    def __init__(
        self,
        mert_cache_dir: Union[str, Path],
        label_file: Union[str, Path],
        fold_assignments: Dict,
        fold_id: int = 0,
        batch_size: int = 16,
        num_workers: int = 4,
        max_frames: int = 1000,
    ):
        """
        Initialize DataModule.

        Args:
            mert_cache_dir: Directory containing cached .pt embeddings
            label_file: Path to PercePiano label JSON file
            fold_assignments: Dict with fold assignments
            fold_id: Fold ID for train/val split (0-3)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            max_frames: Maximum MERT frames per segment
        """
        super().__init__()
        self.mert_cache_dir = Path(mert_cache_dir)
        self.label_file = Path(label_file)
        self.fold_assignments = fold_assignments
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames

        self.train_ds: Optional[AudioPercePianoDataset] = None
        self.val_ds: Optional[AudioPercePianoDataset] = None
        self.test_ds: Optional[AudioPercePianoDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_ds = AudioPercePianoDataset(
                mert_cache_dir=self.mert_cache_dir,
                label_file=self.label_file,
                fold_assignments=self.fold_assignments,
                fold_id=self.fold_id,
                mode="train",
                max_frames=self.max_frames,
            )

            self.val_ds = AudioPercePianoDataset(
                mert_cache_dir=self.mert_cache_dir,
                label_file=self.label_file,
                fold_assignments=self.fold_assignments,
                fold_id=self.fold_id,
                mode="val",
                max_frames=self.max_frames,
            )

        if stage == "test" or stage is None:
            self.test_ds = AudioPercePianoDataset(
                mert_cache_dir=self.mert_cache_dir,
                label_file=self.label_file,
                fold_assignments=self.fold_assignments,
                fold_id=self.fold_id,
                mode="test",
                max_frames=self.max_frames,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=audio_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=audio_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_audio_fold_assignments(
    label_file: Path,
    output_file: Path,
    n_folds: int = 4,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Create piece-based fold assignments for audio data.

    Args:
        label_file: Path to PercePiano label JSON
        output_file: Path to save fold assignments
        n_folds: Number of CV folds
        test_ratio: Fraction of samples for test set
        seed: Random seed

    Returns:
        Fold assignments dictionary
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    with open(label_file) as f:
        labels = json.load(f)

    # Group by piece
    piece_to_keys: Dict[str, List[str]] = defaultdict(list)

    for key in labels.keys():
        # Extract piece name (everything before _Xbars_)
        parts = key.split("_")
        for i, part in enumerate(parts):
            if "bars" in part:
                piece = "_".join(parts[:i])
                break
        else:
            piece = key
        piece_to_keys[piece].append(key)

    print(f"Found {len(piece_to_keys)} unique pieces")
    print(f"Total samples: {len(labels)}")

    # Shuffle pieces
    pieces = list(piece_to_keys.keys())
    random.shuffle(pieces)

    # Assign test set
    fold_assignments: Dict[str, List[str]] = {
        "test": [],
        "fold_0": [],
        "fold_1": [],
        "fold_2": [],
        "fold_3": [],
    }

    # First ~15% to test
    test_count = 0
    target_test = len(labels) * test_ratio
    test_pieces = []

    for piece in pieces:
        if test_count < target_test:
            fold_assignments["test"].extend(piece_to_keys[piece])
            test_count += len(piece_to_keys[piece])
            test_pieces.append(piece)

    # Remaining to folds (round-robin)
    remaining_pieces = [p for p in pieces if p not in test_pieces]
    for i, piece in enumerate(remaining_pieces):
        fold_idx = i % n_folds
        fold_assignments[f"fold_{fold_idx}"].extend(piece_to_keys[piece])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(fold_assignments, f, indent=2)

    print(f"Saved fold assignments to {output_file}")

    # Print statistics
    print("\nFold statistics:")
    for fold_name, keys in fold_assignments.items():
        print(f"  {fold_name}: {len(keys)} samples")

    return fold_assignments
