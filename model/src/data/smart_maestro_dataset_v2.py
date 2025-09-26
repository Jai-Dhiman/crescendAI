"""
SmartMAESTRODatasetV2: Canonical [time, freq] spectrograms with robust crop/pad and pad masks.

Priority-1 remediation for orientation and padding artifacts.

Usage:
    from src.data.smart_maestro_dataset_v2 import SmartMAESTRODatasetV2

    ds = SmartMAESTRODatasetV2(
        original_dir="/path/to/processed_spectrograms",
        augmented_dir="/path/to/augmented_spectrograms",
        split="train",
        use_augmentation=True,
        target_shape=(128, 128),
        random_seed=42,
    )

    specs, masks = ds.get_batch(batch_size=16, return_masks=True, shuffle=True)

Notes:
- Canonical layout is [time, freq], e.g., [T, 128].
- Padding value is -80.0 (dB). A boolean mask is returned with True for padded positions.
- Validation/test batches are deterministic by default.
- Explicit exceptions are raised on malformed inputs (user preference).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class _SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class SmartMAESTRODatasetV2:
    def __init__(
        self,
        original_dir: str,
        augmented_dir: Optional[str] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        use_augmentation: bool = True,
        target_shape: Tuple[int, int] = (128, 128),
        random_seed: int = 42,
    ) -> None:
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        if not os.path.isdir(original_dir):
            raise FileNotFoundError(f"Original spectrogram dir not found: {original_dir}")
        if augmented_dir is not None and not os.path.isdir(augmented_dir):
            raise FileNotFoundError(f"Augmented spectrogram dir not found: {augmented_dir}")

        self.original_dir = original_dir
        self.augmented_dir = augmented_dir
        self.split = split
        self.use_augmentation = use_augmentation and (split == "train") and (augmented_dir is not None)
        self.target_shape = tuple(target_shape)
        self.random_seed = random_seed

        # Seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Collect file lists
        original_files = [f for f in os.listdir(original_dir) if f.endswith("_original.npy")]
        if len(original_files) == 0:
            raise FileNotFoundError(f"No original files found in {original_dir}")

        aug_files: List[str] = []
        if augmented_dir is not None and os.path.isdir(augmented_dir):
            aug_files = [f for f in os.listdir(augmented_dir) if f.endswith(".npy")]

        # Split by originals only; augmented attaches to train originals below
        train_originals, temp_originals = train_test_split(
            original_files,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_originals, test_originals = train_test_split(
            temp_originals, test_size=(1 - val_size), random_state=random_seed
        )

        if split == "train":
            self.original_files = train_originals
            if self.use_augmentation:
                train_augmented: List[str] = []
                for orig_file in train_originals:
                    base = orig_file.replace("_original.npy", "")
                    train_augmented.extend([f for f in aug_files if f.startswith(base + "_aug_")])
                self.augmented_files = train_augmented
            else:
                self.augmented_files = []
        elif split == "val":
            self.original_files = val_originals
            self.augmented_files = []
        elif split == "test":
            self.original_files = test_originals
            self.augmented_files = []
        else:
            raise ValueError(f"Invalid split: {split}")

        self.all_files = list(self.original_files) + list(self.augmented_files)
        self.num_files = len(self.all_files)
        if self.num_files == 0:
            raise RuntimeError("No files available for the selected split")

    def __len__(self) -> int:
        return self.num_files

    def _canonicalize_time_freq(self, spec: np.ndarray, filename: str) -> np.ndarray:
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram for {filename}, got {spec.shape}")
        # Heuristic: n_mels = 128 for this project
        if spec.shape[0] == 128 and spec.shape[1] != 128:
            # [freq, time] -> [time, freq]
            spec = spec.T
        elif spec.shape[1] == 128:
            # already [time, freq]
            pass
        else:
            # Fallback: if first dim > second dim, likely [freq, time]
            if spec.shape[0] > spec.shape[1]:
                spec = spec.T
        return spec

    def _load_one(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        # Resolve path by suffix
        if filename.endswith("_original.npy"):
            path = os.path.join(self.original_dir, filename)
        else:
            if self.augmented_dir is None:
                raise FileNotFoundError("Augmented directory not provided but augmented file requested")
            path = os.path.join(self.augmented_dir, filename)

        spec = np.load(path)
        spec = self._canonicalize_time_freq(spec, filename)

        target_time, target_freq = self.target_shape
        time_len, freq_len = spec.shape

        # Build pad mask (False=valid, True=pad)
        pad_mask = np.zeros((target_time, target_freq), dtype=bool)

        # Time crop/pad
        if time_len >= target_time:
            if self.use_augmentation and self.split == "train" and time_len > target_time:
                start = random.randint(0, time_len - target_time)
            else:
                start = (time_len - target_time) // 2
            spec = spec[start : start + target_time, :]
        else:
            pad_t = target_time - time_len
            spec = np.pad(spec, ((0, pad_t), (0, 0)), mode="constant", constant_values=-80.0)
            pad_mask[time_len:, :] = True

        # Freq crop/pad
        time_len2, freq_len = spec.shape
        if freq_len >= target_freq:
            spec = spec[:, :target_freq]
        else:
            pad_f = target_freq - freq_len
            spec = np.pad(spec, ((0, 0), (0, pad_f)), mode="constant", constant_values=-80.0)
            pad_mask[:, freq_len:] = True

        if spec.shape != self.target_shape:
            raise ValueError(
                f"Post-process shape {spec.shape} != target {self.target_shape} for {filename}"
            )

        return spec.astype(np.float32), pad_mask

    def get_batch(
        self,
        batch_size: int,
        shuffle: Optional[bool] = None,
        return_masks: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        if shuffle is None:
            shuffle = (self.split == "train")

        if shuffle:
            if self.use_augmentation and len(self.augmented_files) > 0:
                n_orig = batch_size // 2
                n_aug = batch_size - n_orig
                idx_o = np.random.choice(len(self.original_files), size=n_orig, replace=True)
                idx_a = np.random.choice(len(self.augmented_files), size=n_aug, replace=True)
                files = [self.original_files[i] for i in idx_o] + [
                    self.augmented_files[i] for i in idx_a
                ]
                random.shuffle(files)
            else:
                idx = np.random.choice(self.num_files, size=batch_size, replace=True)
                files = [self.all_files[i] for i in idx]
        else:
            # Deterministic val/test sampling
            start = 0
            idx = np.arange(start, start + batch_size) % self.num_files
            files = [self.all_files[i] for i in idx]

        specs = []
        masks = []
        for f in files:
            s, m = self._load_one(f)
            specs.append(s)
            masks.append(m)

        specs_np = np.asarray(specs)
        masks_np = np.asarray(masks)
        return (specs_np, masks_np) if return_masks else specs_np
