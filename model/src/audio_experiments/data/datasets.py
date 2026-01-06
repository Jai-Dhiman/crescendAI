"""Dataset classes for MERT, Mel, and Statistics features."""

from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MERTDataset(Dataset):
    """Dataset for MERT embeddings."""

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        fold_assignments: Dict,
        fold_id: int,
        mode: str,
        max_frames: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames

        available = {p.stem for p in self.cache_dir.glob("*.pt")}

        if mode == "test":
            valid_keys = set(fold_assignments.get("test", []))
        elif mode == "val":
            valid_keys = set(fold_assignments.get(f"fold_{fold_id}", []))
        else:  # train
            valid_keys = set()
            for i in range(4):
                if i != fold_id:
                    valid_keys.update(fold_assignments.get(f"fold_{i}", []))

        self.samples = [
            (k, torch.tensor(labels[k][:19], dtype=torch.float32))
            for k in valid_keys
            if k in available and k in labels
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]
        emb = torch.load(self.cache_dir / f"{key}.pt", weights_only=True)
        if emb.shape[0] > self.max_frames:
            emb = emb[: self.max_frames]
        return {
            "embeddings": emb,
            "labels": label,
            "key": key,
            "length": emb.shape[0],
        }


class MelDataset(Dataset):
    """Dataset for mel spectrograms."""

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        fold_assignments: Dict,
        fold_id: int,
        mode: str,
        max_frames: int = 2000,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames

        available = {p.stem for p in self.cache_dir.glob("*.pt")}

        if mode == "test":
            valid_keys = set(fold_assignments.get("test", []))
        elif mode == "val":
            valid_keys = set(fold_assignments.get(f"fold_{fold_id}", []))
        else:
            valid_keys = set()
            for i in range(4):
                if i != fold_id:
                    valid_keys.update(fold_assignments.get(f"fold_{i}", []))

        self.samples = [
            (k, torch.tensor(labels[k][:19], dtype=torch.float32))
            for k in valid_keys
            if k in available and k in labels
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]
        mel = torch.load(self.cache_dir / f"{key}.pt", weights_only=True)  # [128, T]
        if mel.shape[1] > self.max_frames:
            mel = mel[:, : self.max_frames]
        return {
            "mel": mel,
            "labels": label,
            "key": key,
            "length": mel.shape[1],
        }


class StatsDataset(Dataset):
    """Dataset for audio statistics."""

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        fold_assignments: Dict,
        fold_id: int,
        mode: str,
    ):
        self.cache_dir = Path(cache_dir)

        available = {p.stem for p in self.cache_dir.glob("*.pt")}

        if mode == "test":
            valid_keys = set(fold_assignments.get("test", []))
        elif mode == "val":
            valid_keys = set(fold_assignments.get(f"fold_{fold_id}", []))
        else:
            valid_keys = set()
            for i in range(4):
                if i != fold_id:
                    valid_keys.update(fold_assignments.get(f"fold_{i}", []))

        self.samples = [
            (k, torch.tensor(labels[k][:19], dtype=torch.float32))
            for k in valid_keys
            if k in available and k in labels
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]
        stats = torch.load(self.cache_dir / f"{key}.pt", weights_only=True)
        return {"features": stats, "labels": label, "key": key}


def mert_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for MERT embeddings with padding."""
    embs = [b["embeddings"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch])
    padded = pad_sequence(embs, batch_first=True)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "embeddings": padded,
        "attention_mask": mask,
        "labels": labels,
        "keys": [b["key"] for b in batch],
        "lengths": lengths,
    }


def mel_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for mel spectrograms with padding."""
    mels = [b["mel"] for b in batch]  # Each is [128, T]
    labels = torch.stack([b["labels"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch])
    max_len = max(m.shape[1] for m in mels)
    padded = torch.zeros(len(mels), 128, max_len)
    for i, m in enumerate(mels):
        padded[i, :, : m.shape[1]] = m
    return {
        "mel": padded,
        "labels": labels,
        "keys": [b["key"] for b in batch],
        "lengths": lengths,
    }


def stats_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for audio statistics."""
    features = torch.stack([b["features"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "features": features,
        "labels": labels,
        "keys": [b["key"] for b in batch],
    }
