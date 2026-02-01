"""Pairwise ranking datasets for disentanglement experiments.

Implements datasets that sample pairs of performances from the same piece
for pairwise ranking tasks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def build_multi_performer_pieces(
    labels: Dict,
    fold_assignments: Dict,
    min_performers: int = 2,
) -> Dict[str, List[str]]:
    """Build mapping of pieces to their multiple performances.

    Extracts piece ID from recording keys (assumes format: pieceID_performerID)
    and groups recordings by piece.

    Args:
        labels: Dict mapping keys to label arrays.
        fold_assignments: Dict with fold assignments.
        min_performers: Minimum number of performers to include a piece.

    Returns:
        Dict mapping piece_id to list of recording keys.
    """
    # Get all available keys
    all_keys = set()
    for i in range(4):
        all_keys.update(fold_assignments.get(f"fold_{i}", []))

    # Group by piece ID (extract from key format)
    piece_to_recordings: Dict[str, List[str]] = {}

    for key in all_keys:
        if key not in labels:
            continue

        # Try to extract piece ID (format varies by dataset)
        # Common patterns:
        # - "composer_piece_performer" -> "composer_piece"
        # - "pieceID_performerID" -> "pieceID"
        parts = key.rsplit("_", 1)
        if len(parts) >= 2:
            piece_id = parts[0]
        else:
            piece_id = key

        if piece_id not in piece_to_recordings:
            piece_to_recordings[piece_id] = []
        piece_to_recordings[piece_id].append(key)

    # Filter to pieces with multiple performers
    multi_performer = {
        pid: recs
        for pid, recs in piece_to_recordings.items()
        if len(recs) >= min_performers
    }

    return multi_performer


def create_piece_stratified_folds(
    multi_performer_pieces: Dict[str, List[str]],
    n_folds: int = 4,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create stratified folds ensuring no piece leakage.

    All recordings of a piece go into the same fold.

    Args:
        multi_performer_pieces: Dict mapping piece_id to recording keys.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        Dict with fold_0, fold_1, etc. containing recording keys.
    """
    import numpy as np

    np.random.seed(seed)

    # Get all piece IDs and shuffle
    piece_ids = list(multi_performer_pieces.keys())
    np.random.shuffle(piece_ids)

    # Assign pieces to folds (round-robin, balanced by recording count)
    fold_sizes = [0] * n_folds
    fold_pieces: List[List[str]] = [[] for _ in range(n_folds)]

    # Sort pieces by number of recordings (largest first for better balance)
    sorted_pieces = sorted(
        piece_ids, key=lambda p: len(multi_performer_pieces[p]), reverse=True
    )

    for pid in sorted_pieces:
        # Assign to smallest fold
        smallest_fold = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_pieces[smallest_fold].append(pid)
        fold_sizes[smallest_fold] += len(multi_performer_pieces[pid])

    # Convert to recording keys
    fold_assignments = {}
    for i, pieces in enumerate(fold_pieces):
        keys = []
        for pid in pieces:
            keys.extend(multi_performer_pieces[pid])
        fold_assignments[f"fold_{i}"] = keys

    return fold_assignments


class PairwiseRankingDataset(Dataset):
    """Dataset for pairwise ranking within same-piece comparisons.

    Samples pairs (A, B) from the same piece and provides ranking targets
    based on score differences across dimensions.
    """

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        piece_to_keys: Dict[str, List[str]],
        keys: List[str],
        max_frames: int = 1000,
        ambiguous_threshold: float = 0.05,
        pairs_per_sample: int = 1,
    ):
        """Initialize pairwise dataset.

        Args:
            cache_dir: Directory with cached embeddings.
            labels: Dict mapping keys to label arrays.
            piece_to_keys: Dict mapping piece_id to list of recording keys.
            keys: List of keys to include in this split.
            max_frames: Maximum sequence length.
            ambiguous_threshold: Score difference below which pair is ambiguous.
            pairs_per_sample: Number of pairs to generate per __getitem__ call.
        """
        self.cache_dir = Path(cache_dir)
        self.labels = labels
        self.max_frames = max_frames
        self.ambiguous_threshold = ambiguous_threshold
        self.pairs_per_sample = pairs_per_sample

        # Filter to available keys in this split
        available = {p.stem for p in self.cache_dir.glob("*.pt")}
        valid_keys = set(keys) & available & set(labels.keys())

        # Build reverse mapping: key -> piece_id
        self.key_to_piece: Dict[str, str] = {}
        for pid, pkeys in piece_to_keys.items():
            for k in pkeys:
                if k in valid_keys:
                    self.key_to_piece[k] = pid

        # Group valid keys by piece
        self.piece_to_valid_keys: Dict[str, List[str]] = {}
        for k in valid_keys:
            if k not in self.key_to_piece:
                continue
            pid = self.key_to_piece[k]
            if pid not in self.piece_to_valid_keys:
                self.piece_to_valid_keys[pid] = []
            self.piece_to_valid_keys[pid].append(k)

        # Only keep pieces with 2+ recordings
        self.piece_to_valid_keys = {
            p: ks for p, ks in self.piece_to_valid_keys.items() if len(ks) >= 2
        }

        # Create all possible pairs for enumeration
        self.pairs: List[Tuple[str, str, str]] = []  # (piece_id, key_a, key_b)
        for pid, keys_list in self.piece_to_valid_keys.items():
            for i, k_a in enumerate(keys_list):
                for k_b in keys_list[i + 1 :]:
                    self.pairs.append((pid, k_a, k_b))

        # Assign unique integer IDs to pieces
        self.piece_to_id = {p: i for i, p in enumerate(self.piece_to_valid_keys.keys())}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        piece_id, key_a, key_b = self.pairs[idx]

        # Load embeddings
        emb_a = torch.load(self.cache_dir / f"{key_a}.pt", weights_only=True)
        emb_b = torch.load(self.cache_dir / f"{key_b}.pt", weights_only=True)

        # Truncate if needed
        if emb_a.shape[0] > self.max_frames:
            emb_a = emb_a[: self.max_frames]
        if emb_b.shape[0] > self.max_frames:
            emb_b = emb_b[: self.max_frames]

        # Get labels
        labels_a = torch.tensor(self.labels[key_a][:19], dtype=torch.float32)
        labels_b = torch.tensor(self.labels[key_b][:19], dtype=torch.float32)

        # Compute ranking targets
        # 1 = A > B, -1 = B > A, 0 = ambiguous
        diff = labels_a - labels_b
        targets = torch.zeros_like(diff)
        targets[diff > self.ambiguous_threshold] = 1
        targets[diff < -self.ambiguous_threshold] = -1

        return {
            "embeddings_a": emb_a,
            "embeddings_b": emb_b,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "targets": targets,
            "piece_id": self.piece_to_id[piece_id],
            "key_a": key_a,
            "key_b": key_b,
            "length_a": emb_a.shape[0],
            "length_b": emb_b.shape[0],
        }

    def get_num_pieces(self) -> int:
        """Return number of unique pieces."""
        return len(self.piece_to_id)


class DisentanglementDataset(Dataset):
    """Dataset for disentangled dual-encoder training.

    Similar to standard dataset but includes piece ID for adversarial training.
    """

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        piece_to_keys: Dict[str, List[str]],
        keys: List[str],
        max_frames: int = 1000,
    ):
        """Initialize dataset.

        Args:
            cache_dir: Directory with cached embeddings.
            labels: Dict mapping keys to label arrays.
            piece_to_keys: Dict mapping piece_id to list of recording keys.
            keys: List of keys to include in this split.
            max_frames: Maximum sequence length.
        """
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames

        # Filter to available keys
        available = {p.stem for p in self.cache_dir.glob("*.pt")}
        valid_keys = set(keys) & available & set(labels.keys())

        # Build key to piece mapping
        key_to_piece: Dict[str, str] = {}
        for pid, pkeys in piece_to_keys.items():
            for k in pkeys:
                if k in valid_keys:
                    key_to_piece[k] = pid

        # Only keep keys with known piece
        self.samples = [
            (k, torch.tensor(labels[k][:19], dtype=torch.float32), key_to_piece[k])
            for k in valid_keys
            if k in key_to_piece
        ]

        # Assign unique integer IDs to pieces
        all_pieces = sorted(set(p for _, _, p in self.samples))
        self.piece_to_id = {p: i for i, p in enumerate(all_pieces)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        key, labels, piece_id = self.samples[idx]

        emb = torch.load(self.cache_dir / f"{key}.pt", weights_only=True)
        if emb.shape[0] > self.max_frames:
            emb = emb[: self.max_frames]

        return {
            "embeddings": emb,
            "labels": labels,
            "piece_id": self.piece_to_id[piece_id],
            "key": key,
            "length": emb.shape[0],
        }

    def get_num_pieces(self) -> int:
        """Return number of unique pieces."""
        return len(self.piece_to_id)


def pairwise_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for pairwise ranking dataset.

    Pads A and B embeddings separately and creates attention masks.
    """
    embs_a = [b["embeddings_a"] for b in batch]
    embs_b = [b["embeddings_b"] for b in batch]
    labels_a = torch.stack([b["labels_a"] for b in batch])
    labels_b = torch.stack([b["labels_b"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    piece_ids = torch.tensor([b["piece_id"] for b in batch], dtype=torch.long)
    lengths_a = torch.tensor([b["length_a"] for b in batch])
    lengths_b = torch.tensor([b["length_b"] for b in batch])

    # Pad embeddings
    padded_a = pad_sequence(embs_a, batch_first=True)
    padded_b = pad_sequence(embs_b, batch_first=True)

    # Create attention masks
    mask_a = torch.arange(padded_a.shape[1]).unsqueeze(0) < lengths_a.unsqueeze(1)
    mask_b = torch.arange(padded_b.shape[1]).unsqueeze(0) < lengths_b.unsqueeze(1)

    return {
        "embeddings_a": padded_a,
        "embeddings_b": padded_b,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "targets": targets,
        "piece_ids": piece_ids,
        "keys_a": [b["key_a"] for b in batch],
        "keys_b": [b["key_b"] for b in batch],
        "lengths_a": lengths_a,
        "lengths_b": lengths_b,
    }


def disentanglement_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for disentanglement dataset."""
    embs = [b["embeddings"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    piece_ids = torch.tensor([b["piece_id"] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b["length"] for b in batch])

    padded = pad_sequence(embs, batch_first=True)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "embeddings": padded,
        "attention_mask": mask,
        "labels": labels,
        "piece_ids": piece_ids,
        "keys": [b["key"] for b in batch],
        "lengths": lengths,
    }


class HardPairRankingDataset(PairwiseRankingDataset):
    """Dataset that focuses on hard pairs with moderate score differences.

    Hard pairs are more challenging for the model to distinguish and
    can improve generalization when used for training.
    """

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        piece_to_keys: Dict[str, List[str]],
        keys: List[str],
        max_frames: int = 1000,
        ambiguous_threshold: float = 0.05,
        pairs_per_sample: int = 1,
        min_diff: float = 0.05,
        max_diff: float = 0.20,
    ):
        """Initialize hard pair dataset.

        Args:
            cache_dir: Directory with cached embeddings.
            labels: Dict mapping keys to label arrays.
            piece_to_keys: Dict mapping piece_id to list of recording keys.
            keys: List of keys to include in this split.
            max_frames: Maximum sequence length.
            ambiguous_threshold: Score difference below which pair is ambiguous.
            pairs_per_sample: Number of pairs to generate per __getitem__ call.
            min_diff: Minimum mean score difference for hard pairs.
            max_diff: Maximum mean score difference for hard pairs.
        """
        super().__init__(
            cache_dir=cache_dir,
            labels=labels,
            piece_to_keys=piece_to_keys,
            keys=keys,
            max_frames=max_frames,
            ambiguous_threshold=ambiguous_threshold,
            pairs_per_sample=pairs_per_sample,
        )

        self.min_diff = min_diff
        self.max_diff = max_diff

        # Filter pairs to hard range based on mean score difference
        original_count = len(self.pairs)
        hard_pairs = []

        for pid, k_a, k_b in self.pairs:
            if k_a in self.labels and k_b in self.labels:
                label_a = self.labels[k_a][:19]
                label_b = self.labels[k_b][:19]
                mean_diff = sum(abs(a - b) for a, b in zip(label_a, label_b)) / 19

                if self.min_diff <= mean_diff <= self.max_diff:
                    hard_pairs.append((pid, k_a, k_b))

        self.pairs = hard_pairs
        print(f"HardPairRankingDataset: filtered {original_count} -> {len(self.pairs)} pairs "
              f"(diff range: {min_diff:.2f}-{max_diff:.2f})")


class TripletRankingDataset(Dataset):
    """Dataset for triplet-based ranking within same-piece comparisons.

    Samples triplets (anchor, positive, negative) from the same piece where:
    - Positive has higher mean quality score than anchor
    - Negative has lower mean quality score than anchor

    This enables learning quality differences while controlling for piece.
    """

    def __init__(
        self,
        cache_dir: Path,
        labels: Dict,
        piece_to_keys: Dict[str, List[str]],
        keys: List[str],
        max_frames: int = 1000,
        min_score_diff: float = 0.05,
    ):
        """Initialize triplet dataset.

        Args:
            cache_dir: Directory with cached embeddings.
            labels: Dict mapping keys to label arrays.
            piece_to_keys: Dict mapping piece_id to list of recording keys.
            keys: List of keys to include in this split.
            max_frames: Maximum sequence length.
            min_score_diff: Minimum mean score difference for triplet formation.
        """
        self.cache_dir = Path(cache_dir)
        self.labels = labels
        self.max_frames = max_frames
        self.min_score_diff = min_score_diff

        # Filter to available keys in this split
        available = {p.stem for p in self.cache_dir.glob("*.pt")}
        valid_keys = set(keys) & available & set(labels.keys())

        # Build reverse mapping: key -> piece_id
        self.key_to_piece: Dict[str, str] = {}
        for pid, pkeys in piece_to_keys.items():
            for k in pkeys:
                if k in valid_keys:
                    self.key_to_piece[k] = pid

        # Group valid keys by piece
        self.piece_to_valid_keys: Dict[str, List[str]] = {}
        for k in valid_keys:
            if k not in self.key_to_piece:
                continue
            pid = self.key_to_piece[k]
            if pid not in self.piece_to_valid_keys:
                self.piece_to_valid_keys[pid] = []
            self.piece_to_valid_keys[pid].append(k)

        # Only keep pieces with 3+ recordings (needed for triplets)
        self.piece_to_valid_keys = {
            p: ks for p, ks in self.piece_to_valid_keys.items() if len(ks) >= 3
        }

        # Pre-compute mean scores for sorting
        self.key_to_mean_score: Dict[str, float] = {}
        for key in valid_keys:
            if key in labels:
                self.key_to_mean_score[key] = sum(labels[key][:19]) / 19

        # Sort keys within each piece by mean score
        for pid in self.piece_to_valid_keys:
            self.piece_to_valid_keys[pid].sort(
                key=lambda k: self.key_to_mean_score.get(k, 0)
            )

        # Generate all valid triplets
        self.triplets: List[Tuple[str, str, str, str]] = []  # (pid, anchor, pos, neg)
        for pid, keys_list in self.piece_to_valid_keys.items():
            n = len(keys_list)
            for i in range(n):
                anchor_key = keys_list[i]
                anchor_score = self.key_to_mean_score.get(anchor_key, 0)

                # Positives: keys with higher scores
                positives = [
                    k for k in keys_list[i + 1 :]
                    if self.key_to_mean_score.get(k, 0) - anchor_score >= min_score_diff
                ]

                # Negatives: keys with lower scores
                negatives = [
                    k for k in keys_list[:i]
                    if anchor_score - self.key_to_mean_score.get(k, 0) >= min_score_diff
                ]

                # Generate triplets
                for pos_key in positives:
                    for neg_key in negatives:
                        self.triplets.append((pid, anchor_key, pos_key, neg_key))

        # Assign unique integer IDs to pieces
        self.piece_to_id = {p: i for i, p in enumerate(self.piece_to_valid_keys.keys())}

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict:
        piece_id, anchor_key, pos_key, neg_key = self.triplets[idx]

        # Load embeddings
        emb_anchor = torch.load(self.cache_dir / f"{anchor_key}.pt", weights_only=True)
        emb_pos = torch.load(self.cache_dir / f"{pos_key}.pt", weights_only=True)
        emb_neg = torch.load(self.cache_dir / f"{neg_key}.pt", weights_only=True)

        # Truncate if needed
        if emb_anchor.shape[0] > self.max_frames:
            emb_anchor = emb_anchor[: self.max_frames]
        if emb_pos.shape[0] > self.max_frames:
            emb_pos = emb_pos[: self.max_frames]
        if emb_neg.shape[0] > self.max_frames:
            emb_neg = emb_neg[: self.max_frames]

        # Get labels
        labels_anchor = torch.tensor(self.labels[anchor_key][:19], dtype=torch.float32)
        labels_pos = torch.tensor(self.labels[pos_key][:19], dtype=torch.float32)
        labels_neg = torch.tensor(self.labels[neg_key][:19], dtype=torch.float32)

        return {
            "embeddings_anchor": emb_anchor,
            "embeddings_positive": emb_pos,
            "embeddings_negative": emb_neg,
            "labels_anchor": labels_anchor,
            "labels_positive": labels_pos,
            "labels_negative": labels_neg,
            "piece_id": self.piece_to_id[piece_id],
            "key_anchor": anchor_key,
            "key_positive": pos_key,
            "key_negative": neg_key,
            "length_anchor": emb_anchor.shape[0],
            "length_positive": emb_pos.shape[0],
            "length_negative": emb_neg.shape[0],
        }

    def get_num_pieces(self) -> int:
        """Return number of unique pieces."""
        return len(self.piece_to_id)


def triplet_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for triplet ranking dataset.

    Pads anchor, positive, and negative embeddings separately.
    """
    embs_anchor = [b["embeddings_anchor"] for b in batch]
    embs_pos = [b["embeddings_positive"] for b in batch]
    embs_neg = [b["embeddings_negative"] for b in batch]

    labels_anchor = torch.stack([b["labels_anchor"] for b in batch])
    labels_pos = torch.stack([b["labels_positive"] for b in batch])
    labels_neg = torch.stack([b["labels_negative"] for b in batch])

    piece_ids = torch.tensor([b["piece_id"] for b in batch], dtype=torch.long)

    lengths_anchor = torch.tensor([b["length_anchor"] for b in batch])
    lengths_pos = torch.tensor([b["length_positive"] for b in batch])
    lengths_neg = torch.tensor([b["length_negative"] for b in batch])

    # Pad embeddings
    padded_anchor = pad_sequence(embs_anchor, batch_first=True)
    padded_pos = pad_sequence(embs_pos, batch_first=True)
    padded_neg = pad_sequence(embs_neg, batch_first=True)

    # Create attention masks
    mask_anchor = torch.arange(padded_anchor.shape[1]).unsqueeze(0) < lengths_anchor.unsqueeze(1)
    mask_pos = torch.arange(padded_pos.shape[1]).unsqueeze(0) < lengths_pos.unsqueeze(1)
    mask_neg = torch.arange(padded_neg.shape[1]).unsqueeze(0) < lengths_neg.unsqueeze(1)

    return {
        "embeddings_anchor": padded_anchor,
        "embeddings_positive": padded_pos,
        "embeddings_negative": padded_neg,
        "mask_anchor": mask_anchor,
        "mask_positive": mask_pos,
        "mask_negative": mask_neg,
        "labels_anchor": labels_anchor,
        "labels_positive": labels_pos,
        "labels_negative": labels_neg,
        "piece_ids": piece_ids,
        "keys_anchor": [b["key_anchor"] for b in batch],
        "keys_positive": [b["key_positive"] for b in batch],
        "keys_negative": [b["key_negative"] for b in batch],
    }
