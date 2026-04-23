"""Dataset classes for T2-T4 data tiers: competition, paired, augmented, pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import random as _random

from torch.utils.data import BatchSampler, Dataset, Sampler


class CompetitionDataset(Dataset):
    """T2: Competition recordings with ordinal placement labels.

    Takes a list of dicts with keys: recording_id, placement, embeddings.
    Each embeddings is a [T, D] tensor of frame-level features.
    Truncates/pads to max_frames.
    """

    def __init__(self, data: list[dict], max_frames: int = 1000):
        if not data:
            raise ValueError("data must be a non-empty list of recording dicts")
        self.data = data
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        record = self.data[idx]
        embeddings = record["embeddings"]  # [T, D]
        placement = record["placement"]
        recording_id = record["recording_id"]

        T, D = embeddings.shape

        # Truncate if longer than max_frames
        if T > self.max_frames:
            embeddings = embeddings[: self.max_frames]
            T = self.max_frames

        # Create mask for valid frames before padding
        mask = torch.ones(self.max_frames, dtype=torch.bool)

        # Pad if shorter than max_frames
        if T < self.max_frames:
            padding = torch.zeros(self.max_frames - T, D, dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, padding], dim=0)
            mask[T:] = False

        return {
            "embeddings": embeddings,
            "mask": mask,
            "placement": placement,
            "recording_id": recording_id,
        }


class CompetitionPairSampler:
    """Generate ordinal ranking pairs from competition recordings.

    Within-piece pairs: two performers playing the same piece,
    ranked by placement (lower placement = better).

    Cross-round pairs: same performer in different rounds provides
    implicit quality signal (later rounds = higher bar cleared).
    """

    def __init__(self, records: list[dict]):
        if not records:
            raise ValueError("records must be a non-empty list")

        self.records = records

        # Index by recording_id for fast lookup
        self._by_id: Dict[str, dict] = {r["recording_id"]: r for r in records}

        # Group by (competition, edition, piece) for within-piece pairs
        self._piece_groups: Dict[Tuple[str, int, str], List[dict]] = {}
        for r in records:
            key = (r["competition"], r["edition"], r["piece"])
            if key not in self._piece_groups:
                self._piece_groups[key] = []
            self._piece_groups[key].append(r)

        # Group by (competition, edition, performer) for cross-round pairs
        self._performer_groups: Dict[Tuple[str, int, str], List[dict]] = {}
        for r in records:
            key = (r["competition"], r["edition"], r["performer"])
            if key not in self._performer_groups:
                self._performer_groups[key] = []
            self._performer_groups[key].append(r)

        # Pre-compute all within-piece pairs
        self._within_piece_pairs: List[Tuple[str, str, float]] = []
        for group in self._piece_groups.values():
            if len(group) < 2:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    pa, pb = a["placement"], b["placement"]
                    if pa == pb:
                        continue  # skip ties
                    margin = abs(pa - pb)
                    if pa < pb:
                        self._within_piece_pairs.append(
                            (a["recording_id"], b["recording_id"], float(margin))
                        )
                    else:
                        self._within_piece_pairs.append(
                            (b["recording_id"], a["recording_id"], float(margin))
                        )

        # Pre-compute cross-round pairs (same performer, different rounds)
        round_order = {
            "preliminary": 0, "stage1": 1, "stage2": 2,
            "stage3": 3, "final": 4,
        }
        self._cross_round_pairs: List[Tuple[str, str, float]] = []
        for group in self._performer_groups.values():
            if len(group) < 2:
                continue
            sorted_group = sorted(
                group, key=lambda r: round_order.get(r["round"], -1)
            )
            for i, a in enumerate(sorted_group):
                for b in sorted_group[i + 1:]:
                    ra = round_order.get(a["round"], -1)
                    rb = round_order.get(b["round"], -1)
                    if ra == rb:
                        continue
                    # Later round = better (or at least cleared bar)
                    margin = float(rb - ra)
                    self._cross_round_pairs.append(
                        (b["recording_id"], a["recording_id"], margin)
                    )

    @property
    def n_within_piece_pairs(self) -> int:
        return len(self._within_piece_pairs)

    @property
    def n_cross_round_pairs(self) -> int:
        return len(self._cross_round_pairs)

    def sample_pairs(
        self,
        n_pairs: int,
        within_piece_weight: float = 0.8,
    ) -> List[Tuple[str, str, float]]:
        """Return (better_id, worse_id, margin) tuples.

        Args:
            n_pairs: Number of pairs to sample.
            within_piece_weight: Probability of sampling a within-piece pair
                vs cross-round pair. Within-piece pairs have stronger signal.

        Returns:
            List of (better_recording_id, worse_recording_id, margin).
        """
        import random

        pairs: List[Tuple[str, str, float]] = []

        for _ in range(n_pairs):
            if (
                random.random() < within_piece_weight
                and self._within_piece_pairs
            ) or not self._cross_round_pairs:
                if self._within_piece_pairs:
                    pairs.append(random.choice(self._within_piece_pairs))
            else:
                if self._cross_round_pairs:
                    pairs.append(random.choice(self._cross_round_pairs))

        return pairs

    def all_within_piece_pairs(self) -> List[Tuple[str, str, float]]:
        """Return all within-piece pairs (deterministic)."""
        return list(self._within_piece_pairs)

    def all_cross_round_pairs(self) -> List[Tuple[str, str, float]]:
        """Return all cross-round pairs (deterministic)."""
        return list(self._cross_round_pairs)


class PairedPerformanceDataset(Dataset):
    """T3: Extended PairwiseRankingDataset supporting ATEPP + competition data.

    Takes labels dict (key -> 19-dim scores), piece_to_keys mapping, and list of keys.
    Generates all valid pairs within each piece.
    Does NOT load embeddings from disk - just generates pair indices and labels.
    """

    def __init__(
        self,
        cache_dir: Path,
        labels: dict,
        piece_to_keys: dict,
        keys: list[str],
        embedding_keys: set | None = None,
        tier: str = "percepiano",
    ):
        self.cache_dir = Path(cache_dir)
        self.labels = labels
        self.tier = tier

        # Filter to keys that exist in the fold, labels, and (if provided) embeddings
        valid_keys = set(keys) & set(labels.keys())
        if embedding_keys is not None:
            valid_keys &= embedding_keys

        # Build reverse mapping: key -> piece_id
        key_to_piece: Dict[str, str] = {}
        for pid, pkeys in piece_to_keys.items():
            for k in pkeys:
                if k in valid_keys:
                    key_to_piece[k] = pid

        # Group valid keys by piece
        piece_to_valid_keys: Dict[str, List[str]] = {}
        for k in valid_keys:
            if k not in key_to_piece:
                continue
            pid = key_to_piece[k]
            if pid not in piece_to_valid_keys:
                piece_to_valid_keys[pid] = []
            piece_to_valid_keys[pid].append(k)

        # Only keep pieces with 2+ recordings
        piece_to_valid_keys = {
            p: sorted(ks)
            for p, ks in piece_to_valid_keys.items()
            if len(ks) >= 2
        }

        # Pre-compute all valid pairs (within same piece)
        self.pairs: List[Tuple[str, str, str]] = []  # (key_a, key_b, piece_id)
        for pid, keys_list in piece_to_valid_keys.items():
            for i, k_a in enumerate(keys_list):
                for k_b in keys_list[i + 1 :]:
                    self.pairs.append((k_a, k_b, pid))

        # Assign unique integer IDs to pieces
        all_pieces = sorted(piece_to_valid_keys.keys())
        self.piece_to_id = {p: i for i, p in enumerate(all_pieces)}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        key_a, key_b, piece_id = self.pairs[idx]

        labels_a = torch.tensor(self.labels[key_a], dtype=torch.float32)
        labels_b = torch.tensor(self.labels[key_b], dtype=torch.float32)

        return {
            "key_a": key_a,
            "key_b": key_b,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "piece_id": self.piece_to_id[piece_id],
            "tier": self.tier,
        }


class HardNegativePairSampler:
    """Curriculum-based pair sampler that progressively adds harder pairs.

    During warmup, returns all pairs uniformly. After warmup, filters by
    label difficulty: starts with easy pairs (large label gap), progressively
    adds harder pairs (smaller gap) as training continues.

    Optionally accepts model prediction errors to oversample pairs where the
    model is wrong or uncertain.
    """

    def __init__(
        self,
        dataset: PairedPerformanceDataset,
        warmup_epochs: int = 5,
        easy_threshold: float = 0.3,
        hard_oversample: float = 2.0,
    ):
        self.dataset = dataset
        self.warmup_epochs = warmup_epochs
        self.easy_threshold = easy_threshold
        self.hard_oversample = hard_oversample

        # Pre-compute mean absolute label differences per pair
        self.pair_diffs = []
        for key_a, key_b, _pid in dataset.pairs:
            labels_a = np.array(dataset.labels[key_a])
            labels_b = np.array(dataset.labels[key_b])
            mean_diff = float(np.abs(labels_a - labels_b).mean())
            self.pair_diffs.append(mean_diff)
        self.pair_diffs = np.array(self.pair_diffs)

        self._model_errors: np.ndarray | None = None

    def update_model_errors(self, errors: np.ndarray) -> None:
        """Update pair-level model errors for hard negative mining.

        Args:
            errors: Array of shape (n_pairs,) with error magnitude per pair.
                Higher values = model struggled more on this pair.
        """
        self._model_errors = errors

    def get_indices(self, epoch: int) -> list[int]:
        """Get pair indices for the given epoch.

        During warmup: all indices shuffled.
        After warmup: curriculum + optional hard negative oversampling.
        """
        n = len(self.dataset)
        all_indices = list(range(n))

        if epoch < self.warmup_epochs:
            return all_indices

        # Curriculum: threshold decreases over time
        # At warmup_epochs: only easy pairs (diff > easy_threshold)
        # By epoch 50+: all pairs included
        progress = min(1.0, (epoch - self.warmup_epochs) / 45.0)
        min_diff = self.easy_threshold * (1.0 - progress)

        # Filter by difficulty
        mask = self.pair_diffs >= min_diff
        indices = [i for i in all_indices if mask[i]]

        # Oversample hard pairs if model errors available
        if self._model_errors is not None and len(indices) > 0:
            error_threshold = np.percentile(self._model_errors, 75)
            hard_indices = [
                i for i in indices
                if self._model_errors[i] > error_threshold
            ]
            oversample_count = int(len(hard_indices) * (self.hard_oversample - 1))
            if oversample_count > 0 and hard_indices:
                rng = np.random.default_rng(epoch)
                extra = rng.choice(hard_indices, size=oversample_count, replace=True)
                indices.extend(extra.tolist())

        return indices if indices else all_indices


class MixWeightedSampler(Sampler):
    """Sample indices from multiple datasets to maintain target per-tier fractions.

    Treats datasets as laid out end-to-end in a virtual concatenated index space
    (same convention as torch.utils.data.ConcatDataset).  Each epoch draws
    ``epoch_size`` indices total; each dataset contributes approximately its
    target fraction, sampling with replacement when the requested count exceeds
    the dataset size.

    With a single dataset the sampler degrades to uniform random sampling with
    replacement, preserving epoch_size semantics for future multi-tier runs.

    Args:
        dataset_sizes: Number of items in each dataset, in index order.
        weights: Target fraction for each dataset.  Normalized internally so
            they need not sum to 1.0.
        epoch_size: Total indices yielded per epoch.  Defaults to the sum of
            all dataset sizes.
        generator: Optional torch.Generator for reproducibility.
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        weights: list[float],
        epoch_size: int | None = None,
        generator: torch.Generator | None = None,
    ):
        if len(dataset_sizes) != len(weights):
            raise ValueError("dataset_sizes and weights must have the same length")
        if not dataset_sizes:
            raise ValueError("dataset_sizes must be non-empty")
        if any(s <= 0 for s in dataset_sizes):
            raise ValueError("all dataset_sizes must be positive")

        self.dataset_sizes = dataset_sizes
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [w / total_weight for w in weights]
        self.epoch_size = epoch_size if epoch_size is not None else sum(dataset_sizes)
        self.generator = generator

        # Global index offsets: dataset i occupies [offsets[i], offsets[i+1])
        self.offsets: list[int] = []
        offset = 0
        for s in dataset_sizes:
            self.offsets.append(offset)
            offset += s

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        # Allocate sample counts per dataset; last dataset absorbs rounding error
        counts = [max(1, round(self.epoch_size * w)) for w in self.weights]
        counts[-1] = max(1, self.epoch_size - sum(counts[:-1]))

        indices: list[int] = []
        for ds_idx, (count, size, offset) in enumerate(
            zip(counts, self.dataset_sizes, self.offsets)
        ):
            local = torch.randint(
                0, size, (count,), generator=self.generator
            ).tolist()
            indices.extend(i + offset for i in local)

        # Shuffle the combined list
        perm = torch.randperm(len(indices), generator=self.generator).tolist()
        return iter(indices[p] for p in perm)


def apply_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation on embeddings and labels.

    Randomly shuffles the batch and interpolates:
        mixed = lambda * original + (1 - lambda) * shuffled
    where lambda ~ Beta(alpha, alpha).

    Args:
        embeddings: [B, T, D] frame embeddings.
        labels: [B, num_dims] label scores.
        alpha: Beta distribution parameter. 0 = no mixup.

    Returns:
        (mixed_embeddings, mixed_labels) with same shapes as inputs.
    """
    if alpha <= 0.0:
        return embeddings, labels

    batch_size = embeddings.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # Ensure lambda >= 0.5 so the "original" sample dominates
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(batch_size, device=embeddings.device)

    mixed_emb = lam * embeddings + (1.0 - lam) * embeddings[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]

    return mixed_emb, mixed_labels


class AudioSegmentDataset(Dataset):
    """Per-segment dataset for self-supervised audio training (A2 Stage 1).

    Returns individual segments with clean + augmented embeddings and piece IDs.
    Used for contrastive cross-performer learning and augmentation invariance.
    """

    def __init__(
        self,
        embeddings: dict,
        piece_to_keys: dict,
        keys: list[str],
        noise_std: float = 0.01,
        max_frames: int | None = None,
    ):
        self.embeddings = embeddings
        self.noise_std = noise_std
        self.max_frames = max_frames

        # Build key -> piece_id mapping
        key_to_piece_id: Dict[str, int] = {}
        for pid_idx, (pid, pkeys) in enumerate(sorted(piece_to_keys.items())):
            for k in pkeys:
                key_to_piece_id[k] = pid_idx

        # Keep only keys that have both embeddings and piece membership
        self.samples = [
            (k, key_to_piece_id[k])
            for k in keys
            if k in embeddings and k in key_to_piece_id
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        key, piece_id = self.samples[idx]
        emb = self.embeddings[key]
        if self.max_frames is not None and emb.shape[0] > self.max_frames:
            emb = emb[: self.max_frames]
        emb = emb.float()
        aug = emb + torch.randn_like(emb) * self.noise_std
        return {
            "embeddings_clean": emb,
            "embeddings_augmented": aug,
            "piece_ids": torch.tensor(piece_id),
        }


class MaestroContrastiveDataset(Dataset):
    """Lazy-loading MAESTRO dataset for Stage 1 contrastive + invariance training.

    Loads .pt embedding files from disk per __getitem__ to avoid OOM
    when the full dataset (~34GB) exceeds available RAM.
    """

    def __init__(
        self,
        emb_dir: str | Path,
        contrastive_mapping: dict,
        piece_id_offset: int = 0,
        noise_std: float = 0.01,
        max_frames: int | None = None,
    ):
        self.emb_dir = Path(emb_dir)
        self.noise_std = noise_std
        self.max_frames = max_frames
        self.samples: List[Tuple[str, int]] = []
        pid = piece_id_offset
        for piece, keys in sorted(contrastive_mapping.items()):
            valid = [k for k in keys if (self.emb_dir / f"{k}.pt").exists()]
            if len(valid) >= 2:
                for k in valid:
                    self.samples.append((k, pid))
                pid += 1
        print(
            f"MaestroContrastiveDataset: {len(self.samples)} segments, "
            f"{pid - piece_id_offset} pieces (offset={piece_id_offset})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        key, piece_id = self.samples[idx]
        emb = torch.load(  # nosemgrep
            self.emb_dir / f"{key}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if self.max_frames is not None and emb.shape[0] > self.max_frames:
            emb = emb[: self.max_frames]
        emb = emb.float()
        aug = emb + torch.randn_like(emb) * self.noise_std
        return {
            "embeddings_clean": emb,
            "embeddings_augmented": aug,
            "piece_ids": torch.tensor(piece_id),
        }


class PracticeAugmentedDataset(Dataset):
    """Practice-distribution wrapper for MAESTRO contrastive training.

    Wraps a base dataset that has a `.samples` attribute (list of (key, piece_id)
    tuples). On each __getitem__, with probability p_corrupt the pre-rendered
    corrupted embedding is loaded from corrupted_emb_dir instead of the clean
    embedding from emb_dir. If the corrupted embedding is absent for a given
    key, the clean version is returned silently (no error).

    Corrupted embedding filenames follow the convention:
        corrupt_{original_segment_id}.pt

    These are generated offline by scripts/render_corrupted_audio.py.

    Args:
        base_dataset: A MaestroContrastiveDataset (or any Dataset with a
            `.samples` list of (key, piece_id) tuples).
        corrupted_emb_dir: Directory containing corrupt_{key}.pt files.
        p_corrupt: Probability of substituting a corrupted embedding per sample.
            0.0 = never corrupt; 1.0 = always corrupt (for ablation).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corrupted_emb_dir: str | Path,
        p_corrupt: float = 0.5,
    ) -> None:
        if not hasattr(base_dataset, "samples"):
            raise ValueError(
                "base_dataset must expose a .samples list of (key, piece_id) tuples. "
                "MaestroContrastiveDataset satisfies this contract."
            )
        if not 0.0 <= p_corrupt <= 1.0:
            raise ValueError(f"p_corrupt must be in [0, 1], got {p_corrupt}")

        self.base_dataset = base_dataset
        self.corrupted_emb_dir = Path(corrupted_emb_dir)
        self.p_corrupt = p_corrupt

        # Mirror the .samples attribute so callers can introspect keys
        self.samples: List[Tuple[str, int]] = base_dataset.samples  # type: ignore[attr-defined]

        available = sum(
            1 for key, _ in self.samples
            if (self.corrupted_emb_dir / f"corrupt_{key}.pt").exists()
        )
        print(
            f"PracticeAugmentedDataset: {len(self.samples)} segments, "
            f"{available} with pre-rendered corrupted embeddings, "
            f"p_corrupt={p_corrupt}"
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]

        if torch.rand(1).item() < self.p_corrupt:
            key, _piece_id = self.samples[idx]
            corrupted_path = self.corrupted_emb_dir / f"corrupt_{key}.pt"
            if corrupted_path.exists():
                corrupt_emb = torch.load(  # nosemgrep
                    corrupted_path,
                    map_location="cpu",
                    weights_only=True,
                )
                noise_std = getattr(self.base_dataset, "noise_std", 0.01)
                max_frames = getattr(self.base_dataset, "max_frames", None)

                if max_frames is not None and corrupt_emb.shape[0] > max_frames:
                    corrupt_emb = corrupt_emb[:max_frames]

                corrupt_emb = corrupt_emb.float()
                corrupt_aug = corrupt_emb + torch.randn_like(corrupt_emb) * noise_std

                item = dict(item)
                item["embeddings_clean"] = corrupt_emb
                item["embeddings_augmented"] = corrupt_aug

        return item


class AugmentedEmbeddingDataset(Dataset):
    """T4: Returns clean + augmented embedding pairs for invariance training.

    Wraps another dataset and applies augmentation to create pairs.
    The augmentor callable should accept a tensor and return an augmented copy.
    """

    def __init__(self, base_dataset: Dataset, augmentor=None):
        if base_dataset is None:
            raise ValueError("base_dataset must not be None")
        self.base_dataset = base_dataset
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]

        if "embeddings" not in item:
            raise KeyError(
                "Base dataset items must contain 'embeddings' key for augmentation"
            )

        clean = item["embeddings"]

        if self.augmentor is not None:
            augmented = self.augmentor(clean)
        else:
            # Default augmentation: add small Gaussian noise
            augmented = clean + torch.randn_like(clean) * 0.01

        result = dict(item)
        result["embeddings_clean"] = clean
        result["embeddings_augmented"] = augmented
        return result


class MIDIPretrainingDataset(Dataset):
    """T4: Masked token prediction for symbolic encoder pretraining.

    Takes pre-tokenized sequences and applies random masking at mask_prob.
    Uses BERT-style masking: of the masked positions, 80% become a special
    mask token, 10% become a random token, and 10% remain unchanged.
    """

    MASK_REPLACE_PROB = 0.8
    RANDOM_REPLACE_PROB = 0.1
    # Remaining 0.1 stays unchanged

    def __init__(
        self,
        token_sequences: list[list[int]],
        max_seq_len: int = 512,
        mask_prob: float = 0.15,
        vocab_size: int = 500,
    ):
        if not token_sequences:
            raise ValueError("token_sequences must be a non-empty list")
        if not 0.0 < mask_prob < 1.0:
            raise ValueError(f"mask_prob must be between 0 and 1, got {mask_prob}")

        self.token_sequences = token_sequences
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size
        # Use vocab_size as the special [MASK] token ID (outside normal vocab range)
        self.mask_token_id = vocab_size

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> dict:
        tokens = list(self.token_sequences[idx])

        # Truncate if longer than max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        original_len = len(tokens)

        # Pad if shorter than max_seq_len
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:original_len] = 1

        # Create mask for positions to consider (only non-padding)
        mask_candidates = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask_candidates[:original_len] = True

        # Randomly select positions to mask
        rand = torch.rand(self.max_seq_len)
        mask_positions = mask_candidates & (rand < self.mask_prob)

        # Set labels for masked positions to original tokens
        labels[mask_positions] = input_ids[mask_positions]

        # BERT-style replacement for masked positions
        mask_indices = mask_positions.nonzero(as_tuple=True)[0]
        num_masked = mask_indices.shape[0]

        if num_masked > 0:
            replace_probs = torch.rand(num_masked)

            # 80% of masked positions: replace with [MASK] token
            mask_replace = replace_probs < self.MASK_REPLACE_PROB
            input_ids[mask_indices[mask_replace]] = self.mask_token_id

            # 10% of masked positions: replace with random token
            random_replace = (replace_probs >= self.MASK_REPLACE_PROB) & (
                replace_probs < self.MASK_REPLACE_PROB + self.RANDOM_REPLACE_PROB
            )
            num_random = random_replace.sum().item()
            if num_random > 0:
                random_tokens = torch.randint(0, self.vocab_size, (num_random,))
                input_ids[mask_indices[random_replace]] = random_tokens

            # Remaining 10%: keep original token (already in input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def multi_task_collate_fn(batch: list[dict]) -> dict:
    """Collate for multi-task training (handles variable fields per sample).

    For each key present in the batch:
    - If values are tensors with 2+ dims: pad to max length along dim 0, stack
    - If values are tensors with 1 dim: stack directly
    - If values are scalars/strings: collect into a list

    Also creates mask tensors for padded sequences (key + "_mask").
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    # Collect all keys across the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    collated: dict = {}

    for key in sorted(all_keys):
        # Gather values for this key (only from items that have it)
        values = [item[key] for item in batch if key in item]

        if not values:
            continue

        first = values[0]

        if isinstance(first, torch.Tensor):
            if first.dim() >= 2:
                # Variable-length sequences: pad along dim 0 to max length
                max_len = max(v.shape[0] for v in values)
                padded = []
                masks = []
                for v in values:
                    length = v.shape[0]
                    if length < max_len:
                        # Pad with zeros along dim 0
                        pad_shape = list(v.shape)
                        pad_shape[0] = max_len - length
                        padding = torch.zeros(pad_shape, dtype=v.dtype)
                        padded.append(torch.cat([v, padding], dim=0))
                    else:
                        padded.append(v)
                    # Create boolean mask for valid positions
                    mask = torch.zeros(max_len, dtype=torch.bool)
                    mask[:length] = True
                    masks.append(mask)

                dtypes = {p.dtype for p in padded}
                if len(dtypes) > 1:
                    padded = [p.to(torch.float32) for p in padded]

                collated[key] = torch.stack(padded)
                collated[key + "_mask"] = torch.stack(masks)

            elif first.dim() == 1:
                # Fixed-size tensors: stack directly
                collated[key] = torch.stack(values)

            else:
                # Scalar tensors
                collated[key] = torch.stack(values)
        else:
            # Strings, ints, or other non-tensor types: collect into a list
            collated[key] = values

    return collated


def stage1_collate_fn(batch: list[dict]) -> dict:
    """Collate for Stage 1 self-supervised training.

    Wraps multi_task_collate_fn and renames embeddings_clean_mask -> mask
    to match _self_supervised_step expectations.
    """
    collated = multi_task_collate_fn(batch)
    if "embeddings_clean_mask" in collated:
        collated["mask"] = collated["embeddings_clean_mask"]
    return collated


def audio_pair_collate_fn(
    batch: list[dict],
    embeddings: dict,
) -> dict:
    """Collate paired performance data with pre-computed audio embeddings.

    Attaches pre-extracted MuQ embeddings to paired data from PairedPerformanceDataset.
    Used with functools.partial to bind embeddings.

    Args:
        batch: List of dicts from PairedPerformanceDataset.
        embeddings: Dict mapping performance keys to [T, D] embedding tensors.

    Returns:
        Collated batch for MuQ audio encoders.

    Raises:
        RuntimeError: If no valid pairs are found in the batch.
    """
    embs_a, embs_b = [], []
    labels_a_list, labels_b_list = [], []
    piece_ids_a, piece_ids_b = [], []
    tiers: list[str] = []

    for item in batch:
        key_a, key_b = item["key_a"], item["key_b"]
        if key_a not in embeddings or key_b not in embeddings:
            continue

        embs_a.append(embeddings[key_a])
        embs_b.append(embeddings[key_b])
        labels_a_list.append(item["labels_a"])
        labels_b_list.append(item["labels_b"])
        piece_ids_a.append(item["piece_id"])
        piece_ids_b.append(item["piece_id"])
        tiers.append(item.get("tier", "percepiano"))

    if not embs_a:
        raise RuntimeError(
            "audio_pair_collate_fn: no valid pairs in batch -- "
            "check that all dataset keys exist in embeddings dict"
        )

    # Pad variable-length sequences to batch max
    def _pad_and_stack(tensors):
        if tensors[0].dim() < 2:
            return torch.stack(tensors), None
        max_len = max(t.shape[0] for t in tensors)
        D = tensors[0].shape[-1]
        padded, masks = [], []
        for t in tensors:
            T = t.shape[0]
            mask = torch.ones(max_len, dtype=torch.bool)
            if T < max_len:
                t = torch.cat([t, torch.zeros(max_len - T, D)], dim=0)
                mask[T:] = False
            padded.append(t)
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks)

    stacked_a, masks_a = _pad_and_stack(embs_a)
    stacked_b, masks_b = _pad_and_stack(embs_b)

    result = {
        "embeddings_a": stacked_a,
        "embeddings_b": stacked_b,
        "labels_a": torch.stack(labels_a_list),
        "labels_b": torch.stack(labels_b_list),
        "piece_ids_a": torch.tensor(piece_ids_a),
        "piece_ids_b": torch.tensor(piece_ids_b),
        "tiers": tiers,
    }
    if masks_a is not None:
        result["mask_a"] = masks_a
    if masks_b is not None:
        result["mask_b"] = masks_b

    return result


def symbolic_collate_fn(
    batch: list[dict],
    token_sequences: dict,
    max_len: int = 2048,
) -> dict:
    """Collate paired performance data with tokenized MIDI sequences.

    Attaches tokenized MIDI to paired data from PairedPerformanceDataset.
    Used with functools.partial to bind token_sequences.

    Args:
        batch: List of dicts from PairedPerformanceDataset.
        token_sequences: Dict mapping performance keys to token lists.
        max_len: Maximum sequence length for padding/truncation.

    Returns:
        Collated batch for TransformerSymbolicEncoder._finetune_step().

    Raises:
        RuntimeError: If no valid pairs are found in the batch.
    """
    ids_a, ids_b, masks_a, masks_b = [], [], [], []
    labels_a_list, labels_b_list = [], []
    piece_ids_a, piece_ids_b = [], []

    for item in batch:
        key_a, key_b = item["key_a"], item["key_b"]
        if key_a not in token_sequences or key_b not in token_sequences:
            continue

        tok_a = token_sequences[key_a][:max_len]
        tok_b = token_sequences[key_b][:max_len]

        pad_a = tok_a + [0] * (max_len - len(tok_a))
        pad_b = tok_b + [0] * (max_len - len(tok_b))

        ids_a.append(torch.tensor(pad_a, dtype=torch.long))
        ids_b.append(torch.tensor(pad_b, dtype=torch.long))

        mask_a = torch.zeros(max_len, dtype=torch.bool)
        mask_a[: len(tok_a)] = True
        masks_a.append(mask_a)

        mask_b = torch.zeros(max_len, dtype=torch.bool)
        mask_b[: len(tok_b)] = True
        masks_b.append(mask_b)

        labels_a_list.append(item["labels_a"])
        labels_b_list.append(item["labels_b"])
        piece_ids_a.append(item["piece_id"])
        piece_ids_b.append(item["piece_id"])

    if not ids_a:
        raise RuntimeError(
            "symbolic_collate_fn: no valid pairs in batch -- "
            "check that all dataset keys exist in token_sequences dict"
        )

    return {
        "input_ids_a": torch.stack(ids_a),
        "input_ids_b": torch.stack(ids_b),
        "mask_a": torch.stack(masks_a),
        "mask_b": torch.stack(masks_b),
        "labels_a": torch.stack(labels_a_list),
        "labels_b": torch.stack(labels_b_list),
        "piece_ids_a": torch.tensor(piece_ids_a),
        "piece_ids_b": torch.tensor(piece_ids_b),
    }


def continuous_collate_fn(
    batch: list[dict],
    features_dict: dict,
    max_len: int = 2000,
) -> dict:
    """Collate paired performance data with continuous features.

    Attaches continuous features to paired data from PairedPerformanceDataset.
    Used with functools.partial to bind features_dict.

    Args:
        batch: List of dicts from PairedPerformanceDataset.
        features_dict: Dict mapping performance keys to [T, C] feature tensors.
        max_len: Maximum sequence length for padding/truncation.

    Returns:
        Collated batch for ContinuousSymbolicEncoder._finetune_step().

    Raises:
        RuntimeError: If no valid pairs are found in the batch.
    """
    feats_a, feats_b = [], []
    masks_a, masks_b = [], []
    labels_a_list, labels_b_list = [], []
    piece_ids_a, piece_ids_b = [], []

    def _pad_feat(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, C = feat.shape
        if T > max_len:
            feat = feat[:max_len]
            T = max_len
        mask = torch.ones(max_len, dtype=torch.bool)
        if T < max_len:
            padding = torch.zeros(max_len - T, C)
            feat = torch.cat([feat, padding], dim=0)
            mask[T:] = False
        return feat, mask

    for item in batch:
        key_a, key_b = item["key_a"], item["key_b"]
        if key_a not in features_dict or key_b not in features_dict:
            continue

        fa, ma = _pad_feat(features_dict[key_a])
        fb, mb = _pad_feat(features_dict[key_b])

        feats_a.append(fa)
        feats_b.append(fb)
        masks_a.append(ma)
        masks_b.append(mb)
        labels_a_list.append(item["labels_a"])
        labels_b_list.append(item["labels_b"])
        piece_ids_a.append(item["piece_id"])
        piece_ids_b.append(item["piece_id"])

    if not feats_a:
        raise RuntimeError(
            "continuous_collate_fn: no valid pairs in batch -- "
            "check that all dataset keys exist in features_dict"
        )

    return {
        "features_a": torch.stack(feats_a),
        "features_b": torch.stack(feats_b),
        "mask_a": torch.stack(masks_a),
        "mask_b": torch.stack(masks_b),
        "labels_a": torch.stack(labels_a_list),
        "labels_b": torch.stack(labels_b_list),
        "piece_ids_a": torch.tensor(piece_ids_a),
        "piece_ids_b": torch.tensor(piece_ids_b),
    }


class ContinuousPretrainDataset(Dataset):
    """Masked feature prediction pretraining dataset for ContinuousSymbolicEncoder (S3).

    Takes pre-extracted continuous features and applies random masking.
    """

    def __init__(
        self,
        keys: list[str],
        features_dict: dict,
        max_len: int = 2000,
        mask_prob: float = 0.15,
    ):
        if not keys:
            raise ValueError("keys must be a non-empty list")
        self.keys = keys
        self.features = features_dict
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        feat = self.features[key]  # [T, C]
        T, C = feat.shape

        if T > self.max_len:
            feat = feat[: self.max_len]
            T = self.max_len

        mask = torch.ones(self.max_len, dtype=torch.bool)
        if T < self.max_len:
            padding = torch.zeros(self.max_len - T, C)
            feat = torch.cat([feat, padding], dim=0)
            mask[T:] = False

        masked_feat = feat.clone()
        rand = torch.rand(self.max_len)
        masked_positions = mask.clone() & (rand < self.mask_prob)
        masked_feat[masked_positions] = 0.0

        return {
            "features": feat,
            "mask": mask,
            "masked_features": masked_feat,
            "masked_positions": masked_positions,
        }


class ShardedContinuousPretrainDataset(Dataset):
    """Lazy-loads feature shards (~152MB per shard).

    Replaces ContinuousPretrainDataset when LOW_MEMORY=True.
    Returns the same dict format expected by the default collate.
    """

    def __init__(
        self,
        keys: list[str],
        shard_index: dict,
        max_len: int = 2000,
        mask_prob: float = 0.15,
    ):
        self.keys = keys
        self.shard_index = shard_index
        self.max_len = max_len
        self.mask_prob = mask_prob
        self._cached_shard_path = None
        self._cached_shard = None

    def __len__(self) -> int:
        return len(self.keys)

    def _load_shard(self, shard_path):
        if self._cached_shard_path != shard_path:
            del self._cached_shard
            self._cached_shard = torch.load(  # nosemgrep
                shard_path, map_location="cpu", weights_only=False
            )
            self._cached_shard_path = shard_path
        return self._cached_shard

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        shard_path = self.shard_index[key]
        shard = self._load_shard(shard_path)
        feat = shard[key]  # [T, C]
        T, C = feat.shape

        if T > self.max_len:
            feat = feat[: self.max_len]
            T = self.max_len

        mask = torch.ones(self.max_len, dtype=torch.bool)
        if T < self.max_len:
            padding = torch.zeros(self.max_len - T, C)
            feat = torch.cat([feat, padding], dim=0)
            mask[T:] = False

        masked_feat = feat.clone()
        rand = torch.rand(self.max_len)
        masked_positions = mask.clone() & (rand < self.mask_prob)
        masked_feat[masked_positions] = 0.0

        return {
            "features": feat,
            "mask": mask,
            "masked_features": masked_feat,
            "masked_positions": masked_positions,
        }


class SemiSupConBatchSampler(BatchSampler):
    """BatchSampler for semi-supervised contrastive pretraining.

    Guarantees each batch contains at least `min_t5_pairs` items drawn as
    same-ordinal pairs from T5 data, so the labeled positive signal in
    semi_sup_con_loss is activated on every step.

    The remainder of each batch is filled from T1/T2 data via weighted
    tier sampling (identical to WeightedTierSampler logic).

    Args:
        t1_t2_tier_piece_indices: List of (piece_id -> global_indices) dicts
            covering T1 and T2 datasets (global indices into ConcatDataset).
        t1_t2_weights: Sampling weight per T1/T2 tier (normalized internally).
        t5_ordinal_indices: Mapping from ordinal bucket (0-4) to list of global
            indices (into the same ConcatDataset). Only buckets with >=2 items
            are used; others are silently skipped.
        batch_size: Target batch size.
        total_batches: Number of batches to yield per epoch.
        min_t5_pairs: Minimum number of same-ordinal pairs to place in each
            batch. Each pair contributes 2 items, so effective T5 minimum is
            min_t5_pairs * 2. Clamped to batch_size // 2.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        t1_t2_tier_piece_indices: list[dict[int, list[int]]],
        t1_t2_weights: list[float],
        t5_ordinal_indices: dict[int, list[int]],
        batch_size: int,
        total_batches: int,
        min_t5_pairs: int = 2,
        seed: int = 42,
    ):
        # BatchSampler expects a sampler; we override __iter__ entirely.
        super().__init__(sampler=range(0), batch_size=batch_size, drop_last=False)
        self._batch_size = batch_size
        self._total_batches = total_batches
        self._seed = seed

        # Clamp so there is always room for at least one T1/T2 item
        self._min_t5_pairs = min(min_t5_pairs, (batch_size - 1) // 2)
        self._t5_slots = self._min_t5_pairs * 2

        # T1/T2 tier sampling
        w_sum = sum(t1_t2_weights) or 1.0
        self._t1t2_weights = [w / w_sum for w in t1_t2_weights]
        self._t1t2_tier_piece_indices = t1_t2_tier_piece_indices

        # T5 ordinal buckets -- keep only those with >=2 items
        self._t5_ordinals = {
            k: v for k, v in t5_ordinal_indices.items() if len(v) >= 2
        }
        self._t5_has_data = bool(self._t5_ordinals)

    def __iter__(self):
        rng = _random.Random(self._seed)

        tier_pieces = [list(pm.keys()) for pm in self._t1t2_tier_piece_indices]
        valid_tiers = [i for i, p in enumerate(tier_pieces) if p]
        valid_weights = [self._t1t2_weights[i] for i in valid_tiers]

        t5_ordinal_keys = list(self._t5_ordinals.keys())

        for _ in range(self._total_batches):
            batch: list[int] = []

            # T5 labeled pairs: sample min_t5_pairs same-ordinal pairs
            if self._t5_has_data and self._t5_slots > 0:
                for _ in range(self._min_t5_pairs):
                    ordinal = rng.choice(t5_ordinal_keys)
                    pair = rng.sample(self._t5_ordinals[ordinal], k=2)
                    batch.extend(pair)

            # Fill remainder with T1/T2 tier-piece sampling
            remaining = self._batch_size - len(batch)
            for _ in range(remaining):
                if not valid_tiers:
                    break
                tier_idx = rng.choices(valid_tiers, weights=valid_weights, k=1)[0]
                pieces = tier_pieces[tier_idx]
                piece_id = rng.choice(pieces)
                segment_indices = self._t1t2_tier_piece_indices[tier_idx][piece_id]
                batch.append(rng.choice(segment_indices))

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._total_batches
