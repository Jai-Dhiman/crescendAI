"""Dataset classes for T2-T4 data tiers: competition, paired, augmented, pretraining."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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
    ):
        self.cache_dir = Path(cache_dir)
        self.labels = labels

        # Filter to keys that exist in both the provided keys list and labels
        valid_keys = set(keys) & set(labels.keys())

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

        labels_a = torch.tensor(self.labels[key_a][:19], dtype=torch.float32)
        labels_b = torch.tensor(self.labels[key_b][:19], dtype=torch.float32)

        return {
            "key_a": key_a,
            "key_b": key_b,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "piece_id": self.piece_to_id[piece_id],
        }


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
