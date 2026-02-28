"""Dataset classes for T2-T4 data tiers: competition, paired, augmented, pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Edge type names used by hetero_graph_collate_fn
_HETERO_EDGE_TYPES = ["onset", "during", "follow", "silence"]


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

        labels_a = torch.tensor(self.labels[key_a], dtype=torch.float32)
        labels_b = torch.tensor(self.labels[key_b], dtype=torch.float32)

        return {
            "key_a": key_a,
            "key_b": key_b,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "piece_id": self.piece_to_id[piece_id],
        }


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
    ):
        self.embeddings = embeddings
        self.noise_std = noise_std

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
    ):
        self.emb_dir = Path(emb_dir)
        self.noise_std = noise_std
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
        emb = torch.load(
            self.emb_dir / f"{key}.pt",
            map_location="cpu",
            weights_only=True,
        )
        aug = emb + torch.randn_like(emb) * self.noise_std
        return {
            "embeddings_clean": emb,
            "embeddings_augmented": aug,
            "piece_ids": torch.tensor(piece_id),
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


def stage1_collate_fn(batch: list[dict]) -> dict:
    """Collate for Stage 1 self-supervised training.

    Wraps multi_task_collate_fn and renames embeddings_clean_mask -> mask
    to match _self_supervised_step expectations.
    """
    collated = multi_task_collate_fn(batch)
    if "embeddings_clean_mask" in collated:
        collated["mask"] = collated["embeddings_clean_mask"]
    return collated


class ScoreGraphPretrainingDataset(Dataset):
    """Link prediction pretraining dataset for GNN symbolic encoder (S2).

    For each graph, randomly masks a fraction of edges as positive targets
    and samples negative edges. Returns the format expected by
    GNNSymbolicEncoder._pretrain_step().

    Edge sets are built on-the-fly per item (~1-2ms each via set comprehension)
    rather than pre-computed, to avoid ~25GB RAM overhead for 14K+ graphs.
    """

    def __init__(
        self,
        graphs: list,
        mask_fraction: float = 0.15,
        neg_ratio: float = 1.0,
        log_every: int = 500,
    ):
        if not graphs:
            raise ValueError("graphs must be a non-empty list")
        if not 0.0 < mask_fraction < 1.0:
            raise ValueError(f"mask_fraction must be in (0, 1), got {mask_fraction}")

        self.graphs = graphs
        self.mask_fraction = mask_fraction
        self.neg_ratio = neg_ratio
        self._fetch_count = 0
        self._log_every = log_every

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        from model_improvement.graph import build_edge_set, sample_negative_edges

        self._fetch_count += 1
        if self._fetch_count % self._log_every == 0:
            print(
                f"  [ScoreGraphPretrain] {self._fetch_count} items fetched",
                flush=True,
            )

        data = self.graphs[idx]
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Randomly select edges to mask as positive targets
        num_mask = max(1, int(num_edges * self.mask_fraction))
        perm = torch.randperm(num_edges)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]

        pos_edges = edge_index[:, mask_idx]
        remaining_edges = edge_index[:, keep_idx]

        # Sample negative edges (edge set built on-the-fly, GC'd after return)
        num_neg = max(1, int(num_mask * self.neg_ratio))
        edge_set = build_edge_set(data)
        neg_edges = sample_negative_edges(data, num_neg, edge_set=edge_set)

        return {
            "x": data.x,
            "edge_index": remaining_edges,
            "pos_edges": pos_edges,
            "neg_edges": neg_edges,
        }


def graph_pretrain_collate_fn(batch: list[dict]) -> dict:
    """Collate ScoreGraphPretrainingDataset items into a single batched graph."""
    all_x = []
    all_edge_indices = []
    all_pos_edges = []
    all_neg_edges = []
    offset = 0

    for item in batch:
        x = item["x"]
        num_nodes = x.size(0)
        all_x.append(x)
        all_edge_indices.append(item["edge_index"] + offset)
        all_pos_edges.append(item["pos_edges"] + offset)
        all_neg_edges.append(item["neg_edges"] + offset)
        offset += num_nodes

    return {
        "x": torch.cat(all_x, dim=0),
        "edge_index": torch.cat(all_edge_indices, dim=1),
        "pos_edges": torch.cat(all_pos_edges, dim=1),
        "neg_edges": torch.cat(all_neg_edges, dim=1),
    }


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
    }
    if masks_a is not None:
        result["mask_a"] = masks_a
    if masks_b is not None:
        result["mask_b"] = masks_b

    return result


def graph_pair_collate_fn(
    batch: list[dict],
    graphs: dict,
) -> dict:
    """Collate paired performance data with pre-computed score graphs.

    Takes output from PairedPerformanceDataset and attaches graph data
    from pre-computed graphs dict. Batches graphs using PyG's
    Batch.from_data_list().

    Args:
        batch: List of dicts from PairedPerformanceDataset with
            key_a, key_b, labels_a, labels_b, piece_id.
        graphs: Dict mapping performance keys to PyG Data objects.

    Returns:
        Collated batch matching GNNSymbolicEncoder._finetune_step() format.

    Raises:
        RuntimeError: If no valid pairs are found in the batch.
    """
    from torch_geometric.data import Batch

    graphs_a = []
    graphs_b = []
    labels_a_list = []
    labels_b_list = []
    piece_ids_a = []
    piece_ids_b = []

    for item in batch:
        key_a, key_b = item["key_a"], item["key_b"]
        if key_a not in graphs or key_b not in graphs:
            continue

        graphs_a.append(graphs[key_a])
        graphs_b.append(graphs[key_b])
        labels_a_list.append(item["labels_a"])
        labels_b_list.append(item["labels_b"])
        piece_ids_a.append(item["piece_id"])
        piece_ids_b.append(item["piece_id"])

    if not graphs_a:
        raise RuntimeError(
            "graph_pair_collate_fn: no valid pairs in batch -- "
            "check that all dataset keys exist in graphs dict"
        )

    batch_a = Batch.from_data_list(graphs_a)
    batch_b = Batch.from_data_list(graphs_b)

    return {
        "x_a": batch_a.x,
        "edge_index_a": batch_a.edge_index,
        "batch_a": batch_a.batch,
        "x_b": batch_b.x,
        "edge_index_b": batch_b.edge_index,
        "batch_b": batch_b.batch,
        "labels_a": torch.stack(labels_a_list),
        "labels_b": torch.stack(labels_b_list),
        "piece_ids_a": torch.tensor(piece_ids_a),
        "piece_ids_b": torch.tensor(piece_ids_b),
    }


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


def hetero_graph_collate_fn(
    batch: list[dict],
    hetero_graphs: dict,
) -> dict:
    """Collate paired performance data with heterogeneous score graphs.

    Manually tracks node offsets and merges edge_index_dict across batch items.
    Used with functools.partial to bind hetero_graphs.

    Args:
        batch: List of dicts from PairedPerformanceDataset.
        hetero_graphs: Dict mapping performance keys to PyG HeteroData objects.

    Returns:
        Collated batch for GNNHeteroSymbolicEncoder._finetune_step(),
    Raises:
        RuntimeError: If no valid pairs are found in the batch.
    """
    x_dicts_a, ei_dicts_a, batches_a = [], [], []
    x_dicts_b, ei_dicts_b, batches_b = [], [], []
    labels_a_list, labels_b_list = [], []
    piece_ids_a, piece_ids_b = [], []
    offset_a, offset_b = 0, 0

    for item in batch:
        key_a, key_b = item["key_a"], item["key_b"]
        if key_a not in hetero_graphs or key_b not in hetero_graphs:
            continue

        ha = hetero_graphs[key_a]
        hb = hetero_graphs[key_b]
        na = ha["note"].x.shape[0]
        nb = hb["note"].x.shape[0]

        x_dicts_a.append(ha["note"].x)
        x_dicts_b.append(hb["note"].x)

        ei_a = {}
        ei_b = {}
        for etype in _HETERO_EDGE_TYPES:
            et = ("note", etype, "note")
            ei_a[et] = ha[et].edge_index + offset_a
            ei_b[et] = hb[et].edge_index + offset_b
        ei_dicts_a.append(ei_a)
        ei_dicts_b.append(ei_b)

        batches_a.append(torch.full((na,), len(x_dicts_a) - 1, dtype=torch.long))
        batches_b.append(torch.full((nb,), len(x_dicts_b) - 1, dtype=torch.long))
        offset_a += na
        offset_b += nb

        labels_a_list.append(item["labels_a"])
        labels_b_list.append(item["labels_b"])
        piece_ids_a.append(item["piece_id"])
        piece_ids_b.append(item["piece_id"])

    if not x_dicts_a:
        raise RuntimeError(
            "hetero_graph_collate_fn: no valid pairs in batch -- "
            "check that all dataset keys exist in hetero_graphs dict"
        )

    merged_ei_a = {}
    merged_ei_b = {}
    for etype in _HETERO_EDGE_TYPES:
        et = ("note", etype, "note")
        merged_ei_a[et] = torch.cat([d[et] for d in ei_dicts_a], dim=1)
        merged_ei_b[et] = torch.cat([d[et] for d in ei_dicts_b], dim=1)

    return {
        "x_dict_a": {"note": torch.cat(x_dicts_a)},
        "edge_index_dict_a": merged_ei_a,
        "batch_a": torch.cat(batches_a),
        "x_dict_b": {"note": torch.cat(x_dicts_b)},
        "edge_index_dict_b": merged_ei_b,
        "batch_b": torch.cat(batches_b),
        "labels_a": torch.stack(labels_a_list),
        "labels_b": torch.stack(labels_b_list),
        "piece_ids_a": torch.tensor(piece_ids_a),
        "piece_ids_b": torch.tensor(piece_ids_b),
    }


def hetero_pretrain_collate_fn(batch: list[dict]) -> dict:
    """Collate HeteroPretrainDataset items into a single batched dict.

    Concatenates node features with node offsets applied to all edge indices,
    pos_edges, and neg_edges so that node IDs remain consistent across the
    merged graph.
    """
    all_x = []
    all_edge_indices: dict[tuple, list[torch.Tensor]] = {}
    all_pos_edges = []
    all_neg_edges = []
    offset = 0

    for item in batch:
        x = item["x_dict"]["note"]
        num_nodes = x.size(0)
        all_x.append(x)

        for et, ei in item["edge_index_dict"].items():
            if et not in all_edge_indices:
                all_edge_indices[et] = []
            all_edge_indices[et].append(ei + offset)

        all_pos_edges.append(item["pos_edges"] + offset)
        all_neg_edges.append(item["neg_edges"] + offset)
        offset += num_nodes

    merged_ei = {et: torch.cat(eis, dim=1) for et, eis in all_edge_indices.items()}

    return {
        "x_dict": {"note": torch.cat(all_x, dim=0)},
        "edge_index_dict": merged_ei,
        "pos_edges": torch.cat(all_pos_edges, dim=1),
        "neg_edges": torch.cat(all_neg_edges, dim=1),
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


class HeteroPretrainDataset(Dataset):
    """Link prediction pretraining dataset for GNNHeteroSymbolicEncoder (S2H).

    For each key, builds edge masking from the homogeneous graph and returns
    the heterogeneous graph structure for the hetero GNN.

    Edge sets are built on-the-fly per item to avoid ~25GB RAM overhead.
    """

    def __init__(
        self,
        keys: list[str],
        homo_graphs: dict,
        hetero_graphs: dict,
        mask_fraction: float = 0.15,
        log_every: int = 500,
    ):
        self.keys = [k for k in keys if k in homo_graphs and k in hetero_graphs]
        if not self.keys:
            raise ValueError("No valid keys found in both homo_graphs and hetero_graphs")
        self.homo = homo_graphs
        self.hetero = hetero_graphs
        self.mask_fraction = mask_fraction
        self._fetch_count = 0
        self._log_every = log_every

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        from model_improvement.graph import build_edge_set, sample_negative_edges

        self._fetch_count += 1
        if self._fetch_count % self._log_every == 0:
            print(
                f"  [HeteroPretrain] {self._fetch_count} items fetched",
                flush=True,
            )

        key = self.keys[idx]
        homo = self.homo[key]
        hetero = self.hetero[key]

        edge_index = homo.edge_index
        num_edges = edge_index.size(1)
        num_mask = max(1, int(num_edges * self.mask_fraction))
        perm = torch.randperm(num_edges)
        pos_edges = edge_index[:, perm[:num_mask]]
        edge_set = build_edge_set(homo)
        neg_edges = sample_negative_edges(homo, num_mask, edge_set=edge_set)

        x_dict = {"note": hetero["note"].x}
        edge_index_dict = {}
        for etype in _HETERO_EDGE_TYPES:
            edge_index_dict[("note", etype, "note")] = hetero[
                "note", etype, "note"
            ].edge_index

        return {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "pos_edges": pos_edges,
            "neg_edges": neg_edges,
        }
