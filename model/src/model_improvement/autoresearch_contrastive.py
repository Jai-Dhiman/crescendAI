"""Contrastive pretraining pipeline for MuQ and Aria encoders.

Symmetric quality-aware contrastive training using piece-based InfoNCE
and ordinal margin losses. Trains encoder + projection head pairs that
can be transferred to the downstream multi-task model (Phase C).

Architecture mirrors MuQLoRAModel exactly (attn, encoder, projection)
so that pretrained weights can be loaded directly.
"""

import random as _random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from model_improvement.losses import ordinal_margin_loss, piece_based_infonce_loss


class MuQContrastiveEncoder(nn.Module):
    """MuQ contrastive encoder with attention pooling.

    Architecture matches MuQLoRAModel (attn, encoder, projection) exactly
    so Phase C can initialize from these pretrained weights.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Attention pooling over temporal frames
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Shared encoder (2-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-pool variable-length frames, then encode.

        Args:
            x: Frame embeddings [B, T, D].
            mask: Valid frame mask [B, T], True for real frames.

        Returns:
            Encoded representation [B, hidden_dim].
        """
        # Attention scores [B, T, 1]
        scores = self.attn(x)

        # Mask padded frames with -inf before softmax
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = (weights * x).sum(dim=1)  # [B, D]

        return self.encoder(pooled)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoded representation to contrastive space.

        Args:
            z: Encoded representation [B, hidden_dim].

        Returns:
            L2-normalized projection [B, projection_dim].
        """
        return F.normalize(self.projection(z), dim=-1)


class AriaContrastiveEncoder(nn.Module):
    """Aria contrastive encoder for fixed-dim symbolic embeddings.

    No attention pooling needed -- Aria produces a single vector per segment.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder (2-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode fixed-dim symbolic embedding.

        Args:
            x: Symbolic embeddings [B, D].
            mask: Ignored (accepted for API compatibility).

        Returns:
            Encoded representation [B, hidden_dim].
        """
        return self.encoder(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoded representation to contrastive space.

        Args:
            z: Encoded representation [B, hidden_dim].

        Returns:
            L2-normalized projection [B, projection_dim].
        """
        return F.normalize(self.projection(z), dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ContrastiveSegmentDataset(Dataset):
    """Dataset of segments with piece membership and quality scores.

    Each item is a dict with:
        embedding: Tensor [T, D] (MuQ variable-length) or [D] (Aria fixed-dim)
        piece_id: int -- unique piece identifier across tiers
        quality_score: float in [0, 1]
    """

    def __init__(self, items: list[dict]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]

    @property
    def num_pieces(self) -> int:
        return len({item["piece_id"] for item in self._items})

    @classmethod
    def from_t1(
        cls,
        embeddings_dict: dict[str, torch.Tensor],
        labels_dict: dict[str, list[float]],
        piece_to_keys: dict[str, list[str]],
        keys: list[str],
        piece_id_offset: int = 0,
    ) -> "ContrastiveSegmentDataset":
        """Build dataset from T1 (PercePiano) data.

        Args:
            embeddings_dict: segment_key -> embedding tensor [T, D].
            labels_dict: segment_key -> list of 6 dimension scores.
            piece_to_keys: piece_name -> list of segment keys.
            keys: ordered list of segment keys to include.
            piece_id_offset: offset for piece IDs (for multi-tier uniqueness).

        Returns:
            ContrastiveSegmentDataset with piece-level grouping.
        """
        # Build key -> piece_id mapping from sorted piece_to_keys
        key_to_piece_id: dict[str, int] = {}
        for pid, piece_name in enumerate(sorted(piece_to_keys.keys())):
            for key in piece_to_keys[piece_name]:
                key_to_piece_id[key] = pid + piece_id_offset

        items = []
        for key in keys:
            embedding = embeddings_dict[key]
            scores = labels_dict[key]
            quality = float(np.mean(scores[:6]))
            items.append({
                "embedding": embedding,
                "piece_id": key_to_piece_id[key],
                "quality_score": quality,
            })

        return cls(items)

    @classmethod
    def from_t2(
        cls,
        embeddings_dir: Path,
        records: list,
        piece_id_offset: int = 0,
    ) -> "ContrastiveSegmentDataset":
        """Build dataset from T2 (competition) data.

        Groups by (competition, edition, round) -- NOT piece (free-text).
        Quality normalized within each group: 1.0 - (placement - min) / (max - min).

        Args:
            embeddings_dir: Directory containing {recording_id}_seg*.pt files.
            records: List of CompetitionRecord-like objects with competition,
                edition, round, placement, recording_id attributes.
            piece_id_offset: offset for piece IDs (for multi-tier uniqueness).

        Returns:
            ContrastiveSegmentDataset with round-level grouping.

        Raises:
            FileNotFoundError: If embeddings_dir does not exist.
        """
        embeddings_dir = Path(embeddings_dir)
        if not embeddings_dir.exists():
            raise FileNotFoundError(
                "Aria competition embeddings not extracted yet. "
                "Run aria_embeddings.py on competition MIDIs first."
            )

        # Group records by (competition, edition, round)
        groups: dict[tuple, list] = {}
        for rec in records:
            group_key = (rec.competition, rec.edition, rec.round)
            groups.setdefault(group_key, []).append(rec)

        items = []
        for gid, group_key in enumerate(sorted(groups.keys())):
            group = groups[group_key]
            piece_id = gid + piece_id_offset

            # Normalize placements within group
            placements = [r.placement for r in group]
            min_p = min(placements)
            max_p = max(placements)

            for rec in group:
                if max_p == min_p:
                    quality = 1.0
                else:
                    quality = 1.0 - (rec.placement - min_p) / (max_p - min_p)

                # Load all segment files for this recording
                seg_files = sorted(
                    embeddings_dir.glob(f"{rec.recording_id}_seg*.pt")
                )
                for seg_file in seg_files:
                    embedding = torch.load(seg_file, weights_only=True)
                    items.append({
                        "embedding": embedding,
                        "piece_id": piece_id,
                        "quality_score": quality,
                    })

        return cls(items)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def contrastive_collate_muq(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length MuQ embeddings with padding and masking.

    Args:
        batch: List of dicts with embedding [T, D], piece_id, quality_score.

    Returns:
        Dict with:
            embeddings: [B, T_max, D] zero-padded
            mask: [B, T_max] bool, True for real frames
            piece_ids: [B] long
            quality_scores: [B] float
    """
    max_t = max(item["embedding"].shape[0] for item in batch)
    dim = batch[0]["embedding"].shape[1]
    batch_size = len(batch)

    embeddings = torch.zeros(batch_size, max_t, dim)
    mask = torch.zeros(batch_size, max_t, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["embedding"].shape[0]
        embeddings[i, :t] = item["embedding"]
        mask[i, :t] = True

    piece_ids = torch.tensor(
        [item["piece_id"] for item in batch], dtype=torch.long
    )
    quality_scores = torch.tensor(
        [item["quality_score"] for item in batch], dtype=torch.float
    )

    return {
        "embeddings": embeddings,
        "mask": mask,
        "piece_ids": piece_ids,
        "quality_scores": quality_scores,
    }


def contrastive_collate_aria(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate fixed-dim Aria embeddings by stacking.

    Args:
        batch: List of dicts with embedding [D], piece_id, quality_score.

    Returns:
        Dict with:
            embeddings: [B, D]
            piece_ids: [B] long
            quality_scores: [B] float
    """
    embeddings = torch.stack([item["embedding"] for item in batch])
    piece_ids = torch.tensor(
        [item["piece_id"] for item in batch], dtype=torch.long
    )
    quality_scores = torch.tensor(
        [item["quality_score"] for item in batch], dtype=torch.float
    )

    return {
        "embeddings": embeddings,
        "piece_ids": piece_ids,
        "quality_scores": quality_scores,
    }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class WeightedTierSampler(Sampler[int]):
    """Sampler that draws from multiple tier datasets with configurable weights.

    For each sample, picks a tier by weight, then a random piece within that
    tier, then a random segment within that piece. This ensures batches contain
    segments from multiple pieces (required for contrastive loss).

    Args:
        datasets: List of ContrastiveSegmentDataset, one per tier.
        weights: Sampling probability per tier (will be normalized).
        total_samples: Number of indices to yield per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        datasets: list[ContrastiveSegmentDataset],
        weights: list[float],
        total_samples: int,
        seed: int = 42,
    ):
        if len(datasets) != len(weights):
            raise ValueError(
                f"datasets ({len(datasets)}) and weights ({len(weights)}) "
                "must have the same length"
            )

        self._total_samples = total_samples
        self._seed = seed

        # Normalize weights
        w_sum = sum(weights)
        self._weights = [w / w_sum for w in weights]

        # Build per-tier index: tier -> piece_id -> list of global indices
        # Global index = offset + local index within tier dataset
        self._tier_piece_indices: list[dict[int, list[int]]] = []
        offset = 0
        for ds in datasets:
            piece_map: dict[int, list[int]] = {}
            for local_idx in range(len(ds)):
                item = ds[local_idx]
                pid = item["piece_id"]
                piece_map.setdefault(pid, []).append(offset + local_idx)
            self._tier_piece_indices.append(piece_map)
            offset += len(ds)

    def __iter__(self):
        rng = _random.Random(self._seed)

        # Precompute piece lists per tier for fast random choice
        tier_pieces = [
            list(piece_map.keys()) for piece_map in self._tier_piece_indices
        ]

        for _ in range(self._total_samples):
            # Pick tier by weight
            tier_idx = rng.choices(
                range(len(self._weights)), weights=self._weights, k=1
            )[0]

            # Pick random piece from tier
            pieces = tier_pieces[tier_idx]
            piece_id = rng.choice(pieces)

            # Pick random segment from piece
            segment_indices = self._tier_piece_indices[tier_idx][piece_id]
            yield rng.choice(segment_indices)

    def __len__(self) -> int:
        return self._total_samples


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class ContrastivePretrainModel(pl.LightningModule):
    """Quality-aware contrastive pretraining for MuQ or Aria encoders.

    Combines piece-based InfoNCE (same piece = positive) with ordinal margin
    loss (higher quality should be closer to cross-piece anchors).

    Uses LinearLR warmup followed by cosine annealing.
    """

    def __init__(
        self,
        encoder: nn.Module,
        lambda_infonce: float = 1.0,
        lambda_ordinal: float = 0.5,
        temperature: float = 0.07,
        margin_scale: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.lambda_infonce = lambda_infonce
        self.lambda_ordinal = lambda_ordinal
        self.temperature = temperature
        self.margin_scale = margin_scale
        self.lr = learning_rate
        self.wd = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = self.encoder.encode(batch["embeddings"], batch.get("mask"))
        proj = self.encoder.project(z)

        l_infonce = piece_based_infonce_loss(
            proj, batch["piece_ids"], temperature=self.temperature
        )
        l_ordinal = ordinal_margin_loss(
            proj, batch["piece_ids"], batch["quality_scores"],
            margin_scale=self.margin_scale,
        )

        loss = self.lambda_infonce * l_infonce + self.lambda_ordinal * l_ordinal
        return {"loss": loss, "infonce": l_infonce, "ordinal": l_ordinal}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch)
        self.log("train_loss", out["loss"], prog_bar=True)
        self.log("train_infonce", out["infonce"])
        self.log("train_ordinal", out["ordinal"])
        return out["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self._shared_step(batch)
        self.log("val_loss", out["loss"], prog_bar=True)
        self.log("val_infonce", out["infonce"])
        self.log("val_ordinal", out["ordinal"])

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
