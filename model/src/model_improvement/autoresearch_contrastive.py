"""Contrastive pretraining pipeline for MuQ and Aria encoders.

Symmetric quality-aware contrastive training using piece-based InfoNCE
and ordinal margin losses. Trains encoder + projection head pairs that
can be transferred to the downstream multi-task model (Phase C).

Architecture mirrors MuQLoRAModel exactly (attn, encoder, projection)
so that pretrained weights can be loaded directly.

Usage:
    cd model/
    uv run python -m model_improvement.autoresearch_contrastive --encoder muq
    uv run python -m model_improvement.autoresearch_contrastive --encoder aria \
        --lambda-infonce 1.0 --lambda-ordinal 0.5

Exit code 0 on success, 1 on failure.
"""

import argparse
import gc
import json
import random as _random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from model_improvement.losses import ordinal_margin_loss, piece_based_infonce_loss
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS
from model_improvement.training import train_model
from model_improvement.aria_linear_probe import (
    train_linear_probe,
    compute_pairwise_from_regression,
    load_embeddings_as_matrix,
)
from model_improvement.metrics import MetricsSuite
from src.paths import Checkpoints, Embeddings, Labels, Manifests


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
            if key not in embeddings_dict or key not in labels_dict or key not in key_to_piece_id:
                continue
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
                f"Competition embeddings not found at {embeddings_dir}. "
                "Extract embeddings first (e.g. run aria_embeddings.py for Aria)."
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
                    embedding = torch.load(  # nosemgrep
                        seg_file, map_location="cpu", weights_only=True
                    )
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

        # Filter to non-empty tiers and adjust weights
        valid_tiers = [i for i, p in enumerate(tier_pieces) if p]
        if not valid_tiers:
            return
        valid_weights = [self._weights[i] for i in valid_tiers]

        for _ in range(self._total_samples):
            # Pick tier by weight (only from non-empty tiers)
            tier_idx = rng.choices(valid_tiers, weights=valid_weights, k=1)[0]

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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

FOLD_IDX = 0
BATCH_SIZE = 16
ACCUM_BATCHES = 2
CHECKPOINT_BASE = Checkpoints.root / "contrastive_pretrain"


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_t2_records(split: str) -> list[dict]:
    """Load T2 competition records from recordings.jsonl.

    Args:
        split: "train" (everything except Cliburn 2022) or "val" (Cliburn 2022 only).

    Returns:
        List of record dicts with keys: recording_id, competition, edition,
        round, placement, performer, piece, audio_path, etc.

    Raises:
        ValueError: If split is not "train" or "val".
        FileNotFoundError: If recordings.jsonl does not exist.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    manifest_path = Manifests.competition / "recordings.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"T2 manifest not found: {manifest_path}")

    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    def _is_cliburn_2022(r: dict) -> bool:
        return r["competition"] == "cliburn" and r["edition"] == 2022

    if split == "val":
        return [r for r in records if _is_cliburn_2022(r)]
    else:
        return [r for r in records if not _is_cliburn_2022(r)]


# ---------------------------------------------------------------------------
# Single-Fold Training
# ---------------------------------------------------------------------------


def run_single_fold(
    encoder_type: str,
    lambda_infonce: float = 1.0,
    lambda_ordinal: float = 0.5,
    temperature: float = 0.07,
    ordinal_margin: float = 0.1,
    t1_weight: float = 0.2,
    t2_weight: float = 0.8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
) -> dict:
    """Train a single contrastive pretraining fold and run linear probe.

    Args:
        encoder_type: "muq" or "aria".
        lambda_infonce: Weight for InfoNCE loss.
        lambda_ordinal: Weight for ordinal margin loss.
        temperature: InfoNCE temperature.
        ordinal_margin: Margin scale for ordinal loss.
        t1_weight: Sampling weight for T1 (PercePiano).
        t2_weight: Sampling weight for T2 (competition).
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Maximum training epochs.

    Returns:
        Dict with: contrastive_loss, ordinal_loss, probe_pairwise, probe_r2,
        elapsed_seconds.

    Raises:
        ValueError: If encoder_type is not "muq" or "aria".
        RuntimeError: If training fails.
    """
    if encoder_type not in ("muq", "aria"):
        raise ValueError(f"encoder_type must be 'muq' or 'aria', got {encoder_type!r}")

    pl.seed_everything(42, workers=True)

    # ---- Load T1 (PercePiano) data ----
    composite_path = Labels.composite / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    with open(Labels.percepiano / "folds.json") as f:
        folds = json.load(f)
    with open(Labels.percepiano / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    fold = folds[FOLD_IDX]

    if encoder_type == "muq":
        emb_path = Embeddings.percepiano / "muq_embeddings.pt"
        t1_embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep
        input_dim = 1024
    else:
        emb_path = Embeddings.percepiano / "aria_embedding.pt"
        t1_embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)  # nosemgrep
        input_dim = 512

    # ---- Load T2 (competition) data ----
    t2_train_records = _load_t2_records("train")
    t2_val_records = _load_t2_records("val")

    # Convert dicts to SimpleNamespace for from_t2 (expects attribute access)
    t2_train_ns = [SimpleNamespace(**r) for r in t2_train_records]
    t2_val_ns = [SimpleNamespace(**r) for r in t2_val_records]

    if encoder_type == "muq":
        comp_emb_dir = Embeddings.competition / "muq_embeddings"
    else:
        comp_emb_dir = Embeddings.competition / "aria_embeddings"

    # ---- Build datasets ----
    t1_train_ds = ContrastiveSegmentDataset.from_t1(
        t1_embeddings, labels, piece_to_keys, fold["train"], piece_id_offset=0,
    )
    t1_val_ds = ContrastiveSegmentDataset.from_t1(
        t1_embeddings, labels, piece_to_keys, fold["val"], piece_id_offset=0,
    )

    t2_piece_offset = t1_train_ds.num_pieces + t1_val_ds.num_pieces + 100

    t2_train_ds = ContrastiveSegmentDataset.from_t2(
        comp_emb_dir, t2_train_ns, piece_id_offset=t2_piece_offset,
    )
    t2_val_ds = ContrastiveSegmentDataset.from_t2(
        comp_emb_dir, t2_val_ns, piece_id_offset=t2_piece_offset + 10000,
    )

    # Combined datasets for DataLoader (ConcatDataset)
    train_ds = ConcatDataset([t1_train_ds, t2_train_ds])
    val_ds = ConcatDataset([t1_val_ds, t2_val_ds])

    # ---- Samplers ----
    total_train = max(len(t1_train_ds) + len(t2_train_ds), BATCH_SIZE * 10)
    total_val = max(len(t1_val_ds) + len(t2_val_ds), BATCH_SIZE * 4)

    train_sampler = WeightedTierSampler(
        [t1_train_ds, t2_train_ds], [t1_weight, t2_weight],
        total_samples=total_train, seed=42,
    )
    val_sampler = WeightedTierSampler(
        [t1_val_ds, t2_val_ds], [t1_weight, t2_weight],
        total_samples=total_val, seed=123,
    )

    # ---- DataLoaders ----
    if encoder_type == "muq":
        collate_fn = contrastive_collate_muq
    else:
        collate_fn = contrastive_collate_aria

    train_loader = DataLoader(  # nosemgrep
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(  # nosemgrep
        val_ds, batch_size=BATCH_SIZE, sampler=val_sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    # ---- Build encoder and model ----
    if encoder_type == "muq":
        encoder = MuQContrastiveEncoder(input_dim=input_dim)
    else:
        encoder = AriaContrastiveEncoder(input_dim=input_dim)

    model = ContrastivePretrainModel(
        encoder=encoder,
        lambda_infonce=lambda_infonce,
        lambda_ordinal=lambda_ordinal,
        temperature=temperature,
        margin_scale=ordinal_margin,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_epochs=5,
        max_epochs=max_epochs,
    )

    # ---- Train ----
    CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Contrastive pretrain: encoder={encoder_type}, "
          f"lambda_infonce={lambda_infonce}, lambda_ordinal={lambda_ordinal}, "
          f"temperature={temperature}, margin={ordinal_margin}")

    _cleanup_memory()
    start_time = time.time()

    trainer = train_model(
        model, train_loader, val_loader,
        f"contrastive_{encoder_type}", FOLD_IDX,
        checkpoint_dir=CHECKPOINT_BASE,
        max_epochs=max_epochs,
        patience=7,
        accumulate_grad_batches=ACCUM_BATCHES,
    )

    # ---- Load best checkpoint ----
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint):
            ckpt_callback = cb
            break

    if ckpt_callback is not None and ckpt_callback.best_model_path:
        print(f"Loading best checkpoint: {ckpt_callback.best_model_path}")
        best_model = ContrastivePretrainModel.load_from_checkpoint(
            ckpt_callback.best_model_path, encoder=encoder,
        )
    else:
        best_model = trainer.lightning_module

    best_model.cpu()
    best_model.eval()

    # Extract final validation losses
    metrics = trainer.callback_metrics
    contrastive_loss = float(metrics.get("val_infonce", float("nan")))
    ordinal_loss_val = float(metrics.get("val_ordinal", float("nan")))

    # ---- Linear probe on T1 PercePiano ----
    # Use encode() output (hidden_dim), NOT project() output
    best_encoder = best_model.encoder
    best_encoder.eval()

    # Build probe embeddings from T1 data
    probe_train_embs = []
    probe_train_keys = []
    probe_val_embs = []
    probe_val_keys = []

    with torch.no_grad():
        for key in fold["train"]:
            if key not in t1_embeddings:
                continue
            emb = t1_embeddings[key]
            if encoder_type == "muq":
                # [T, 1024] -> [1, T, 1024]
                emb_in = emb.unsqueeze(0)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool)
                z = best_encoder.encode(emb_in, mask)  # [1, hidden_dim]
            else:
                # [512] -> [1, 512]
                emb_in = emb.unsqueeze(0)
                z = best_encoder.encode(emb_in)  # [1, hidden_dim]
            probe_train_embs.append(z.squeeze(0))
            probe_train_keys.append(key)

        for key in fold["val"]:
            if key not in t1_embeddings:
                continue
            emb = t1_embeddings[key]
            if encoder_type == "muq":
                emb_in = emb.unsqueeze(0)
                mask = torch.ones(1, emb.shape[0], dtype=torch.bool)
                z = best_encoder.encode(emb_in, mask)
            else:
                emb_in = emb.unsqueeze(0)
                z = best_encoder.encode(emb_in)
            probe_val_embs.append(z.squeeze(0))
            probe_val_keys.append(key)

    probe_train_matrix = torch.stack(probe_train_embs)
    probe_val_matrix = torch.stack(probe_val_embs)

    # Build label tensors
    probe_train_labels = torch.tensor(
        np.array([labels[k][:NUM_DIMS] for k in probe_train_keys]),
        dtype=torch.float32,
    )
    probe_val_labels = torch.tensor(
        np.array([labels[k][:NUM_DIMS] for k in probe_val_keys]),
        dtype=torch.float32,
    )

    # Train linear probe
    val_preds, _ = train_linear_probe(
        probe_train_matrix, probe_train_labels,
        probe_val_matrix, probe_val_labels,
    )

    # Compute pairwise accuracy
    pw = compute_pairwise_from_regression(val_preds, probe_val_keys, labels)
    probe_pairwise = pw["overall"]

    # Compute R2
    suite = MetricsSuite()
    probe_r2 = suite.regression_r2(val_preds, probe_val_labels)

    elapsed = time.time() - start_time

    # Cleanup
    del model, trainer, best_model, best_encoder
    del train_ds, val_ds, train_loader, val_loader
    del t1_embeddings
    _cleanup_memory()

    return {
        "contrastive_loss": round(contrastive_loss, 6),
        "ordinal_loss": round(ordinal_loss_val, 6),
        "probe_pairwise": round(probe_pairwise, 6),
        "probe_r2": round(probe_r2, 6),
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive pretraining for autoresearch loop"
    )
    parser.add_argument(
        "--encoder", required=True, choices=["muq", "aria"],
        help="Encoder type to pretrain",
    )
    parser.add_argument("--lambda-infonce", type=float, default=1.0)
    parser.add_argument("--lambda-ordinal", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ordinal-margin", type=float, default=0.1)
    parser.add_argument("--t1-weight", type=float, default=0.2)
    parser.add_argument("--t2-weight", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    result = run_single_fold(
        encoder_type=args.encoder,
        lambda_infonce=args.lambda_infonce,
        lambda_ordinal=args.lambda_ordinal,
        temperature=args.temperature,
        ordinal_margin=args.ordinal_margin,
        t1_weight=args.t1_weight,
        t2_weight=args.t2_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    # Structured output for autoresearch parsing
    print(f"\n{'='*60}")
    print("AUTORESEARCH_RESULT")
    print(f"contrastive_loss={result['contrastive_loss']:.6f}")
    print(f"ordinal_loss={result['ordinal_loss']:.6f}")
    print(f"probe_pairwise={result['probe_pairwise']:.6f}")
    print(f"probe_r2={result['probe_r2']:.6f}")
    print(f"elapsed={result['elapsed_seconds']}s")
    print(f"{'='*60}")

    # Also dump as JSON for machine parsing
    print("AUTORESEARCH_JSON=" + json.dumps(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
