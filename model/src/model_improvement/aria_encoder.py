"""Aria LoRA fine-tune model for symbolic quality prediction (Phase C).

Cloud run command (do NOT execute locally — requires A100 + HF data bucket):

    uv run python -m model_improvement.train \
        --model aria-lora \
        --data-root /data \
        --output-dir /checkpoints/aria-lora-phase-c \
        --batch-size 8 \
        --max-epochs 50 \
        --learning-rate 1e-5 \
        --lora-rank 32 \
        --lambda-contrastive 0.6 \
        --lambda-regression 0.8 \
        --lambda-listmle 1.5 \
        --midi-dir /data/midi/percepiano \
        --labels-path /data/labels/composite/composite_labels.json \
        --folds-path /data/labels/percepiano/folds.json \
        --piece-mapping-path /data/labels/percepiano/piece_mapping.json

Expected VRAM:  ~18GB on A100-80GB at batch_size=8, fp32, seq_len=512.
                Use --precision bf16-mixed to halve this (~9GB).
Recommended:    A100-80GB ($2.50/hr on HF Jobs).
T1 only:        ~30 min/epoch. T1+T2: ~4-6 hr/epoch (requires AMT MIDI for T2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from model_improvement.losses import (
    DimensionWiseRankingLoss,
    ListMLELoss,
    ccc_loss,
    piece_based_infonce_loss,
)
from model_improvement.taxonomy import NUM_DIMS

logger = logging.getLogger(__name__)

_ARIA_REPO = "loubb/aria-medium-embedding"

# Aria-medium has 16 transformer layers (0-indexed 0-15).
# We target layers 8-15 (top 50%), proportional to MuQ A1-Max's layers 7-12
# of ~12 total (also top ~50%). Upper layers encode abstract quality features;
# lower layers handle token-level note identity and onset encoding.
_DEFAULT_LORA_LAYERS = tuple(range(8, 16))

# Lazy tokenizer cache — AbsTokenizer construction is expensive (~200ms).
_tokenizer_cache: dict[str, AbsTokenizer] = {}


def _get_tokenizer() -> AbsTokenizer:
    if "abs" not in _tokenizer_cache:
        _tokenizer_cache["abs"] = AbsTokenizer()
    return _tokenizer_cache["abs"]


def _load_aria_backbone_with_lora(
    lora_rank: int,
    lora_target_layers: tuple[int, ...],
) -> nn.Module:
    """Load aria-medium-embedding and apply PEFT LoRA to upper layers.

    Downloads weights from HuggingFace on first call (2.5GB, cached thereafter).

    Args:
        lora_rank: LoRA rank r. Alpha is set to 2*r (standard scaling).
        lora_target_layers: 0-indexed layer indices to apply LoRA to.

    Returns:
        PEFT-wrapped TransformerEMB with LoRA on specified layers.
        Only LoRA params are trainable; all base params are frozen.

    Raises:
        RuntimeError: If weight loading fails.
        ImportError: If peft or safetensors are not installed.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from peft import LoraConfig, get_peft_model
    from aria.model import ModelConfig, TransformerEMB

    logger.info("Loading Aria config from %s", _ARIA_REPO)
    config_path = hf_hub_download(_ARIA_REPO, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    mc = ModelConfig(
        d_model=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        n_layers=config["num_hidden_layers"],
        max_seq_len=config.get("max_seq_len", 2048),
        ff_mult=config["intermediate_size"] // config["hidden_size"],
        emb_size=config["embedding_size"],
        vocab_size=config["vocab_size"],
        drop_p=0.0,
        grad_checkpoint=False,
    )
    n_layers = config["num_hidden_layers"]

    invalid = [i for i in lora_target_layers if i >= n_layers]
    if invalid:
        raise ValueError(
            f"LoRA target layers {invalid} out of range for {n_layers}-layer Aria"
        )

    logger.info("Downloading Aria weights (~2.5GB, cached after first run)")
    weights_path = hf_hub_download(_ARIA_REPO, "model.safetensors")
    state_dict = load_file(weights_path)

    model = TransformerEMB(mc)
    model.load_state_dict(state_dict)
    logger.info("Aria weights loaded")

    # Build PEFT target module paths.
    # TransformerEMB stores its transformer as self.model, so paths are
    # model.encode_layers.{i}.mixed_qkv (fused QKV) and
    # model.encode_layers.{i}.att_proj_linear (output projection).
    target_modules = []
    for i in lora_target_layers:
        target_modules.append(f"model.encode_layers.{i}.mixed_qkv")
        target_modules.append(f"model.encode_layers.{i}.att_proj_linear")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


class AriaLoRAModel(pl.LightningModule):
    """Phase C: Aria-medium-embedding + LoRA for symbolic quality prediction.

    Mirrors MuQLoRAMaxModel in interface and loss weights (A1-Max defaults).
    Takes batched MIDI token sequences instead of audio frame embeddings.

    Architecture:
        MIDI tokens [B, seq_len]
        -> Aria TransformerEMB + LoRA (layers 8-15 of 16, top ~50%)
        -> EOS-position embedding [B, 512]
        -> Shared encoder (2-layer MLP) -> z [B, hidden_dim]
        -> Projection head (contrastive)
        -> Comparator + per-dim ranking heads
        -> Regression head (sigmoid)

    Batch keys expected by training_step:
        tokens_a, tokens_b: [B, seq_len] int64 token IDs (padded)
        eos_pos_a, eos_pos_b: [B] int64 EOS token positions
        labels_a, labels_b: [B, NUM_DIMS] float quality scores in [0, 1]
        piece_ids_a, piece_ids_b: [B] int64 piece IDs for contrastive loss
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = NUM_DIMS,
        dropout: float = 0.2,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_listmle: float = 1.5,
        lambda_contrastive: float = 0.6,
        lambda_regression: float = 0.8,
        lambda_invariance: float = 0.1,
        lora_rank: int = 32,
        lora_target_layers: tuple[int, ...] = _DEFAULT_LORA_LAYERS,
        mixup_alpha: float = 0.2,
        label_smoothing: float = 0.1,
        ambiguous_threshold: float = 0.05,
        margin: float = 0.2,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_listmle = lambda_listmle
        self.lambda_contrastive = lambda_contrastive
        self.lambda_regression = lambda_regression
        self.lambda_invariance = lambda_invariance
        self.mixup_alpha = mixup_alpha
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # Aria backbone with LoRA (downloads weights on first init)
        self.backbone = _load_aria_backbone_with_lora(lora_rank, lora_target_layers)
        # aria-medium-embedding always outputs 512-dim from its emb_head projection
        aria_emb_dim = 512

        # Shared encoder (same structure as MuQLoRAMaxModel)
        self.encoder = nn.Sequential(
            nn.Linear(aria_emb_dim, hidden_dim),
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

        # Comparator: takes [z_a; z_b; z_a - z_b; z_a * z_b]
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-dimension ranking heads
        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        # Losses
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )
        self._listmle = ListMLELoss()

        self.val_outputs: list[dict] = []

    def _extract_eos_embedding(
        self, tokens: torch.Tensor, eos_positions: torch.Tensor
    ) -> torch.Tensor:
        """Run Aria forward and extract EOS-position embedding.

        Args:
            tokens: Token IDs [B, seq_len].
            eos_positions: EOS token index per sequence [B].

        Returns:
            [B, aria_emb_dim] embeddings at EOS position.
        """
        # [B, seq_len, aria_emb_dim]
        out = self.backbone(tokens)
        B = tokens.shape[0]
        return out[torch.arange(B, device=tokens.device), eos_positions]

    def _encode(
        self, tokens: torch.Tensor, eos_positions: torch.Tensor
    ) -> torch.Tensor:
        """Extract EOS embedding and apply shared encoder MLP."""
        emb = self._extract_eos_embedding(tokens, eos_positions)
        return self.encoder(emb)

    def _compare(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def forward(
        self,
        tokens_a: torch.Tensor,
        eos_pos_a: torch.Tensor,
        tokens_b: torch.Tensor,
        eos_pos_b: torch.Tensor,
    ) -> dict:
        z_a = self._encode(tokens_a, eos_pos_a)
        z_b = self._encode(tokens_b, eos_pos_b)
        proj_a = self.projection(z_a)
        proj_b = self.projection(z_b)
        ranking_logits = self._compare(z_a, z_b)
        return {
            "ranking_logits": ranking_logits,
            "z_a": z_a,
            "z_b": z_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
        }

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        tokens_a = batch["tokens_a"]
        tokens_b = batch["tokens_b"]
        eos_pos_a = batch["eos_pos_a"]
        eos_pos_b = batch["eos_pos_b"]
        labels_a = batch["labels_a"]
        labels_b = batch["labels_b"]

        outputs = self(tokens_a, eos_pos_a, tokens_b, eos_pos_b)

        # 1. Pairwise ranking (BCE)
        l_rank = self.ranking_loss(
            outputs["ranking_logits"], labels_a, labels_b
        )

        # 2. Contrastive
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat(
            [batch["piece_ids_a"], batch["piece_ids_b"]], dim=0
        )
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # 3. Regression (CCC)
        scores_a = self.regression_head(outputs["z_a"])
        scores_b = self.regression_head(outputs["z_b"])
        l_reg = (
            ccc_loss(scores_a, labels_a) + ccc_loss(scores_b, labels_b)
        ) / 2.0

        # 4. ListMLE grouped by piece
        l_listmle = torch.tensor(0.0, device=self.device)
        if self.lambda_listmle > 0:
            all_scores = torch.cat([scores_a, scores_b], dim=0)
            all_labels = torch.cat([labels_a, labels_b], dim=0)
            piece_ids = torch.cat(
                [batch["piece_ids_a"], batch["piece_ids_b"]], dim=0
            )
            count = 0
            for pid in piece_ids.unique():
                pmask = piece_ids == pid
                if pmask.sum() < 2:
                    continue
                l_listmle = l_listmle + self._listmle(
                    all_scores[pmask], all_labels[pmask]
                )
                count += 1
            if count > 0:
                l_listmle = l_listmle / count

        loss = (
            l_rank
            + self.lambda_listmle * l_listmle
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_listmle_loss", l_listmle)
        self.log("train_contrast_loss", l_contrast)
        self.log("train_ccc_loss", l_reg)

        return loss

    def validation_step(self, batch: dict, idx: int) -> None:
        outputs = self(
            batch["tokens_a"], batch["eos_pos_a"],
            batch["tokens_b"], batch["eos_pos_b"],
        )
        l_rank = self.ranking_loss(
            outputs["ranking_logits"], batch["labels_a"], batch["labels_b"]
        )
        self.log("val_loss", l_rank, prog_bar=True)
        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self.val_outputs:
            return
        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])
        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()
        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= 0.05
        if non_ambiguous.any():
            correct = (
                pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]
            ).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5
        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class AriaMidiPairDataset(Dataset):
    """T1 PercePiano MIDI pair dataset for Aria LoRA training.

    Loads MIDI files, tokenizes with AbsTokenizer, and generates all
    valid within-piece pairs. Labels are 6-dim composite quality scores.

    Args:
        midi_dir: Directory of .mid files (stems match label keys).
        labels: Dict mapping segment_id -> float[6] composite labels.
        piece_to_keys: Dict mapping piece_id -> list of segment IDs.
        keys: Subset of segment IDs to use (e.g., fold train split).
        max_seq_len: Max AbsTokenizer sequence length (tokens). Segments
            longer than this are truncated; EOS is re-inserted at the end.
    """

    def __init__(
        self,
        midi_dir: Path,
        labels: dict[str, list[float]],
        piece_to_keys: dict[str, list[str]],
        keys: list[str],
        max_seq_len: int = 512,
    ):
        self.midi_dir = Path(midi_dir)
        self.labels = labels
        self.max_seq_len = max_seq_len

        tokenizer = _get_tokenizer()
        self._pad_id: int = tokenizer.encode([tokenizer.pad_tok])[0]
        self._eos_id: int = tokenizer.encode([tokenizer.eos_tok])[0]

        # Tokenize all segments up front (fast: ~64 notes each)
        valid_keys = set(keys) & set(labels.keys())
        self._tokens: dict[str, list[int]] = {}
        self._eos_pos: dict[str, int] = {}

        for seg_id in valid_keys:
            midi_path = self.midi_dir / f"{seg_id}.mid"
            if not midi_path.exists():
                continue
            token_ids, eos_pos = self._tokenize(midi_path)
            self._tokens[seg_id] = token_ids
            self._eos_pos[seg_id] = eos_pos

        # Build within-piece pairs over valid tokenized segments
        key_to_piece: dict[str, str] = {}
        for pid, pkeys in piece_to_keys.items():
            for k in pkeys:
                if k in self._tokens:
                    key_to_piece[k] = pid

        piece_groups: dict[str, list[str]] = {}
        for k in self._tokens:
            if k not in key_to_piece:
                continue
            pid = key_to_piece[k]
            piece_groups.setdefault(pid, []).append(k)

        self.pairs: list[tuple[str, str, str]] = []
        for pid, seg_keys in piece_groups.items():
            if len(seg_keys) < 2:
                continue
            seg_keys = sorted(seg_keys)
            for i, ka in enumerate(seg_keys):
                for kb in seg_keys[i + 1:]:
                    self.pairs.append((ka, kb, pid))

        all_pieces = sorted(piece_groups.keys())
        self.piece_to_id: dict[str, int] = {p: i for i, p in enumerate(all_pieces)}

        logger.info(
            "AriaMidiPairDataset: %d segments, %d pairs, %d pieces",
            len(self._tokens), len(self.pairs), len(all_pieces),
        )

    def _tokenize(self, midi_path: Path) -> tuple[list[int], int]:
        tokenizer = _get_tokenizer()
        midi_dict = MidiDict.from_midi(str(midi_path))
        tokens = tokenizer.tokenize(midi_dict, add_dim_tok=False)
        token_ids = tokenizer.encode(tokens)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
            # Re-insert EOS at the last position if truncated
            token_ids[-1] = self._eos_id
        eos_pos = token_ids.index(self._eos_id)
        return token_ids, eos_pos

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ka, kb, piece_id = self.pairs[idx]
        return {
            "token_ids_a": self._tokens[ka],
            "eos_pos_a": self._eos_pos[ka],
            "token_ids_b": self._tokens[kb],
            "eos_pos_b": self._eos_pos[kb],
            "labels_a": torch.tensor(self.labels[ka], dtype=torch.float32),
            "labels_b": torch.tensor(self.labels[kb], dtype=torch.float32),
            "piece_id": self.piece_to_id[piece_id],
        }


def aria_midi_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length token sequences into padded batch tensors.

    Pads all sequences to the max length in the batch using the pad token.
    EOS positions are preserved as absolute indices into the padded tensor.
    """
    tokenizer = _get_tokenizer()
    pad_id = tokenizer.encode([tokenizer.pad_tok])[0]

    def pad_seqs(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]
        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        return padded

    tokens_a = pad_seqs([item["token_ids_a"] for item in batch])
    tokens_b = pad_seqs([item["token_ids_b"] for item in batch])
    eos_pos_a = torch.tensor([item["eos_pos_a"] for item in batch], dtype=torch.long)
    eos_pos_b = torch.tensor([item["eos_pos_b"] for item in batch], dtype=torch.long)
    labels_a = torch.stack([item["labels_a"] for item in batch])
    labels_b = torch.stack([item["labels_b"] for item in batch])
    piece_ids = torch.tensor([item["piece_id"] for item in batch], dtype=torch.long)

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "eos_pos_a": eos_pos_a,
        "eos_pos_b": eos_pos_b,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "piece_ids_a": piece_ids,
        "piece_ids_b": piece_ids,
    }
