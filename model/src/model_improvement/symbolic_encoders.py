"""Symbolic encoder experiments: Transformer (S1), GNN (S2), continuous (S3) on MIDI."""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from disentanglement.losses import DimensionWiseRankingLoss, piece_based_infonce_loss


class TransformerSymbolicEncoder(pl.LightningModule):
    """S1: BERT-style Transformer on REMI-tokenized MIDI.

    Two stages:
    - pretrain: Masked token prediction (MLM) on unlabeled MIDI data.
    - finetune: Pairwise ranking + regression on labeled pairs.

    Architecture: token embed + sinusoidal positional encoding +
    N TransformerEncoder layers + attention pooling + projection +
    ranking/regression heads.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        stage: str = "pretrain",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.5,
        margin: float = 0.2,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_regression = lambda_regression
        self.max_epochs = max_epochs
        self.num_labels = num_labels
        self.d_model = d_model
        self.stage = stage

        if stage not in ("pretrain", "finetune"):
            raise ValueError(f"stage must be 'pretrain' or 'finetune', got '{stage}'")

        # Token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention pooling to get sequence-level representation
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Projection to hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -- Pretrain head: masked token prediction --
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size + 1),
        )

        # -- Finetune heads --
        # Comparator for pairwise ranking
        comparison_dim = hidden_dim * 4
        self.comparator = nn.Sequential(
            nn.Linear(comparison_dim, hidden_dim),
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

        # Regression head for absolute quality prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Losses
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )

        self.val_outputs: list[dict] = []

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embed + positional encoding."""
        x = self.token_embed(input_ids)
        x = self.pos_encoding(x)
        return self.embed_dropout(x)

    def _transformer_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run transformer encoder, returning per-token hidden states [B, T, d_model]."""
        x = self._embed(input_ids)
        # TransformerEncoder expects src_key_padding_mask with True for padded positions
        padding_mask = ~attention_mask
        return self.transformer(x, src_key_padding_mask=padding_mask)

    def _attention_pool(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Attention-pool transformer output to [B, d_model]."""
        scores = self.attn_pool(hidden).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * w).sum(1)

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pooled embedding z_symbolic [B, hidden_dim]."""
        hidden = self._transformer_forward(input_ids, attention_mask)
        pooled = self._attention_pool(hidden, attention_mask)
        return self.projection(pooled)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:
        z = self.encode(input_ids, attention_mask)
        scores = self.regression_head(z)
        return {"z_symbolic": z, "scores": scores}

    def compare(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        if self.stage == "pretrain":
            return self._pretrain_step(batch)
        return self._finetune_step(batch)

    def _pretrain_step(self, batch: dict) -> torch.Tensor:
        """Masked token prediction loss."""
        hidden = self._transformer_forward(batch["input_ids"], batch["attention_mask"])
        logits = self.mlm_head(hidden)  # [B, T, vocab_size+1]
        # labels has -100 for non-masked positions (ignored by cross_entropy)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )
        self.log("train_mlm_loss", loss, prog_bar=True)
        return loss

    def _finetune_step(self, batch: dict) -> torch.Tensor:
        """Pairwise ranking + contrastive + regression loss."""
        z_a = self.encode(batch["input_ids_a"], batch["mask_a"])
        z_b = self.encode(batch["input_ids_b"], batch["mask_b"])

        ranking_logits = self.compare(z_a, z_b)

        # Ranking loss
        l_rank = self.ranking_loss(
            ranking_logits, batch["labels_a"], batch["labels_b"],
        )

        # Contrastive loss
        proj_a = self.contrastive_proj(z_a)
        proj_b = self.contrastive_proj(z_b)
        all_proj = torch.cat([proj_a, proj_b], dim=0)
        all_pieces = torch.cat([batch["piece_ids_a"], batch["piece_ids_b"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # Regression loss
        scores_a = self.regression_head(z_a)
        scores_b = self.regression_head(z_b)
        l_reg = (
            F.mse_loss(scores_a, batch["labels_a"])
            + F.mse_loss(scores_b, batch["labels_b"])
        ) / 2.0

        loss = (
            l_rank
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_contrast_loss", l_contrast)
        self.log("train_reg_loss", l_reg)

        return loss

    def validation_step(self, batch: dict, idx: int) -> None:
        if self.stage == "pretrain":
            hidden = self._transformer_forward(batch["input_ids"], batch["attention_mask"])
            logits = self.mlm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            self.log("val_mlm_loss", loss, prog_bar=True)
        else:
            z_a = self.encode(batch["input_ids_a"], batch["mask_a"])
            z_b = self.encode(batch["input_ids_b"], batch["mask_b"])
            ranking_logits = self.compare(z_a, z_b)

            l_rank = self.ranking_loss(
                ranking_logits, batch["labels_a"], batch["labels_b"],
            )
            self.log("val_loss", l_rank, prog_bar=True)

            self.val_outputs.append({
                "logits": ranking_logits.cpu(),
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
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input [B, T, D]."""
        return x + self.pe[:, : x.size(1)]
