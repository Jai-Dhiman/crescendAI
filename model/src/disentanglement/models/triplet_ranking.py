"""Triplet-based ranking model for performer discrimination.

This model uses triplet sampling within same-piece performances to learn
quality differences while preserving performer variance.

Experiment: E11
Reference: V7Labs triplet loss literature
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from ..losses import TripletPerformerLoss, DimensionWiseRankingLoss


class TripletRankingModel(pl.LightningModule):
    """Triplet-based model for performer discrimination.

    Architecture:
    - Shared encoder for all inputs (anchor, positive, negative)
    - Triplet loss encourages anchor closer to positive than negative
    - Optional ranking head for per-dimension comparisons

    The triplet sampling ensures:
    - Anchor, positive, negative are all from the same piece
    - Positive has higher mean quality score than anchor
    - Negative has lower mean quality score than anchor
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        margin: float = 0.5,
        lambda_ranking: float = 0.5,
        ambiguous_threshold: float = 0.05,
        pooling: str = "attention",
        distance_fn: str = "euclidean",
        max_epochs: int = 100,
    ):
        """Initialize triplet ranking model.

        Args:
            input_dim: Input embedding dimension (1024 for MuQ).
            hidden_dim: Hidden layer dimension.
            embedding_dim: Output embedding dimension for triplet loss.
            num_labels: Number of dimensions to predict (19 for PercePiano).
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            margin: Triplet loss margin.
            lambda_ranking: Weight for ranking loss component.
            ambiguous_threshold: Threshold for ambiguous pair filtering.
            pooling: Pooling strategy ("mean" or "attention").
            distance_fn: Distance function ("euclidean" or "cosine").
            max_epochs: Maximum training epochs.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.lambda_ranking = lambda_ranking

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Shared encoder
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

        # Embedding projection for triplet loss
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)

        # Ranking head for per-dimension comparison
        # Takes concatenated difference [h_anchor; h_anchor - h_other]
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        # Losses
        self.triplet_loss = TripletPerformerLoss(
            margin=margin, distance_fn=distance_fn
        )
        self.ranking_loss = DimensionWiseRankingLoss(
            ambiguous_threshold=ambiguous_threshold
        )

        self.val_outputs = []

    def pool(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Pool frame embeddings to sequence representation."""
        if self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode input embeddings to hidden representation."""
        pooled = self.pool(x, mask)
        return self.encoder(pooled)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        mask_anchor: torch.Tensor = None,
        mask_positive: torch.Tensor = None,
        mask_negative: torch.Tensor = None,
    ) -> dict:
        """Forward pass computing embeddings and rankings.

        Args:
            anchor: Anchor embeddings [B, T, D].
            positive: Positive embeddings [B, T, D].
            negative: Negative embeddings [B, T, D].
            mask_anchor: Attention mask for anchor [B, T].
            mask_positive: Attention mask for positive [B, T].
            mask_negative: Attention mask for negative [B, T].

        Returns:
            Dict with:
                - h_anchor, h_positive, h_negative: Hidden representations
                - z_anchor, z_positive, z_negative: Projected embeddings
                - ranking_logits_pos: A vs positive ranking logits
                - ranking_logits_neg: A vs negative ranking logits
        """
        # Encode all three inputs
        h_anchor = self.encode(anchor, mask_anchor)
        h_positive = self.encode(positive, mask_positive)
        h_negative = self.encode(negative, mask_negative)

        # Project to embedding space for triplet loss
        z_anchor = self.embedding_proj(h_anchor)
        z_positive = self.embedding_proj(h_positive)
        z_negative = self.embedding_proj(h_negative)

        # Ranking logits: anchor vs positive and anchor vs negative
        diff_pos = h_anchor - h_positive
        diff_neg = h_anchor - h_negative

        ranking_input_pos = torch.cat([h_anchor, diff_pos], dim=-1)
        ranking_input_neg = torch.cat([h_anchor, diff_neg], dim=-1)

        ranking_logits_pos = self.ranking_head(ranking_input_pos)
        ranking_logits_neg = self.ranking_head(ranking_input_neg)

        return {
            "h_anchor": h_anchor,
            "h_positive": h_positive,
            "h_negative": h_negative,
            "z_anchor": z_anchor,
            "z_positive": z_positive,
            "z_negative": z_negative,
            "ranking_logits_pos": ranking_logits_pos,
            "ranking_logits_neg": ranking_logits_neg,
        }

    def training_step(self, batch, idx):
        outputs = self(
            batch["embeddings_anchor"],
            batch["embeddings_positive"],
            batch["embeddings_negative"],
            batch.get("mask_anchor"),
            batch.get("mask_positive"),
            batch.get("mask_negative"),
        )

        # Triplet loss
        l_triplet = self.triplet_loss(
            outputs["z_anchor"],
            outputs["z_positive"],
            outputs["z_negative"],
        )

        # Ranking loss: anchor should rank below positive (pos scores higher)
        l_rank_pos = self.ranking_loss(
            outputs["ranking_logits_pos"],
            batch["labels_anchor"],
            batch["labels_positive"],
        )

        # Ranking loss: anchor should rank above negative (neg scores lower)
        l_rank_neg = self.ranking_loss(
            outputs["ranking_logits_neg"],
            batch["labels_anchor"],
            batch["labels_negative"],
        )

        l_ranking = (l_rank_pos + l_rank_neg) / 2

        loss = l_triplet + self.lambda_ranking * l_ranking

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_triplet_loss", l_triplet)
        self.log("train_ranking_loss", l_ranking)

        return loss

    def validation_step(self, batch, idx):
        outputs = self(
            batch["embeddings_anchor"],
            batch["embeddings_positive"],
            batch["embeddings_negative"],
            batch.get("mask_anchor"),
            batch.get("mask_positive"),
            batch.get("mask_negative"),
        )

        l_triplet = self.triplet_loss(
            outputs["z_anchor"],
            outputs["z_positive"],
            outputs["z_negative"],
        )

        self.log("val_triplet_loss", l_triplet, prog_bar=True)

        # Store outputs for accuracy computation
        self.val_outputs.append({
            "ranking_logits_neg": outputs["ranking_logits_neg"].cpu(),
            "labels_anchor": batch["labels_anchor"].cpu(),
            "labels_negative": batch["labels_negative"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        # Compute pairwise accuracy on anchor vs negative
        all_logits = torch.cat([x["ranking_logits_neg"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_anchor"] for x in self.val_outputs])
        all_labels_n = torch.cat([x["labels_negative"] for x in self.val_outputs])

        # True ranking: anchor > negative where anchor has higher label
        label_diff = all_labels_a - all_labels_n
        non_ambiguous = label_diff.abs() >= 0.05

        if non_ambiguous.any():
            true_ranking = label_diff > 0
            pred_ranking = all_logits > 0

            correct = (true_ranking == pred_ranking) & non_ambiguous
            acc = correct.sum().float() / non_ambiguous.sum().float()
            self.log("val_pairwise_acc", acc, prog_bar=True)

        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
