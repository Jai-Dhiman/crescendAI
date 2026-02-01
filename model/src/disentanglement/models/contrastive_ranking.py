"""Approach A: Contrastive Pairwise Ranking Model.

Combines InfoNCE contrastive learning with pairwise ranking.
The contrastive loss uses piece membership as positive pairs.

Architecture:
- Shared encoder (pooling + projection)
- Projection head for contrastive learning
- Ranking heads for each dimension

Loss:
    L_total = L_ranking + lambda * L_infonce

References:
- InfoNCE: https://lilianweng.github.io/posts/2021-05-31-contrastive/
- SimCLR: https://arxiv.org/abs/2002.05709
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from ..losses import piece_based_infonce_loss, DimensionWiseRankingLoss


class ContrastivePairwiseRankingModel(pl.LightningModule):
    """Contrastive pairwise ranking model.

    Uses InfoNCE with piece-based positives to learn piece-invariant
    representations, then applies pairwise ranking for each dimension.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        margin: float = 0.2,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

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

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Comparison module for pairwise ranking
        # Takes [z_a; z_b; z_a - z_b; z_a * z_b]
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

        # Loss
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )

        self.val_outputs = []

    def pool(self, x, mask=None):
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

    def encode(self, x, mask=None):
        """Encode and pool embeddings."""
        pooled = self.pool(x, mask)
        return self.encoder(pooled)

    def project(self, z):
        """Project to contrastive space."""
        return self.projection(z)

    def compare(self, z_a, z_b):
        """Compare two embeddings and produce ranking logits."""
        # Compute comparison features
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)

        # Get comparison representation
        comp = self.comparator(concat)

        # Per-dimension ranking predictions
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)  # [B, num_labels]

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        """Forward pass for pairwise ranking.

        Args:
            emb_a: Embeddings for sample A [B, T, D].
            emb_b: Embeddings for sample B [B, T, D].
            mask_a: Attention mask for A [B, T].
            mask_b: Attention mask for B [B, T].

        Returns:
            Dict with ranking_logits, z_a, z_b, proj_a, proj_b.
        """
        # Encode both samples
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)

        # Project for contrastive loss
        proj_a = self.project(z_a)
        proj_b = self.project(z_b)

        # Compare for ranking
        ranking_logits = self.compare(z_a, z_b)

        return {
            "ranking_logits": ranking_logits,
            "z_a": z_a,
            "z_b": z_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
        }

    def training_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        # Ranking loss
        l_rank = self.ranking_loss(
            outputs["ranking_logits"],
            batch["labels_a"],
            batch["labels_b"],
        )

        # Contrastive loss
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # Total loss
        loss = l_rank + self.lambda_contrastive * l_contrast

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_contrast_loss", l_contrast)

        return loss

    def validation_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        # Ranking loss
        l_rank = self.ranking_loss(
            outputs["ranking_logits"],
            batch["labels_a"],
            batch["labels_b"],
        )

        self.log("val_loss", l_rank, prog_bar=True)

        # Store for accuracy computation
        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        # Compute pairwise accuracy
        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        # True ranking: A > B if labels_a > labels_b
        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        # Only count non-ambiguous pairs (where there's a clear winner)
        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= 0.05

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
