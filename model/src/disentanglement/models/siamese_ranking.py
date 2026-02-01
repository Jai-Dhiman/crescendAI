"""Approach B: Siamese Dimension-Specific Ranking Model.

A siamese network that takes pairs of performances and predicts
which one is better for each dimension.

Architecture:
- Shared encoder (processes both inputs identically)
- Comparison module (combines pair representations)
- 19 dimension-specific ranking heads

Loss:
    BCE with label smoothing, ignoring ambiguous pairs

References:
- RankNet: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
- DirectRanker: https://arxiv.org/abs/1909.02768
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseDimensionRankingModel(pl.LightningModule):
    """Siamese network for dimension-specific pairwise ranking.

    Processes both performances with a shared encoder and predicts
    A > B probability for each of the 19 dimensions.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        comparison_type: str = "concat_diff",  # "concat_diff" or "bilinear"
        margin: float = 0.3,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.05,
        pooling: str = "attention",
        max_epochs: int = 200,
        dimension_indices: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.comparison_type = comparison_type
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels
        # Dimension indices for per-group training (None = all dimensions)
        self.dimension_indices = dimension_indices

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

        # Comparison module
        if comparison_type == "concat_diff":
            # [z_a; z_b; z_a - z_b; z_a * z_b]
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
            final_dim = hidden_dim // 2
        else:  # bilinear
            # Bilinear comparison: z_a^T W z_b for each dimension
            self.bilinear = nn.ModuleList([
                nn.Bilinear(hidden_dim, hidden_dim, hidden_dim // 4)
                for _ in range(num_labels)
            ])
            final_dim = hidden_dim // 4

        # Per-dimension ranking heads
        self.ranking_heads = nn.ModuleList([
            nn.Linear(final_dim, 1) for _ in range(num_labels)
        ])

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

    def compare(self, z_a, z_b):
        """Compare two embeddings and produce ranking logits."""
        if self.comparison_type == "concat_diff":
            diff = z_a - z_b
            prod = z_a * z_b
            concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
            comp = self.comparator(concat)
            logits = [head(comp) for head in self.ranking_heads]
        else:  # bilinear
            logits = []
            for i, (bilinear, head) in enumerate(zip(self.bilinear, self.ranking_heads)):
                bi_out = bilinear(z_a, z_b)
                logits.append(head(bi_out))

        return torch.cat(logits, dim=-1)  # [B, num_labels]

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        """Forward pass for pairwise ranking.

        Returns:
            ranking_logits: [B, num_labels] where >0 means A > B.
        """
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)
        return self.compare(z_a, z_b)

    def compute_loss(self, logits, labels_a, labels_b):
        """Compute dimension-wise ranking loss.

        Args:
            logits: Predicted A>B logits [B, num_labels].
            labels_a: Ground truth for A [B, num_labels].
            labels_b: Ground truth for B [B, num_labels].

        Returns:
            Scalar loss.
        """
        # True ranking direction
        label_diff = labels_a - labels_b  # [B, D]

        # Targets: 1 if A > B, 0 if B > A
        targets = (label_diff > 0).float()

        # Mask for non-ambiguous pairs
        non_ambiguous = label_diff.abs() >= self.ambiguous_threshold

        if not non_ambiguous.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss on non-ambiguous pairs only
        loss = F.binary_cross_entropy_with_logits(
            logits[non_ambiguous],
            targets[non_ambiguous],
            reduction="mean",
        )

        return loss

    def _filter_labels(self, labels):
        """Filter labels to dimension_indices if specified."""
        if self.dimension_indices is not None:
            return labels[:, self.dimension_indices]
        return labels

    def training_step(self, batch, idx):
        logits = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        labels_a = self._filter_labels(batch["labels_a"])
        labels_b = self._filter_labels(batch["labels_b"])
        loss = self.compute_loss(logits, labels_a, labels_b)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        logits = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        labels_a = self._filter_labels(batch["labels_a"])
        labels_b = self._filter_labels(batch["labels_b"])
        loss = self.compute_loss(logits, labels_a, labels_b)
        self.log("val_loss", loss, prog_bar=True)

        self.val_outputs.append({
            "logits": logits.cpu(),
            "labels_a": labels_a.cpu(),
            "labels_b": labels_b.cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        # True ranking
        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        # Non-ambiguous pairs
        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= self.ambiguous_threshold

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()

            # Per-dimension accuracy
            per_dim_acc = []
            for d in range(self.num_labels):
                dim_mask = non_ambiguous[:, d]
                if dim_mask.any():
                    dim_correct = (pred_ranking[:, d][dim_mask] == true_ranking[:, d][dim_mask]).float()
                    per_dim_acc.append(dim_correct.mean().item())
            if per_dim_acc:
                self.log("val_per_dim_acc_mean", sum(per_dim_acc) / len(per_dim_acc))
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
