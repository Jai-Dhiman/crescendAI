"""Contrastive auxiliary loss models for audio experiments.

Models that use contrastive learning as an auxiliary objective to improve
representation learning for piano performance evaluation.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from .losses import contrastive_auxiliary_loss, supervised_contrastive_loss


class ContrastiveAuxiliaryModel(pl.LightningModule):
    """MERT model with contrastive auxiliary loss.

    Total loss = MSE_loss + lambda_contrastive * contrastive_loss

    The contrastive loss encourages:
    - Similar performances (similar labels) to have similar embeddings
    - Different performances to have different embeddings

    This is particularly useful for:
    - Learning more discriminative representations
    - Improving generalization by enforcing structure in embedding space
    - Better handling of ambiguous cases (performances with similar quality)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        contrastive_lambda: float = 0.1,
        temperature: float = 0.07,
        similarity_threshold: float = 0.8,
        pooling: str = "attention",
        contrastive_type: str = "threshold",  # "threshold" or "supervised"
        max_epochs: int = 200,
    ):
        """Initialize contrastive auxiliary model.

        Args:
            input_dim: Input embedding dimension (1024 for MERT).
            hidden_dim: Hidden layer dimension.
            num_labels: Number of output dimensions (19 for PercePiano).
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            contrastive_lambda: Weight for contrastive loss component.
            temperature: Temperature for contrastive loss softmax.
            similarity_threshold: Threshold for positive pairs (threshold mode).
            pooling: Pooling strategy ("mean" or "attention").
            contrastive_type: Type of contrastive loss ("threshold" or "supervised").
            max_epochs: Maximum training epochs.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.contrastive_lambda = contrastive_lambda
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.pooling = pooling
        self.contrastive_type = contrastive_type
        self.max_epochs = max_epochs

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Projection head for contrastive learning
        # Projects pooled representation to a lower-dimensional space
        # for contrastive objective (doesn't affect main task)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # MLP head for main prediction task
        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
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

    def forward(self, x, mask=None, lengths=None):
        """Forward pass for inference (predictions only)."""
        pooled = self.pool(x, mask)
        return self.clf(pooled)

    def get_contrastive_loss(self, pooled, labels):
        """Compute contrastive loss on projected embeddings."""
        # Project to contrastive space
        projected = self.projection(pooled)
        projected = F.normalize(projected, dim=-1)

        if self.contrastive_type == "supervised":
            return supervised_contrastive_loss(
                projected, labels, temperature=self.temperature
            )
        else:  # threshold
            return contrastive_auxiliary_loss(
                projected,
                labels,
                temperature=self.temperature,
                similarity_threshold=self.similarity_threshold,
            )

    def training_step(self, batch, idx):
        # Get pooled representation
        pooled = self.pool(batch["embeddings"], batch.get("attention_mask"))

        # Main task prediction
        pred = self.clf(pooled)
        mse_loss = self.mse_loss(pred, batch["labels"])

        # Contrastive auxiliary loss
        contrastive = self.get_contrastive_loss(pooled, batch["labels"])

        # Total loss
        total_loss = mse_loss + self.contrastive_lambda * contrastive

        # Logging
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("mse_loss", mse_loss)
        self.log("contrastive_loss", contrastive)

        return total_loss

    def validation_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        self.log("val_loss", self.mse_loss(pred, batch["labels"]), prog_bar=True)
        self.val_outputs.append({"p": pred.cpu(), "l": batch["labels"].cpu()})

    def on_validation_epoch_end(self):
        if self.val_outputs:
            p = torch.cat([x["p"] for x in self.val_outputs]).numpy()
            l = torch.cat([x["l"] for x in self.val_outputs]).numpy()
            self.log("val_r2", r2_score(l, p), prog_bar=True)
            self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


class ContrastiveWarmupModel(pl.LightningModule):
    """Model with contrastive pretraining warmup phase.

    Starts with higher contrastive weight that decreases during training,
    allowing the model to first learn good representations before focusing
    on the main prediction task.

    lambda(epoch) = lambda_start * (lambda_end / lambda_start) ^ (epoch / max_epochs)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        contrastive_lambda_start: float = 0.5,
        contrastive_lambda_end: float = 0.05,
        temperature: float = 0.07,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.lambda_start = contrastive_lambda_start
        self.lambda_end = contrastive_lambda_end
        self.temperature = temperature
        self.pooling = pooling
        self.max_epochs = max_epochs

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # MLP head
        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
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

    def forward(self, x, mask=None, lengths=None):
        pooled = self.pool(x, mask)
        return self.clf(pooled)

    def get_current_lambda(self):
        """Get contrastive lambda for current epoch (exponential decay)."""
        if self.current_epoch >= self.max_epochs:
            return self.lambda_end

        # Exponential interpolation
        progress = self.current_epoch / self.max_epochs
        ratio = self.lambda_end / (self.lambda_start + 1e-8)
        return self.lambda_start * (ratio ** progress)

    def training_step(self, batch, idx):
        pooled = self.pool(batch["embeddings"], batch.get("attention_mask"))
        pred = self.clf(pooled)
        mse_loss = self.mse_loss(pred, batch["labels"])

        # Contrastive loss with current lambda
        projected = self.projection(pooled)
        projected = F.normalize(projected, dim=-1)
        contrastive = supervised_contrastive_loss(
            projected, batch["labels"], temperature=self.temperature
        )

        current_lambda = self.get_current_lambda()
        total_loss = mse_loss + current_lambda * contrastive

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("mse_loss", mse_loss)
        self.log("contrastive_loss", contrastive)
        self.log("contrastive_lambda", current_lambda)

        return total_loss

    def validation_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        self.log("val_loss", self.mse_loss(pred, batch["labels"]), prog_bar=True)
        self.val_outputs.append({"p": pred.cpu(), "l": batch["labels"].cpu()})

    def on_validation_epoch_end(self):
        if self.val_outputs:
            p = torch.cat([x["p"] for x in self.val_outputs]).numpy()
            l = torch.cat([x["l"] for x in self.val_outputs]).numpy()
            self.log("val_r2", r2_score(l, p), prog_bar=True)
            self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
