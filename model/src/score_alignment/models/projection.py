"""Learned projection models for embedding space alignment.

These models learn to project MuQ embeddings into a space where
soft-DTW alignment is more accurate.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..alignment.soft_dtw import SoftDTWDivergence
from ..config import ProjectionConfig, TrainingConfig


class ProjectionMLP(nn.Module):
    """Multi-layer projection network for embedding transformation.

    Projects embeddings from input_dim to output_dim through hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize projection MLP.

        Args:
            input_dim: Input embedding dimension (MuQ hidden size).
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of layers (minimum 2).
            dropout: Dropout probability.
            activation: Activation function ("gelu", "relu").
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
        ])

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])

        # Final layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings.

        Args:
            x: Input embeddings of shape [..., input_dim].

        Returns:
            Projected embeddings of shape [..., output_dim].
        """
        return self.mlp(x)


class AlignmentProjectionModel(pl.LightningModule):
    """PyTorch Lightning model for learning alignment-optimized projections.

    Trains a projection network to minimize soft-DTW divergence between
    projected score and performance embeddings.
    """

    def __init__(
        self,
        projection_config: Optional[ProjectionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        # Direct params for simpler instantiation
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        soft_dtw_gamma: float = 1.0,
        distance_metric: str = "cosine",
        max_epochs: int = 100,
    ):
        """Initialize alignment projection model.

        Can be initialized with config objects or direct parameters.

        Args:
            projection_config: Projection network configuration.
            training_config: Training configuration.
            input_dim: Input dimension if not using config.
            hidden_dim: Hidden dimension if not using config.
            output_dim: Output dimension if not using config.
            num_layers: Number of layers if not using config.
            dropout: Dropout probability if not using config.
            learning_rate: Learning rate if not using config.
            weight_decay: Weight decay if not using config.
            soft_dtw_gamma: Soft-DTW gamma if not using config.
            distance_metric: Distance metric if not using config.
            max_epochs: Max epochs for scheduler if not using config.
        """
        super().__init__()
        self.save_hyperparameters()

        # Use configs if provided, otherwise use direct params
        if projection_config:
            input_dim = projection_config.input_dim
            hidden_dim = projection_config.hidden_dim
            output_dim = projection_config.output_dim
            num_layers = projection_config.num_layers
            dropout = projection_config.dropout

        if training_config:
            learning_rate = training_config.learning_rate
            weight_decay = training_config.weight_decay
            soft_dtw_gamma = training_config.soft_dtw_gamma
            distance_metric = training_config.distance_metric
            max_epochs = training_config.max_epochs

        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

        # Build projection network
        self.projection = ProjectionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Soft-DTW divergence loss
        self.sdtw_loss = SoftDTWDivergence(
            gamma=soft_dtw_gamma,
            distance=distance_metric,
        )

        # Tracking
        self.val_losses = []

    def forward(
        self,
        score_emb: torch.Tensor,
        perf_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both score and performance embeddings.

        Args:
            score_emb: Score embeddings of shape [B, T1, D] or [T1, D].
            perf_emb: Performance embeddings of shape [B, T2, D] or [T2, D].

        Returns:
            Tuple of (projected_score, projected_perf).
        """
        proj_score = self.projection(score_emb)
        proj_perf = self.projection(perf_emb)
        return proj_score, proj_perf

    def _compute_sdtw_loss(self, batch):
        """Compute soft-DTW divergence loss per-sample to limit memory.

        Soft-DTW allocates [T1, T2] distance and DP matrices. Processing the
        full padded batch ([B, T_max, T_max] * 3 for divergence) easily OOMs
        on MPS/limited-VRAM GPUs. Instead, we project in batch (cheap MLP)
        then loop over samples using their actual (unpadded) lengths.
        """
        score_emb = batch["score_embeddings"]  # [B, T1_padded, D]
        perf_emb = batch["perf_embeddings"]  # [B, T2_padded, D]
        score_lengths = batch["score_lengths"]  # [B]
        perf_lengths = batch["perf_lengths"]  # [B]

        # Project embeddings (batched - efficient for MLP)
        proj_score, proj_perf = self(score_emb, perf_emb)

        # Compute loss per sample using actual lengths
        losses = []
        for i in range(proj_score.shape[0]):
            s_len = score_lengths[i].item()
            p_len = perf_lengths[i].item()
            s = proj_score[i, :s_len].unsqueeze(0)  # [1, T1_actual, D]
            p = proj_perf[i, :p_len].unsqueeze(0)  # [1, T2_actual, D]
            losses.append(self.sdtw_loss(s, p))

        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        """Training step: minimize soft-DTW divergence."""
        loss = self._compute_sdtw_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self._compute_sdtw_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val_loss_epoch", avg_loss)
            self.val_losses.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_projected_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Get projected embeddings for inference.

        Args:
            embeddings: Input embeddings of shape [..., input_dim].

        Returns:
            Projected embeddings of shape [..., output_dim].
        """
        self.eval()
        with torch.no_grad():
            return self.projection(embeddings)
