"""Learnable fusion models for multimodal combination.

These models are trained end-to-end with specialized training objectives:
- ModalityDropoutFusion: Randomly drops modalities during training
- OrthogonalityFusion: Penalizes correlated representations
- ResidualFusion: Symbolic predicts audio's residual errors
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class ModalityDropoutFusion(pl.LightningModule):
    """Fusion model with modality dropout during training.

    Randomly drops entire modality branches during training, forcing each
    encoder to capture unique information rather than relying on shared
    patterns. At test time, uses both modalities.

    Reference: Apple ML Research showed 7.4% improvement in robustness.
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        symbolic_dim: int = 256,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        modality_dropout: float = 0.25,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.modality_dropout = modality_dropout

        # Audio projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Symbolic projection
        self.symbolic_proj = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def forward(self, audio_repr, symbolic_repr):
        """Forward pass with optional modality dropout during training."""
        audio_proj = self.audio_proj(audio_repr)
        symbolic_proj = self.symbolic_proj(symbolic_repr)

        if self.training and self.modality_dropout > 0:
            # Randomly drop one modality (never both)
            drop_audio = torch.rand(1).item() < self.modality_dropout
            drop_symbolic = torch.rand(1).item() < self.modality_dropout

            # Never drop both
            if drop_audio and drop_symbolic:
                drop_audio = torch.rand(1).item() < 0.5
                drop_symbolic = not drop_audio

            if drop_audio:
                audio_proj = torch.zeros_like(audio_proj)
            if drop_symbolic:
                symbolic_proj = torch.zeros_like(symbolic_proj)

        # Concatenate and fuse
        fused = torch.cat([audio_proj, symbolic_proj], dim=-1)
        return self.fusion_head(fused)

    def training_step(self, batch, idx):
        pred = self(batch["audio_repr"], batch["symbolic_repr"])
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["audio_repr"], batch["symbolic_repr"])
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


class OrthogonalityFusion(pl.LightningModule):
    """Fusion model with orthogonality loss to encourage complementary representations.

    Adds a penalty for correlated audio and symbolic representations, forcing
    each modality to capture unique information.

    Loss = MSE + lambda * |cosine_similarity(audio_repr, symbolic_repr)|.mean()
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        symbolic_dim: int = 256,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        orthogonality_lambda: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.orthogonality_lambda = orthogonality_lambda

        # Audio projection to shared dimension
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Symbolic projection to shared dimension
        self.symbolic_proj = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.val_outputs = []

    def forward(self, audio_repr, symbolic_repr):
        audio_proj = self.audio_proj(audio_repr)
        symbolic_proj = self.symbolic_proj(symbolic_repr)
        fused = torch.cat([audio_proj, symbolic_proj], dim=-1)
        return self.fusion_head(fused), audio_proj, symbolic_proj

    def compute_orthogonality_loss(self, audio_proj, symbolic_proj):
        """Penalize correlation between audio and symbolic representations."""
        cos_sim = self.cos_sim(audio_proj, symbolic_proj)
        return torch.abs(cos_sim).mean()

    def training_step(self, batch, idx):
        pred, audio_proj, symbolic_proj = self(
            batch["audio_repr"], batch["symbolic_repr"]
        )
        mse_loss = self.mse_loss(pred, batch["labels"])
        orth_loss = self.compute_orthogonality_loss(audio_proj, symbolic_proj)
        total_loss = mse_loss + self.orthogonality_lambda * orth_loss

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("mse_loss", mse_loss)
        self.log("orth_loss", orth_loss)

        return total_loss

    def validation_step(self, batch, idx):
        pred, _, _ = self(batch["audio_repr"], batch["symbolic_repr"])
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


class ResidualFusion(pl.LightningModule):
    """Residual fusion where symbolic predicts audio's errors.

    Instead of directly predicting targets, the symbolic branch learns to
    predict the residual errors from the audio model. Final prediction is:
    audio_pred + symbolic_residual

    This explicitly targets complementary information.
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        symbolic_dim: int = 256,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

        # Audio head (predicts targets directly)
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        # Symbolic residual head (predicts errors, no sigmoid - can be negative)
        self.symbolic_residual = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Tanh(),  # Residuals bounded to [-1, 1]
        )

        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def forward(self, audio_repr, symbolic_repr):
        audio_pred = self.audio_head(audio_repr)
        residual = self.symbolic_residual(symbolic_repr)

        # Scale residual and add to audio prediction
        scaled_residual = residual * self.residual_scale
        fused_pred = torch.clamp(audio_pred + scaled_residual, 0, 1)

        return fused_pred, audio_pred, residual

    def training_step(self, batch, idx):
        fused_pred, audio_pred, _ = self(
            batch["audio_repr"], batch["symbolic_repr"]
        )

        # Main loss on fused prediction
        fused_loss = self.mse_loss(fused_pred, batch["labels"])

        # Auxiliary loss on audio prediction (encourages good audio baseline)
        audio_loss = self.mse_loss(audio_pred, batch["labels"])

        total_loss = fused_loss + 0.5 * audio_loss

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("fused_loss", fused_loss)
        self.log("audio_loss", audio_loss)
        self.log("residual_scale", self.residual_scale)

        return total_loss

    def validation_step(self, batch, idx):
        fused_pred, _, _ = self(batch["audio_repr"], batch["symbolic_repr"])
        self.log("val_loss", self.mse_loss(fused_pred, batch["labels"]), prog_bar=True)
        self.val_outputs.append({"p": fused_pred.cpu(), "l": batch["labels"].cpu()})

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


class DimensionWeightedFusion(pl.LightningModule):
    """Learnable per-dimension fusion weights.

    Learns soft routing between audio and symbolic for each of the 19
    dimensions. The weights are learned end-to-end.
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        symbolic_dim: int = 256,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Audio head
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        # Symbolic head
        self.symbolic_head = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        # Learnable per-dimension weights (logits, will be softmaxed)
        # Shape: [19, 2] - for each dimension, weight for [audio, symbolic]
        self.dim_weight_logits = nn.Parameter(torch.zeros(num_labels, 2))

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def forward(self, audio_repr, symbolic_repr):
        audio_pred = self.audio_head(audio_repr)
        symbolic_pred = self.symbolic_head(symbolic_repr)

        # Get softmax weights per dimension
        weights = torch.softmax(self.dim_weight_logits, dim=1)  # [19, 2]

        # Weighted combination: [B, 19]
        fused = weights[:, 0] * audio_pred + weights[:, 1] * symbolic_pred

        return fused, audio_pred, symbolic_pred

    def training_step(self, batch, idx):
        fused_pred, _, _ = self(batch["audio_repr"], batch["symbolic_repr"])
        loss = self.mse_loss(fused_pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)

        # Log weight statistics
        if self.global_step % 50 == 0:
            weights = torch.softmax(self.dim_weight_logits, dim=1)
            self.log("audio_weight_mean", weights[:, 0].mean())

        return loss

    def validation_step(self, batch, idx):
        fused_pred, _, _ = self(batch["audio_repr"], batch["symbolic_repr"])
        self.log("val_loss", self.mse_loss(fused_pred, batch["labels"]), prog_bar=True)
        self.val_outputs.append({"p": fused_pred.cpu(), "l": batch["labels"].cpu()})

    def on_validation_epoch_end(self):
        if self.val_outputs:
            p = torch.cat([x["p"] for x in self.val_outputs]).numpy()
            l = torch.cat([x["l"] for x in self.val_outputs]).numpy()
            self.log("val_r2", r2_score(l, p), prog_bar=True)
            self.val_outputs.clear()

    def get_learned_weights(self):
        """Return the learned per-dimension weights."""
        with torch.no_grad():
            weights = torch.softmax(self.dim_weight_logits, dim=1)
            return {
                "audio": weights[:, 0].cpu().numpy(),
                "symbolic": weights[:, 1].cpu().numpy(),
            }

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
