"""Phase 3 model architectures for ISMIR experiments.

New architectures based on research recommendations:
- StatsPoolingModel: Statistical pooling (mean+std+min+max)
- UncertaintyWeightedModel: Learnable per-dimension uncertainty weighting
- DimensionSpecificModel: Separate heads for timing vs other dimensions
- TransformerPoolingModel: 2-layer transformer before pooling
- MultiScalePoolingModel: Pool at multiple temporal resolutions
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import ccc_loss


class StatsPoolingModel(pl.LightningModule):
    """MERT model with statistical pooling (mean + std + min + max).

    Concatenates 4 statistics across frames, resulting in 4x input dimension.
    This preserves temporal variability information that mean pooling discards.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pooling_stats: str = "mean_std",  # "mean_std" or "mean_std_min_max"
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.pooling_stats = pooling_stats
        self.max_epochs = max_epochs

        # Determine pooled dimension
        if pooling_stats == "mean_std":
            pooled_dim = input_dim * 2  # 2048
        else:  # mean_std_min_max
            pooled_dim = input_dim * 4  # 4096

        # MLP head (larger first layer to handle increased input)
        self.clf = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
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
        """Statistical pooling with masking support."""
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            # Masked mean
            mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
            # Masked std (unbiased=False for stability with small counts)
            diff = (x - mean.unsqueeze(1)) * m
            var = (diff ** 2).sum(1) / m.sum(1).clamp(min=1)
            std = (var + 1e-8).sqrt()

            if self.pooling_stats == "mean_std":
                return torch.cat([mean, std], dim=-1)

            # Masked min/max
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float("inf"))
            min_val = x_masked.min(1)[0]
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            max_val = x_masked.max(1)[0]
            return torch.cat([mean, std, min_val, max_val], dim=-1)
        else:
            mean = x.mean(1)
            std = x.std(1, unbiased=False)
            if self.pooling_stats == "mean_std":
                return torch.cat([mean, std], dim=-1)
            min_val = x.min(1)[0]
            max_val = x.max(1)[0]
            return torch.cat([mean, std, min_val, max_val], dim=-1)

    def forward(self, x, mask=None, lengths=None):
        pooled = self.pool(x, mask)
        return self.clf(pooled)

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

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


class UncertaintyWeightedModel(pl.LightningModule):
    """MERT model with uncertainty-weighted multi-task loss.

    Learns per-dimension uncertainty parameters (log_sigma) that automatically
    balance the loss across dimensions with different scales/difficulties.

    Loss = sum_t [ (1 / (2 * sigma_t^2)) * MSE_t + log(sigma_t) ]

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics" (CVPR 2018)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pooling: str = "mean",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Learnable log(sigma) for each dimension (initialized to 0 = sigma=1)
        self.log_sigma = nn.Parameter(torch.zeros(num_labels))

        # Attention pooling (optional)
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
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

        self.mse_loss = nn.MSELoss(reduction="none")
        self.val_outputs = []

    def pool(self, x, mask=None):
        if self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:  # mean pooling
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, x, mask=None, lengths=None):
        pooled = self.pool(x, mask)
        return self.clf(pooled)

    def compute_uncertainty_loss(self, pred, target):
        """Uncertainty-weighted loss with learnable per-dimension weights."""
        # Per-dimension MSE: [batch, num_labels]
        mse_per_dim = self.mse_loss(pred, target)
        # Mean over batch: [num_labels]
        mse_per_dim = mse_per_dim.mean(dim=0)

        # Uncertainty weighting: (1 / (2 * sigma^2)) * MSE + log(sigma)
        # Using log_sigma for numerical stability: sigma = exp(log_sigma)
        precision = torch.exp(-2 * self.log_sigma)  # 1 / sigma^2
        loss = 0.5 * precision * mse_per_dim + self.log_sigma

        return loss.sum()

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.compute_uncertainty_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)

        # Log learned uncertainties (every 50 steps)
        if self.global_step % 50 == 0:
            sigma = torch.exp(self.log_sigma)
            self.log("sigma_mean", sigma.mean())
            self.log("sigma_std", sigma.std())

        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        # Use standard MSE for validation comparability
        mse = self.mse_loss(pred, batch["labels"]).mean()
        self.log("val_loss", mse, prog_bar=True)
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

    def get_learned_weights(self):
        """Return the learned dimension weights (inverse uncertainty)."""
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma)
            weights = 1.0 / (sigma ** 2)
            weights = weights / weights.sum()  # Normalize
            return weights.cpu().numpy()


class DimensionSpecificModel(pl.LightningModule):
    """Model with dimension-specific heads for timing vs. other dimensions.

    Uses BiLSTM on frame sequences for timing-related dimensions (which need
    temporal structure), and mean-pooled features for other dimensions.

    Timing dimensions (indices 0, 10): timing, tempo
    Other dimensions: articulation, pedal, timbre, dynamics, emotion, interpretation
    """

    # Timing-related dimension indices
    TIMING_DIMS = [0, 10]  # timing, tempo

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lstm_hidden: int = 256,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

        self.timing_dims = self.TIMING_DIMS
        self.other_dims = [i for i in range(num_labels) if i not in self.timing_dims]

        # BiLSTM for timing dimensions
        self.timing_lstm = nn.LSTM(
            input_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        self.timing_attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.timing_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(self.timing_dims)),
            nn.Sigmoid(),
        )

        # MLP for other dimensions (uses mean pooling)
        self.other_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.other_dims)),
            nn.Sigmoid(),
        )

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def forward(self, x, mask=None, lengths=None):
        batch_size = x.shape[0]

        # Timing branch: BiLSTM with attention
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.timing_lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.timing_lstm(x)

        # Attention pooling over LSTM outputs
        attn_scores = self.timing_attn(lstm_out).squeeze(-1)
        if mask is not None:
            if mask.shape[1] > lstm_out.shape[1]:
                mask = mask[:, :lstm_out.shape[1]]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        timing_repr = (lstm_out * attn_weights).sum(1)
        timing_pred = self.timing_head(timing_repr)  # [B, 2]

        # Other branch: mean pooling
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            other_repr = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            other_repr = x.mean(1)
        other_pred = self.other_head(other_repr)  # [B, 17]

        # Combine predictions in correct order
        output = torch.zeros(batch_size, 19, device=x.device)
        for i, dim_idx in enumerate(self.timing_dims):
            output[:, dim_idx] = timing_pred[:, i]
        for i, dim_idx in enumerate(self.other_dims):
            output[:, dim_idx] = other_pred[:, i]

        return output

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"), batch.get("lengths"))
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"), batch.get("lengths"))
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


class TransformerPoolingModel(pl.LightningModule):
    """MERT model with lightweight transformer before pooling.

    Adds a 2-layer transformer encoder on top of MERT frames to model
    temporal dependencies before aggregation. Uses ALiBi-style relative
    position bias for variable-length sequences.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_heads: int = 8,
        num_layers: int = 2,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.pooling = pooling

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
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

    def forward(self, x, mask=None, lengths=None):
        # Create attention mask for transformer (True = ignore)
        if mask is not None:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Attention pooling
        if self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (x * w).sum(1)
        else:  # mean
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                pooled = x.mean(1)

        return self.clf(pooled)

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

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


class MultiScalePoolingModel(pl.LightningModule):
    """MERT model with multi-scale temporal pooling.

    Pools at multiple temporal resolutions (4, 8, 16, 32 frames) to capture
    hierarchical structure (beat, measure, phrase). Each scale is mean-pooled,
    then results are concatenated and projected.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scales: tuple = (4, 8, 16, 32),
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.scales = scales

        # Each scale produces input_dim features, then we pool across scales
        # Plus global mean gives len(scales) + 1 = 5 representations
        num_scales = len(scales) + 1  # +1 for global
        pooled_dim = input_dim * num_scales

        # Projection to reduce dimensionality
        self.scale_proj = nn.Linear(pooled_dim, input_dim)

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

    def multi_scale_pool(self, x, mask=None):
        """Pool at multiple temporal scales."""
        batch_size, seq_len, dim = x.shape
        pooled_features = []

        # Global mean pool
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            global_pool = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            global_pool = x.mean(1)
        pooled_features.append(global_pool)

        # Multi-scale pooling
        for scale in self.scales:
            if seq_len < scale:
                # If sequence shorter than scale, use global pool
                pooled_features.append(global_pool)
            else:
                # Reshape to [batch, num_windows, scale, dim] and pool
                num_windows = seq_len // scale
                truncated = x[:, :num_windows * scale, :]
                reshaped = truncated.view(batch_size, num_windows, scale, dim)
                # Mean over window (scale dimension), then mean over windows
                window_means = reshaped.mean(dim=2)  # [B, num_windows, dim]
                scale_pool = window_means.mean(dim=1)  # [B, dim]
                pooled_features.append(scale_pool)

        # Concatenate all scales
        return torch.cat(pooled_features, dim=-1)

    def forward(self, x, mask=None, lengths=None):
        pooled = self.multi_scale_pool(x, mask)
        projected = self.scale_proj(pooled)
        return self.clf(projected)

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

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


class MultiLayerMERTModel(pl.LightningModule):
    """Model for concatenated multi-layer MERT embeddings.

    Takes pre-concatenated embeddings from multiple MERT layers (e.g., [6,9,12])
    with input_dim = 1024 * num_layers = 3072.
    """

    def __init__(
        self,
        input_dim: int = 3072,  # 1024 * 3 layers
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
        self.pooling = pooling

        # Layer-wise projection (reduce each 1024-dim layer to 256, then concat)
        self.layer_proj = nn.Linear(input_dim, 1024)

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # MLP head
        self.clf = nn.Sequential(
            nn.Linear(1024, hidden_dim),
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
        else:  # mean
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, x, mask=None, lengths=None):
        # Project concatenated layers
        x = self.layer_proj(x)
        pooled = self.pool(x, mask)
        return self.clf(pooled)

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

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
