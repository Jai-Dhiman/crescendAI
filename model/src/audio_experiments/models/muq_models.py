"""MuQ models for piano performance evaluation.

Models using MuQ (Music Understanding Quantized) embeddings from ByteDance/OpenMuQ.
MuQ is similar to MERT but uses different training objectives, potentially
capturing complementary musical features.

MuQ hidden size: 1024 (same as MERT-330M)
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class MuQBaseModel(pl.LightningModule):
    """Base model for MuQ embeddings with configurable pooling.

    Same architecture as BaseMERTModel but designed for MuQ embeddings.
    Supports mean, max, and attention pooling strategies.
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

        # Attention pooling
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

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
        """Pool frame embeddings to sequence representation."""
        if self.pooling == "mean":
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(1)[0]

        elif self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)

        return x.mean(1)

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


class MuQStatsModel(pl.LightningModule):
    """MuQ model with statistical pooling (mean + std).

    Concatenates mean and standard deviation across frames to preserve
    temporal variability information. Results in 2x input dimension.
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

        # MLP head
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
            # Masked std
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


class MERTMuQEnsemble(pl.LightningModule):
    """Late fusion ensemble of MERT and MuQ predictions.

    Takes pre-computed predictions from both MERT and MuQ models and
    combines them via weighted averaging. This is a "prediction-level"
    ensemble that doesn't require joint training.

    For training, this model learns separate heads for MERT and MuQ
    embeddings and averages their predictions.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pooling: str = "attention",
        fusion_weight: float = 0.5,  # Weight for MERT predictions
        max_epochs: int = 200,
    ):
        """Initialize ensemble model.

        Args:
            input_dim: Embedding dimension (same for MERT and MuQ).
            hidden_dim: Hidden layer dimension.
            num_labels: Number of output dimensions (19 for PercePiano).
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            pooling: Pooling strategy ("mean" or "attention").
            fusion_weight: Weight for MERT predictions (1-weight for MuQ).
            max_epochs: Maximum training epochs.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.pooling = pooling
        self.fusion_weight = fusion_weight
        self.max_epochs = max_epochs

        # Shared attention pooling (applied separately to each modality)
        if pooling == "attention":
            self.mert_attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )
            self.muq_attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Separate heads for each modality
        self.mert_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.muq_head = nn.Sequential(
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

    def pool(self, x, attn_module, mask=None):
        """Pool embeddings using attention or mean pooling."""
        if self.pooling == "attention":
            scores = attn_module(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, mert_emb, muq_emb, mert_mask=None, muq_mask=None):
        """Forward pass combining MERT and MuQ embeddings.

        Args:
            mert_emb: MERT embeddings [B, T, D].
            muq_emb: MuQ embeddings [B, T, D].
            mert_mask: Attention mask for MERT [B, T].
            muq_mask: Attention mask for MuQ [B, T].

        Returns:
            Combined predictions [B, num_labels].
        """
        # Pool MERT
        attn_mod = self.mert_attn if self.pooling == "attention" else None
        mert_pooled = self.pool(mert_emb, attn_mod, mert_mask)
        mert_pred = self.mert_head(mert_pooled)

        # Pool MuQ
        attn_mod = self.muq_attn if self.pooling == "attention" else None
        muq_pooled = self.pool(muq_emb, attn_mod, muq_mask)
        muq_pred = self.muq_head(muq_pooled)

        # Weighted average
        combined = self.fusion_weight * mert_pred + (1 - self.fusion_weight) * muq_pred
        return combined

    def training_step(self, batch, idx):
        """Training step expecting both MERT and MuQ embeddings in batch."""
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
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


class MERTMuQConcatModel(pl.LightningModule):
    """Early fusion model that concatenates MERT and MuQ embeddings.

    Concatenates pooled representations from both models before the
    classification head. This allows the model to learn cross-modal
    interactions.

    Supports different dimensions for MERT and MuQ embeddings. Both are
    projected to a shared dimension before concatenation.
    """

    def __init__(
        self,
        mert_dim: int = 1024,
        muq_dim: int = 1024,
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
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.mert_dim = mert_dim
        self.muq_dim = muq_dim

        # Separate attention modules for each modality (use respective dims)
        if pooling == "attention":
            self.mert_attn = nn.Sequential(
                nn.Linear(mert_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )
            self.muq_attn = nn.Sequential(
                nn.Linear(muq_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Project each modality to hidden_dim before concatenation
        self.mert_proj = nn.Linear(mert_dim, hidden_dim)
        self.muq_proj = nn.Linear(muq_dim, hidden_dim)

        # Projection after concatenation (2 * hidden_dim -> hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # MLP head (input is hidden_dim from projection)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

    def pool(self, x, attn_module, mask=None):
        """Pool embeddings using attention or mean pooling."""
        if self.pooling == "attention":
            scores = attn_module(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, mert_emb, muq_emb, mert_mask=None, muq_mask=None):
        # Pool each modality
        attn_mod = self.mert_attn if self.pooling == "attention" else None
        mert_pooled = self.pool(mert_emb, attn_mod, mert_mask)

        attn_mod = self.muq_attn if self.pooling == "attention" else None
        muq_pooled = self.pool(muq_emb, attn_mod, muq_mask)

        # Project each modality to shared dimension
        mert_proj = self.mert_proj(mert_pooled)
        muq_proj = self.muq_proj(muq_pooled)

        # Concatenate and project to hidden_dim
        concat = torch.cat([mert_proj, muq_proj], dim=-1)
        projected = self.proj(concat)

        return self.clf(projected)

    def training_step(self, batch, idx):
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
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


class AsymmetricGatedFusion(pl.LightningModule):
    """Asymmetric fusion with per-dimension gating for MERT and MuQ.

    Uses asymmetric projections that respect the information difference between
    modalities (MERT 6144-dim vs MuQ 1024-dim). Learns per-dimension gates to
    softly route between modalities for each of the 19 performance dimensions.

    Architecture:
    - MERT: 6144 -> 768 -> 512 (2-stage projection preserves more detail)
    - MuQ: 1024 -> 512 (single projection)
    - Per-dimension gating learns which modality matters for each output
    """

    def __init__(
        self,
        mert_dim: int = 6144,
        muq_dim: int = 1024,
        mert_hidden: int = 768,
        shared_dim: int = 512,
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
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Attention pooling modules
        if pooling == "attention":
            self.mert_attn = nn.Sequential(
                nn.Linear(mert_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )
            self.muq_attn = nn.Sequential(
                nn.Linear(muq_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Asymmetric projections
        # MERT: 2-stage to preserve more acoustic detail from 6144-dim
        self.mert_proj = nn.Sequential(
            nn.Linear(mert_dim, mert_hidden),
            nn.LayerNorm(mert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mert_hidden, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # MuQ: single-stage projection
        self.muq_proj = nn.Sequential(
            nn.Linear(muq_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-dimension gating network
        # Takes concatenated projections, outputs gate weights for each label
        self.gate_net = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, num_labels),
            nn.Sigmoid(),
        )

        # Per-dimension prediction heads
        # Each head predicts one label from the gated representation
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_labels)
        ])

        self.mse_loss = nn.MSELoss()
        self.val_outputs = []

    def pool(self, x, attn_module, mask=None):
        """Pool sequence embeddings using attention or mean pooling."""
        if self.pooling == "attention":
            scores = attn_module(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, mert_emb, muq_emb, mert_mask=None, muq_mask=None):
        # Pool sequences to fixed-size vectors
        attn_mod = self.mert_attn if self.pooling == "attention" else None
        mert_pooled = self.pool(mert_emb, attn_mod, mert_mask)

        attn_mod = self.muq_attn if self.pooling == "attention" else None
        muq_pooled = self.pool(muq_emb, attn_mod, muq_mask)

        # Asymmetric projections
        mert_proj = self.mert_proj(mert_pooled)  # [B, shared_dim]
        muq_proj = self.muq_proj(muq_pooled)     # [B, shared_dim]

        # Compute per-dimension gates
        combined = torch.cat([mert_proj, muq_proj], dim=-1)  # [B, shared_dim*2]
        gates = self.gate_net(combined)  # [B, num_labels]

        # Apply per-dimension gated fusion and predict
        outputs = []
        for i, head in enumerate(self.heads):
            gate = gates[:, i:i+1]  # [B, 1]
            # Soft routing: gate * mert + (1-gate) * muq
            gated = gate * mert_proj + (1 - gate) * muq_proj  # [B, shared_dim]
            out = head(gated)  # [B, 1]
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # [B, num_labels]

    def training_step(self, batch, idx):
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
        loss = self.mse_loss(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)

        # Log gate statistics periodically
        if self.global_step % 100 == 0:
            with torch.no_grad():
                mert_pooled = self.pool(
                    batch["mert_embeddings"],
                    self.mert_attn if self.pooling == "attention" else None,
                    batch.get("mert_mask")
                )
                muq_pooled = self.pool(
                    batch["muq_embeddings"],
                    self.muq_attn if self.pooling == "attention" else None,
                    batch.get("muq_mask")
                )
                mert_proj = self.mert_proj(mert_pooled)
                muq_proj = self.muq_proj(muq_pooled)
                combined = torch.cat([mert_proj, muq_proj], dim=-1)
                gates = self.gate_net(combined)
                self.log("mert_gate_mean", gates.mean())
                self.log("mert_gate_std", gates.std())

        return loss

    def validation_step(self, batch, idx):
        pred = self(
            batch["mert_embeddings"],
            batch["muq_embeddings"],
            batch.get("mert_mask"),
            batch.get("muq_mask"),
        )
        self.log("val_loss", self.mse_loss(pred, batch["labels"]), prog_bar=True)
        self.val_outputs.append({"p": pred.cpu(), "l": batch["labels"].cpu()})

    def on_validation_epoch_end(self):
        if self.val_outputs:
            p = torch.cat([x["p"] for x in self.val_outputs]).numpy()
            l = torch.cat([x["l"] for x in self.val_outputs]).numpy()
            self.log("val_r2", r2_score(l, p), prog_bar=True)
            self.val_outputs.clear()

    def get_learned_gates(self, mert_emb, muq_emb, mert_mask=None, muq_mask=None):
        """Return per-dimension gate values for interpretability.

        Returns dict with gate values per dimension (higher = more MERT).
        """
        with torch.no_grad():
            attn_mod = self.mert_attn if self.pooling == "attention" else None
            mert_pooled = self.pool(mert_emb, attn_mod, mert_mask)

            attn_mod = self.muq_attn if self.pooling == "attention" else None
            muq_pooled = self.pool(muq_emb, attn_mod, muq_mask)

            mert_proj = self.mert_proj(mert_pooled)
            muq_proj = self.muq_proj(muq_pooled)
            combined = torch.cat([mert_proj, muq_proj], dim=-1)
            gates = self.gate_net(combined)

            return {
                "gates": gates.cpu().numpy(),
                "mert_weight_per_dim": gates.mean(0).cpu().numpy(),
            }

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
