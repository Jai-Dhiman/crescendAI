"""Cross-attention fusion model for multimodal piano performance evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..constants import BASE_CONFIG, PERCEPIANO_DIMENSIONS


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention where queries attend to key-value pairs."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, query_len, query_dim) or (batch, query_dim)
            key_value: (batch, kv_len, kv_dim)
            kv_mask: (batch, kv_len) - True for valid positions

        Returns:
            (batch, query_len, hidden_dim) or (batch, hidden_dim)
        """
        # Handle single query vector
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]

        # Project
        Q = self.q_proj(query)  # (batch, query_len, hidden_dim)
        K = self.k_proj(key_value)  # (batch, kv_len, hidden_dim)
        V = self.v_proj(key_value)  # (batch, kv_len, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, heads, q_len, kv_len)

        # Apply mask if provided
        if kv_mask is not None:
            # Expand mask for heads and query positions
            mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, kv_len)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, heads, q_len, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        out = self.out_proj(out)

        if squeeze_output:
            out = out.squeeze(1)

        return out


class CrossAttentionFusion(pl.LightningModule):
    """Cross-attention fusion model: MERT embeddings attend to symbolic features.

    Architecture:
    1. Pool MERT sequence -> query vector (1024-dim)
    2. Cross-attend query to symbolic sequence features
    3. Concatenate pooled MERT + attended symbolic
    4. MLP head -> 19 dimension predictions
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        symbolic_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 4,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        audio_pooling: str = "mean",
    ):
        """
        Args:
            audio_dim: MERT embedding dimension
            symbolic_dim: Symbolic feature dimension
            hidden_dim: Cross-attention hidden dimension
            n_heads: Number of attention heads
            num_labels: Number of output dimensions (19 for PercePiano)
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
            audio_pooling: How to pool MERT sequence ('mean', 'max', 'attention')
        """
        super().__init__()
        self.save_hyperparameters()

        self.audio_dim = audio_dim
        self.symbolic_dim = symbolic_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.audio_pooling = audio_pooling
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Audio pooling
        if audio_pooling == "attention":
            self.audio_attn = nn.Sequential(
                nn.Linear(audio_dim, 1),
                nn.Softmax(dim=1),
            )

        # Project audio to hidden dim
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-attention: audio query attends to symbolic key-values
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=hidden_dim,
            kv_dim=symbolic_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Layer norm after attention
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Fusion MLP: concat(audio_pooled, attended_symbolic) -> predictions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        # Training tracking
        self.training_history: List[Dict[str, float]] = []

    def pool_audio(
        self,
        audio_emb: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool variable-length MERT embeddings to fixed-size vector.

        Args:
            audio_emb: (batch, seq_len, audio_dim)
            audio_mask: (batch, seq_len) - True for valid positions

        Returns:
            (batch, audio_dim)
        """
        if audio_mask is not None:
            # Mask out padding
            mask_expanded = audio_mask.unsqueeze(-1).float()
            audio_emb = audio_emb * mask_expanded

        if self.audio_pooling == "mean":
            if audio_mask is not None:
                lengths = audio_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = audio_emb.sum(dim=1) / lengths
            else:
                pooled = audio_emb.mean(dim=1)

        elif self.audio_pooling == "max":
            if audio_mask is not None:
                audio_emb = audio_emb.masked_fill(~audio_mask.unsqueeze(-1), float("-inf"))
            pooled = audio_emb.max(dim=1)[0]

        elif self.audio_pooling == "attention":
            attn_weights = self.audio_attn(audio_emb)  # (batch, seq_len, 1)
            if audio_mask is not None:
                attn_weights = attn_weights.masked_fill(~audio_mask.unsqueeze(-1), float("-inf"))
                attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (audio_emb * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling: {self.audio_pooling}")

        return pooled

    def forward(
        self,
        audio_emb: torch.Tensor,
        symbolic_emb: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        symbolic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_emb: MERT embeddings (batch, audio_len, audio_dim)
            symbolic_emb: Symbolic features (batch, symbolic_len, symbolic_dim)
            audio_mask: (batch, audio_len) - True for valid
            symbolic_mask: (batch, symbolic_len) - True for valid

        Returns:
            predictions: (batch, num_labels)
        """
        # Pool audio sequence
        audio_pooled = self.pool_audio(audio_emb, audio_mask)  # (batch, audio_dim)

        # Project audio to hidden dim
        audio_query = self.audio_proj(audio_pooled)  # (batch, hidden_dim)

        # Cross-attend to symbolic features
        attended_symbolic = self.cross_attention(
            query=audio_query,
            key_value=symbolic_emb,
            kv_mask=symbolic_mask,
        )  # (batch, hidden_dim)

        # Apply layer norm
        attended_symbolic = self.layer_norm(attended_symbolic)

        # Concatenate and predict
        fused = torch.cat([audio_query, attended_symbolic], dim=-1)  # (batch, hidden_dim*2)
        predictions = self.fusion_mlp(fused)  # (batch, num_labels)

        return predictions

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        audio_emb = batch["audio_emb"]
        symbolic_emb = batch["symbolic_emb"]
        labels = batch["labels"]
        audio_mask = batch.get("audio_mask")
        symbolic_mask = batch.get("symbolic_mask")

        preds = self(audio_emb, symbolic_emb, audio_mask, symbolic_mask)
        loss = F.mse_loss(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        audio_emb = batch["audio_emb"]
        symbolic_emb = batch["symbolic_emb"]
        labels = batch["labels"]
        audio_mask = batch.get("audio_mask")
        symbolic_mask = batch.get("symbolic_mask")

        preds = self(audio_emb, symbolic_emb, audio_mask, symbolic_mask)
        loss = F.mse_loss(preds, labels)

        self.log("val_loss", loss, prog_bar=True)

        return {"preds": preds, "labels": labels, "loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class FusionMLPModel(pl.LightningModule):
    """Simple MLP fusion model for prediction-level fusion.

    Takes concatenated audio and symbolic predictions as input.
    """

    def __init__(
        self,
        input_dim: int = 38,  # 19 audio + 19 symbolic
        hidden_dim: int = 64,
        num_labels: int = 19,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        self.training_history: List[Dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"preds": preds, "labels": y, "loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def train_fusion_mlp_cv(
    audio_preds: torch.Tensor,
    symbolic_preds: torch.Tensor,
    labels: torch.Tensor,
    fold_assignments: Dict[str, int],
    sample_keys: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[FusionMLPModel], Dict[str, List[Dict[str, float]]]]:
    """Train FusionMLP with 4-fold cross-validation.

    Args:
        audio_preds: Audio predictions (n_samples, 19)
        symbolic_preds: Symbolic predictions (n_samples, 19)
        labels: Ground truth (n_samples, 19)
        fold_assignments: Dict mapping sample_key -> fold_id
        sample_keys: List of sample keys
        config: Training configuration

    Returns:
        Tuple of:
        - cv_predictions: Predictions from held-out folds
        - models: Trained models per fold
        - training_curves: Dict with per-fold training history
    """
    from torch.utils.data import DataLoader, TensorDataset

    if config is None:
        config = {
            "hidden_dim": 64,
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 100,
            "patience": 10,
            "batch_size": 64,
        }

    # Build input features
    X = torch.cat([audio_preds, symbolic_preds], dim=1)  # (n_samples, 38)
    y = labels

    # Build fold masks
    fold_ids = torch.tensor([fold_assignments[k] for k in sample_keys])
    n_folds = len(set(fold_ids.tolist()))

    cv_preds = torch.zeros_like(labels)
    models: List[FusionMLPModel] = []
    training_curves: Dict[str, List[Dict[str, float]]] = {}

    for fold in range(n_folds):
        train_mask = fold_ids != fold
        val_mask = fold_ids == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
        )

        # Create model
        model = FusionMLPModel(
            input_dim=38,
            hidden_dim=config["hidden_dim"],
            num_labels=19,
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config["patience"],
                    mode="min",
                ),
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, train_loader, val_loader)

        # Get predictions for held-out fold
        model.eval()
        with torch.no_grad():
            cv_preds[val_mask] = model(X_val)

        models.append(model)
        training_curves[f"fold_{fold}"] = model.training_history

    return cv_preds, models, training_curves
