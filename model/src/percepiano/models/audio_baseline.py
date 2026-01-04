"""
Audio Baseline Model for PercePiano Evaluation.

Uses MERT-330M embeddings with mean pooling and MLP head for
predicting 19 perceptual dimensions.
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Define locally to avoid circular import
PERCEPIANO_DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamic_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "sophistication",
    "interpretation",
]


class AudioPercePianoModel(pl.LightningModule):
    """
    MERT-based baseline model for PercePiano evaluation.

    Architecture:
        MERT embeddings [B, T, 1024]
            -> Pooling (mean/max/attention) [B, 1024]
            -> Linear(1024, 512) + GELU + Dropout
            -> Linear(512, 512) + GELU + Dropout
            -> Linear(512, 19) + Sigmoid

    This is intentionally simple to establish a floor. Future versions
    can add Bi-LSTM on MERT frames or attention pooling.

    Attributes:
        input_dim: MERT embedding dimension (1024)
        hidden_dim: MLP hidden dimension
        num_labels: Number of output dimensions (19)
        pooling: Pooling strategy ("mean", "max", "attention")
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
    ):
        """
        Initialize model.

        Args:
            input_dim: MERT embedding dimension (1024 for MERT-330M)
            hidden_dim: MLP hidden dimension
            num_labels: Number of PercePiano dimensions (19)
            dropout: Dropout probability
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            pooling: Pooling strategy ("mean", "max", "attention")
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pooling = pooling

        # Optional attention pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )

        # Classifier head (matches symbolic baseline architecture)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Metrics storage
        self.validation_outputs: list = []
        self.test_outputs: list = []

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [B, T, input_dim] MERT embeddings
            attention_mask: [B, T] mask (1 for valid, 0 for padding)

        Returns:
            predictions: [B, num_labels] in range [0, 1]
        """
        # Pool across time dimension
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = embeddings.mean(dim=1)

        elif self.pooling == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), float("-inf")
                )
            pooled = embeddings.max(dim=1).values

        elif self.pooling == "attention":
            # Attention weights
            scores = self.attention(embeddings).squeeze(-1)  # [B, T]
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
            pooled = (embeddings * weights).sum(dim=1)  # [B, H]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        predictions = self.classifier(pooled)
        return predictions

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch["embeddings"], batch["attention_mask"])
        loss = self.loss_fn(predictions, batch["labels"])

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch["embeddings"], batch["attention_mask"])
        loss = self.loss_fn(predictions, batch["labels"])

        self.log("val_loss", loss, prog_bar=True)

        # Store for epoch-end metrics
        self.validation_outputs.append(
            {
                "predictions": predictions.detach().cpu(),
                "labels": batch["labels"].detach().cpu(),
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return

        # Aggregate predictions and labels
        all_preds = torch.cat([x["predictions"] for x in self.validation_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_outputs])

        # Compute R2
        r2 = r2_score(all_labels.numpy(), all_preds.numpy())
        self.log("val_r2", r2, prog_bar=True)

        # Per-dimension R2
        for i, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
            dim_r2 = r2_score(all_labels[:, i].numpy(), all_preds[:, i].numpy())
            self.log(f"val_r2_{dim_name}", dim_r2)

        self.validation_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch["embeddings"], batch["attention_mask"])
        loss = self.loss_fn(predictions, batch["labels"])

        self.log("test_loss", loss)

        # Store for epoch-end metrics
        self.test_outputs.append(
            {
                "predictions": predictions.detach().cpu(),
                "labels": batch["labels"].detach().cpu(),
                "keys": batch["keys"],
            }
        )

    def on_test_epoch_end(self) -> None:
        if not self.test_outputs:
            return

        # Aggregate predictions and labels
        all_preds = torch.cat([x["predictions"] for x in self.test_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_outputs])

        # Compute R2
        r2 = r2_score(all_labels.numpy(), all_preds.numpy())
        self.log("test_r2", r2)

        # Per-dimension R2
        for i, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
            dim_r2 = r2_score(all_labels[:, i].numpy(), all_preds[:, i].numpy())
            self.log(f"test_r2_{dim_name}", dim_r2)

        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class AudioPercePianoWithLSTM(pl.LightningModule):
    """
    MERT + Bi-LSTM model for PercePiano evaluation.

    Architecture:
        MERT embeddings [B, T, 1024]
            -> Bi-LSTM [B, T, 512]
            -> Mean pooling [B, 512]
            -> Linear(512, 512) + GELU + Dropout
            -> Linear(512, 19) + Sigmoid

    This is a more sophisticated variant that processes temporal
    structure in the MERT embeddings.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        lstm_hidden: int = 256,
        mlp_hidden: int = 512,
        num_labels: int = 19,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize model.

        Args:
            input_dim: MERT embedding dimension
            lstm_hidden: LSTM hidden dimension (bidirectional -> 2x)
            mlp_hidden: MLP hidden dimension
            num_labels: Number of PercePiano dimensions
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classifier head
        lstm_out_dim = lstm_hidden * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
            nn.Sigmoid(),
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Metrics storage
        self.validation_outputs: list = []

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Pack sequences for efficient LSTM processing
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeddings)

        # Masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = lstm_out.mean(dim=1)

        # Classify
        predictions = self.classifier(pooled)
        return predictions

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch["embeddings"], batch["attention_mask"])
        loss = self.loss_fn(predictions, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch["embeddings"], batch["attention_mask"])
        loss = self.loss_fn(predictions, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)

        self.validation_outputs.append(
            {
                "predictions": predictions.detach().cpu(),
                "labels": batch["labels"].detach().cpu(),
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_outputs])

        r2 = r2_score(all_labels.numpy(), all_preds.numpy())
        self.log("val_r2", r2, prog_bar=True)

        for i, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
            dim_r2 = r2_score(all_labels[:, i].numpy(), all_preds[:, i].numpy())
            self.log(f"val_r2_{dim_name}", dim_r2)

        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
