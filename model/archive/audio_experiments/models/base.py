"""Base MERT model with configurable pooling and loss."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Concordance Correlation Coefficient loss.

    CCC measures agreement between predicted and target values,
    accounting for both correlation and bias.

    Loss = 1 - CCC (averaged across dimensions)
    """
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    covar = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    ccc = (2 * covar) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)
    return (1 - ccc).mean()


class BaseMERTModel(pl.LightningModule):
    """Base model for MERT embeddings with configurable pooling and loss.

    Pooling options:
    - mean: Simple mean pooling over frames
    - max: Max pooling over frames
    - attention: Learned attention weights
    - lstm: Bi-LSTM with attention

    Loss options:
    - mse: Mean squared error
    - ccc: Concordance correlation coefficient
    - hybrid: MSE + 0.5 * CCC
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
        loss_type: str = "mse",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.pooling = pooling
        self.loss_type = loss_type
        self.max_epochs = max_epochs

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # LSTM pooling
        if pooling == "lstm":
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
            self.lstm_attn = nn.Sequential(
                nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1)
            )
            input_dim = hidden_dim  # LSTM output dim

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

    def pool(self, x, mask=None, lengths=None):
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

        elif self.pooling == "lstm":
            if lengths is not None:
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.lstm(packed)
                x, _ = pad_packed_sequence(lstm_out, batch_first=True)
            else:
                x, _ = self.lstm(x)
            # Attention over LSTM outputs
            scores = self.lstm_attn(x).squeeze(-1)
            if mask is not None:
                # Adjust mask size if needed
                if mask.shape[1] > x.shape[1]:
                    mask = mask[:, : x.shape[1]]
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)

        return x.mean(1)

    def forward(self, x, mask=None, lengths=None):
        pooled = self.pool(x, mask, lengths)
        return self.clf(pooled)

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return self.mse_loss(pred, target)
        elif self.loss_type == "ccc":
            return ccc_loss(pred, target)
        elif self.loss_type == "hybrid":
            return self.mse_loss(pred, target) + 0.5 * ccc_loss(pred, target)
        return self.mse_loss(pred, target)

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"), batch.get("lengths"))
        loss = self.compute_loss(pred, batch["labels"])
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
