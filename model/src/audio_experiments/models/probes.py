"""Simple probe models: Linear probe and Statistics MLP."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class LinearProbeModel(pl.LightningModule):
    """Simple linear probe on MERT embeddings.

    Just a single linear layer with mean pooling - tests whether
    the MLP head adds value over linear readout.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_labels: int = 19,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

        self.linear = nn.Linear(input_dim, num_labels)
        self.loss_fn = nn.MSELoss()
        self.val_outputs = []

    def forward(self, x, mask=None, lengths=None):
        # Mean pooling
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)
        return torch.sigmoid(self.linear(pooled))

    def training_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        loss = self.loss_fn(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["embeddings"], batch.get("attention_mask"))
        self.log("val_loss", self.loss_fn(pred, batch["labels"]), prog_bar=True)
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


class StatsMLPModel(pl.LightningModule):
    """MLP on hand-crafted audio statistics.

    Trivial baseline to ensure the task isn't solvable with simple features.
    """

    def __init__(
        self,
        input_dim: int = 49,
        hidden_dim: int = 256,
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

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.MSELoss()
        self.val_outputs = []

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, idx):
        pred = self(batch["features"])
        loss = self.loss_fn(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["features"])
        self.log("val_loss", self.loss_fn(pred, batch["labels"]), prog_bar=True)
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
