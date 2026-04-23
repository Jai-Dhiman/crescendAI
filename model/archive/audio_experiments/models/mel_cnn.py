"""4-layer CNN on mel spectrograms."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


class MelCNNModel(pl.LightningModule):
    """4-layer CNN on mel spectrograms.

    Non-foundation model baseline to demonstrate the value of MERT.
    Trained from scratch on 128-bin mel spectrograms.
    """

    def __init__(
        self,
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

        # Conv stack: [B, 1, 128, T] -> [B, 256, 1, 1]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # MLP head (same structure as MERT baseline for fair comparison)
        self.head = nn.Sequential(
            nn.Linear(256, hidden_dim),
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

    def forward(self, mel):
        # mel: [B, 128, T]
        x = mel.unsqueeze(1)  # [B, 1, 128, T]
        x = self.conv(x)  # [B, 256, 1, 1]
        x = x.flatten(1)  # [B, 256]
        return self.head(x)

    def training_step(self, batch, idx):
        pred = self(batch["mel"])
        loss = self.loss_fn(pred, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        pred = self(batch["mel"])
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
