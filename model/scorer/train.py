import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scorer.dataset import SegmentDataset, collate_fn, ALL_DIMS
from scorer.model import Evaluator, masked_mae, masked_mae_weighted, cosine_distillation_loss


class LitEvaluator(pl.LightningModule):
    def __init__(self, lr: float = 3e-4, num_dims: int = len(ALL_DIMS), num_datasets: int = 8,
                 ds_embed_dim: int = 16, alpha_pseudo: float = 0.3, beta_distill: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = Evaluator(num_dims=num_dims, num_datasets=num_datasets, ds_embed_dim=ds_embed_dim)

    def training_step(self, batch, batch_idx):
        mel, y, m, ds, py, pm, pw, distill = batch
        pred, emb = self.model(mel, ds)

        loss_h = masked_mae(pred, y, m)

        loss_p = torch.tensor(0.0, device=self.device)
        if pm.sum() > 0:
            loss_p = masked_mae_weighted(pred, py, pm, pw)

        loss_d = torch.tensor(0.0, device=self.device)
        if distill.abs().sum() > 0:
            loss_d = cosine_distillation_loss(emb, distill, self.model.distill_proj)

        loss = loss_h + self.hparams.alpha_pseudo * loss_p + self.hparams.beta_distill * loss_d

        self.log("loss/train_total", loss, prog_bar=True)
        self.log("loss/train_human", loss_h)
        if pm.sum() > 0:
            self.log("loss/train_pseudo", loss_p)
        if distill.abs().sum() > 0:
            self.log("loss/train_distill", loss_d)
        return loss

    def validation_step(self, batch, batch_idx):
        mel, y, m, ds, py, pm, pw, distill = batch
        pred, emb = self.model(mel, ds)
        loss_h = masked_mae(pred, y, m)
        self.log("loss/val_human", loss_h, prog_bar=True)
        return loss_h

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def main():
    train_manifest = os.environ.get("TRAIN_MANIFEST", "data/splits/train.jsonl")
    valid_manifest = os.environ.get("VALID_MANIFEST", "data/splits/valid.jsonl")

    train_ds = SegmentDataset(train_manifest)
    val_ds = SegmentDataset(valid_manifest)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)

    lit = LitEvaluator(lr=3e-4, alpha_pseudo=0.3, beta_distill=0.1)
    trainer = pl.Trainer(max_epochs=30, accelerator="auto", devices="auto", log_every_n_steps=20)
    trainer.fit(lit, train_loader, val_loader)


if __name__ == "__main__":
    main()