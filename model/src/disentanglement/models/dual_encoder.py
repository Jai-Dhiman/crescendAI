"""Approach C: Disentangled Dual-Encoder Model.

Uses separate encoders for piece-level and style-level features,
with adversarial training to ensure style encoder doesn't capture piece info.

Architecture:
- Piece encoder: Captures piece-level information
- Style encoder: Captures performer expression (with GRL for adversarial training)
- Adversarial piece classifier: Tries to predict piece from style features
- Dimension prediction heads: Predict performance scores from style features

Loss:
    L_total = L_regression + lambda_adv * L_adversarial

References:
- DANN: https://jmlr.org/papers/volume17/15-239/15-239.pdf
- DADA: http://proceedings.mlr.press/v97/peng19b/peng19b.pdf
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from .grl import GradientReversalLayer, get_grl_lambda


class DisentangledDualEncoderModel(pl.LightningModule):
    """Dual encoder with adversarial disentanglement.

    Separates piece-level features from performer style using:
    - A piece encoder that captures compositional characteristics
    - A style encoder that captures performer expression
    - Adversarial training to prevent style encoder from encoding piece info
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        num_pieces: int = 206,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_adversarial: float = 0.5,
        grl_schedule: str = "linear",  # "constant", "linear", "cosine"
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.lambda_adversarial = lambda_adversarial
        self.grl_schedule = grl_schedule
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels
        self.num_pieces = num_pieces

        # Attention pooling (shared)
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Piece encoder - captures compositional characteristics
        self.piece_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Style encoder - captures performer expression
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gradient reversal layer for adversarial training
        self.grl = GradientReversalLayer(lambda_=0.0)

        # Adversarial piece classifier (from style features)
        # Should fail to classify piece if style encoder is truly disentangled
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pieces),
        )

        # Dimension prediction heads (from style features)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
        """Pool frame embeddings."""
        if self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: Input embeddings [B, T, D].
            mask: Attention mask [B, T].

        Returns:
            Dict with predictions, piece_logits, z_piece, z_style.
        """
        pooled = self.pool(x, mask)

        # Encode with both encoders
        z_piece = self.piece_encoder(pooled)
        z_style = self.style_encoder(pooled)

        # Adversarial piece classification from style features
        # GRL reverses gradients so minimizing CE actually maximizes it for style encoder
        z_style_rev = self.grl(z_style)
        piece_logits = self.adversarial_classifier(z_style_rev)

        # Dimension predictions from style features only
        predictions = self.prediction_head(z_style)

        return {
            "predictions": predictions,
            "piece_logits": piece_logits,
            "z_piece": z_piece,
            "z_style": z_style,
        }

    def on_train_epoch_start(self):
        """Update GRL lambda based on training progress."""
        lambda_ = get_grl_lambda(
            self.current_epoch,
            self.max_epochs,
            schedule=self.grl_schedule,
        )
        self.grl.set_lambda(lambda_)
        self.log("grl_lambda", lambda_)

    def training_step(self, batch, idx):
        outputs = self(batch["embeddings"], batch.get("attention_mask"))

        # Regression loss
        l_reg = self.mse(outputs["predictions"], batch["labels"])

        # Adversarial loss (piece classification from style)
        l_adv = self.ce(outputs["piece_logits"], batch["piece_ids"])

        # Total loss
        # Note: GRL makes gradients for l_adv negative for style_encoder,
        # so the style encoder learns to make piece classification harder
        loss = l_reg + self.lambda_adversarial * l_adv

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_reg_loss", l_reg)
        self.log("train_adv_loss", l_adv)

        # Log adversarial accuracy
        with torch.no_grad():
            pred_piece = outputs["piece_logits"].argmax(dim=-1)
            adv_acc = (pred_piece == batch["piece_ids"]).float().mean()
            self.log("train_piece_acc", adv_acc)

        return loss

    def validation_step(self, batch, idx):
        outputs = self(batch["embeddings"], batch.get("attention_mask"))

        l_reg = self.mse(outputs["predictions"], batch["labels"])
        l_adv = self.ce(outputs["piece_logits"], batch["piece_ids"])

        self.log("val_loss", l_reg, prog_bar=True)
        self.log("val_adv_loss", l_adv)

        # Piece classification accuracy from style encoder (should be low if disentangled)
        pred_piece = outputs["piece_logits"].argmax(dim=-1)
        adv_acc = (pred_piece == batch["piece_ids"]).float().mean()
        self.log("val_style_piece_acc", adv_acc)

        self.val_outputs.append({
            "p": outputs["predictions"].cpu(),
            "l": batch["labels"].cpu(),
            "z_style": outputs["z_style"].cpu(),
            "piece_ids": batch["piece_ids"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        p = torch.cat([x["p"] for x in self.val_outputs]).numpy()
        l = torch.cat([x["l"] for x in self.val_outputs]).numpy()

        # R2 score
        self.log("val_r2", r2_score(l, p), prog_bar=True)

        # Compute intra-piece variance (how different are predictions for same piece)
        z_style = torch.cat([x["z_style"] for x in self.val_outputs])
        piece_ids = torch.cat([x["piece_ids"] for x in self.val_outputs])

        # Within-piece variance of style embeddings
        unique_pieces = piece_ids.unique()
        within_vars = []
        for pid in unique_pieces:
            mask = piece_ids == pid
            if mask.sum() > 1:
                z_piece = z_style[mask]
                var = z_piece.var(dim=0).mean().item()
                within_vars.append(var)

        if within_vars:
            self.log("val_within_piece_var", sum(within_vars) / len(within_vars))

        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def get_disentanglement_metrics(self, dataloader):
        """Evaluate disentanglement quality.

        Returns metrics indicating how well piece and style are separated.
        """
        self.eval()
        device = next(self.parameters()).device

        all_z_piece = []
        all_z_style = []
        all_piece_ids = []
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                emb = batch["embeddings"].to(device)
                mask = batch.get("attention_mask")
                if mask is not None:
                    mask = mask.to(device)

                outputs = self(emb, mask)
                all_z_piece.append(outputs["z_piece"].cpu())
                all_z_style.append(outputs["z_style"].cpu())
                all_piece_ids.append(batch["piece_ids"])
                all_predictions.append(outputs["predictions"].cpu())

        z_piece = torch.cat(all_z_piece)
        z_style = torch.cat(all_z_style)
        piece_ids = torch.cat(all_piece_ids)
        predictions = torch.cat(all_predictions)

        # 1. Piece classification from z_style (should be near chance)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        style_piece_acc = cross_val_score(
            clf, z_style.numpy(), piece_ids.numpy(), cv=3, scoring="accuracy"
        ).mean()

        # 2. Within-piece variance (should be high for z_style)
        unique_pieces = piece_ids.unique()
        within_vars = []
        for pid in unique_pieces:
            mask = piece_ids == pid
            if mask.sum() > 1:
                var = z_style[mask].var(dim=0).mean().item()
                within_vars.append(var)
        within_piece_var = sum(within_vars) / len(within_vars) if within_vars else 0

        # 3. Silhouette score for piece clustering
        from sklearn.metrics import silhouette_score

        if len(unique_pieces) > 1:
            piece_silhouette = silhouette_score(
                z_piece.numpy(), piece_ids.numpy(), sample_size=min(1000, len(piece_ids))
            )
            style_silhouette = silhouette_score(
                z_style.numpy(), piece_ids.numpy(), sample_size=min(1000, len(piece_ids))
            )
        else:
            piece_silhouette = 0.0
            style_silhouette = 0.0

        # 4. Intra-piece prediction std
        pred_stds = []
        for pid in unique_pieces:
            mask = piece_ids == pid
            if mask.sum() > 1:
                std = predictions[mask].std(dim=0).mean().item()
                pred_stds.append(std)
        intra_piece_std = sum(pred_stds) / len(pred_stds) if pred_stds else 0

        return {
            "style_piece_acc": style_piece_acc,
            "within_piece_var": within_piece_var,
            "piece_silhouette": piece_silhouette,
            "style_silhouette": style_silhouette,
            "intra_piece_std": intra_piece_std,
        }
