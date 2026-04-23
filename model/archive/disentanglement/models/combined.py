"""Combined models that integrate multiple disentanglement approaches.

Combinations:
- A+B: Contrastive pretrain, siamese fine-tune
- B+C: Siamese with disentangled encoders
- A+C: Contrastive + adversarial
- A+B+C: Full combination
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from .grl import GradientReversalLayer, get_grl_lambda
from ..losses import piece_based_infonce_loss


class ContrastiveDisentangledModel(pl.LightningModule):
    """A+C: Combines contrastive learning with adversarial disentanglement.

    Uses:
    - InfoNCE with piece-based positives for piece-invariant representations
    - Adversarial piece classification to disentangle style from piece
    - Pairwise ranking heads for dimension predictions
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = 19,
        num_pieces: int = 206,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_adversarial: float = 0.5,
        grl_schedule: str = "linear",
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_adversarial = lambda_adversarial
        self.grl_schedule = grl_schedule
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Style encoder (main encoder for predictions)
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

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # GRL and adversarial classifier
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pieces),
        )

        # Comparison module for pairwise ranking
        comparison_dim = hidden_dim * 4
        self.comparator = nn.Sequential(
            nn.Linear(comparison_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Ranking heads
        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        self.ce = nn.CrossEntropyLoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
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

    def encode(self, x, mask=None):
        pooled = self.pool(x, mask)
        return self.style_encoder(pooled)

    def compare(self, z_a, z_b):
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)

        proj_a = self.projection(z_a)
        proj_b = self.projection(z_b)

        ranking_logits = self.compare(z_a, z_b)

        # Adversarial piece logits (for both samples)
        z_a_rev = self.grl(z_a)
        z_b_rev = self.grl(z_b)
        piece_logits_a = self.adversarial_classifier(z_a_rev)
        piece_logits_b = self.adversarial_classifier(z_b_rev)

        return {
            "ranking_logits": ranking_logits,
            "z_a": z_a,
            "z_b": z_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
            "piece_logits_a": piece_logits_a,
            "piece_logits_b": piece_logits_b,
        }

    def on_train_epoch_start(self):
        lambda_ = get_grl_lambda(self.current_epoch, self.max_epochs, self.grl_schedule)
        self.grl.set_lambda(lambda_)
        self.log("grl_lambda", lambda_)

    def compute_ranking_loss(self, logits, labels_a, labels_b):
        label_diff = labels_a - labels_b
        targets = (label_diff > 0).float()
        non_ambiguous = label_diff.abs() >= self.ambiguous_threshold

        if not non_ambiguous.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            logits[non_ambiguous], targets[non_ambiguous], reduction="mean"
        )

    def training_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        # Ranking loss
        l_rank = self.compute_ranking_loss(
            outputs["ranking_logits"], batch["labels_a"], batch["labels_b"]
        )

        # Contrastive loss
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
        l_contrast = piece_based_infonce_loss(all_proj, all_pieces, self.temperature)

        # Adversarial loss
        l_adv = 0.5 * (
            self.ce(outputs["piece_logits_a"], batch["piece_ids"]) +
            self.ce(outputs["piece_logits_b"], batch["piece_ids"])
        )

        loss = l_rank + self.lambda_contrastive * l_contrast + self.lambda_adversarial * l_adv

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank", l_rank)
        self.log("train_contrast", l_contrast)
        self.log("train_adv", l_adv)

        return loss

    def validation_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        l_rank = self.compute_ranking_loss(
            outputs["ranking_logits"], batch["labels_a"], batch["labels_b"]
        )
        self.log("val_loss", l_rank, prog_bar=True)

        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= self.ambiguous_threshold

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


class SiameseDisentangledModel(pl.LightningModule):
    """B+C: Siamese ranking with disentangled encoders.

    Uses:
    - Disentangled piece/style encoders from Approach C
    - Siamese comparison from Approach B
    - Adversarial training to ensure style encoder is piece-invariant
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
        grl_schedule: str = "linear",
        comparison_type: str = "concat_diff",
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.05,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.lambda_adversarial = lambda_adversarial
        self.grl_schedule = grl_schedule
        self.comparison_type = comparison_type
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Style encoder (for ranking predictions)
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

        # GRL and adversarial classifier
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pieces),
        )

        # Comparison module (siamese)
        comparison_dim = hidden_dim * 4
        self.comparator = nn.Sequential(
            nn.Linear(comparison_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Ranking heads
        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        self.ce = nn.CrossEntropyLoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
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

    def encode(self, x, mask=None):
        pooled = self.pool(x, mask)
        return self.style_encoder(pooled)

    def compare(self, z_a, z_b):
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)

        ranking_logits = self.compare(z_a, z_b)

        z_a_rev = self.grl(z_a)
        z_b_rev = self.grl(z_b)
        piece_logits_a = self.adversarial_classifier(z_a_rev)
        piece_logits_b = self.adversarial_classifier(z_b_rev)

        return {
            "ranking_logits": ranking_logits,
            "z_a": z_a,
            "z_b": z_b,
            "piece_logits_a": piece_logits_a,
            "piece_logits_b": piece_logits_b,
        }

    def on_train_epoch_start(self):
        lambda_ = get_grl_lambda(self.current_epoch, self.max_epochs, self.grl_schedule)
        self.grl.set_lambda(lambda_)

    def training_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        # Ranking loss
        label_diff = batch["labels_a"] - batch["labels_b"]
        targets = (label_diff > 0).float()
        non_ambiguous = label_diff.abs() >= self.ambiguous_threshold

        if non_ambiguous.any():
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            l_rank = F.binary_cross_entropy_with_logits(
                outputs["ranking_logits"][non_ambiguous],
                targets[non_ambiguous],
                reduction="mean",
            )
        else:
            l_rank = torch.tensor(0.0, device=outputs["ranking_logits"].device)

        # Adversarial loss
        l_adv = 0.5 * (
            self.ce(outputs["piece_logits_a"], batch["piece_ids"]) +
            self.ce(outputs["piece_logits_b"], batch["piece_ids"])
        )

        loss = l_rank + self.lambda_adversarial * l_adv

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank", l_rank)
        self.log("train_adv", l_adv)

        return loss

    def validation_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= self.ambiguous_threshold

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


class FullCombinedModel(pl.LightningModule):
    """A+B+C: Full combination of all three approaches.

    Combines:
    - InfoNCE contrastive loss (piece-based positives)
    - Siamese dimension-specific ranking
    - Adversarial disentanglement
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = 19,
        num_pieces: int = 206,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_adversarial: float = 0.5,
        grl_schedule: str = "linear",
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.05,
        pooling: str = "attention",
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_adversarial = lambda_adversarial
        self.grl_schedule = grl_schedule
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing
        self.pooling = pooling
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Attention pooling
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection for contrastive
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # GRL + adversarial
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pieces),
        )

        # Siamese comparator
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        self.ce = nn.CrossEntropyLoss()
        self.val_outputs = []

    def pool(self, x, mask=None):
        if self.pooling == "attention":
            scores = self.attn(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (x * m).sum(1) / m.sum(1).clamp(min=1)
        return x.mean(1)

    def encode(self, x, mask=None):
        return self.encoder(self.pool(x, mask))

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)

        proj_a = self.projection(z_a)
        proj_b = self.projection(z_b)

        diff = z_a - z_b
        prod = z_a * z_b
        comp = self.comparator(torch.cat([z_a, z_b, diff, prod], dim=-1))
        ranking_logits = torch.cat([h(comp) for h in self.ranking_heads], dim=-1)

        z_a_rev = self.grl(z_a)
        z_b_rev = self.grl(z_b)

        return {
            "ranking_logits": ranking_logits,
            "proj_a": proj_a,
            "proj_b": proj_b,
            "piece_logits_a": self.adversarial_classifier(z_a_rev),
            "piece_logits_b": self.adversarial_classifier(z_b_rev),
        }

    def on_train_epoch_start(self):
        lambda_ = get_grl_lambda(self.current_epoch, self.max_epochs, self.grl_schedule)
        self.grl.set_lambda(lambda_)

    def training_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"], batch["embeddings_b"],
            batch.get("mask_a"), batch.get("mask_b"),
        )

        # Ranking
        diff = batch["labels_a"] - batch["labels_b"]
        targets = (diff > 0).float()
        non_amb = diff.abs() >= self.ambiguous_threshold
        if non_amb.any():
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            l_rank = F.binary_cross_entropy_with_logits(
                outputs["ranking_logits"][non_amb], targets[non_amb], reduction="mean"
            )
        else:
            l_rank = torch.tensor(0.0, device=outputs["ranking_logits"].device)

        # Contrastive
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
        l_contrast = piece_based_infonce_loss(all_proj, all_pieces, self.temperature)

        # Adversarial
        l_adv = 0.5 * (
            self.ce(outputs["piece_logits_a"], batch["piece_ids"]) +
            self.ce(outputs["piece_logits_b"], batch["piece_ids"])
        )

        loss = l_rank + self.lambda_contrastive * l_contrast + self.lambda_adversarial * l_adv
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        outputs = self(
            batch["embeddings_a"], batch["embeddings_b"],
            batch.get("mask_a"), batch.get("mask_b"),
        )
        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        true_rank = (all_a > all_b).float()
        pred_rank = (all_logits > 0).float()
        non_amb = (all_a - all_b).abs() >= self.ambiguous_threshold

        if non_amb.any():
            acc = (pred_rank[non_amb] == true_rank[non_amb]).float().mean().item()
        else:
            acc = 0.5
        self.log("val_pairwise_acc", acc, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
