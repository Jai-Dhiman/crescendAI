"""Audio encoder experiments: MuQ domain adaptation via LoRA, staged, and full unfreeze."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from disentanglement.losses import DimensionWiseRankingLoss, piece_based_infonce_loss


class MuQLoRAModel(pl.LightningModule):
    """A1: MuQ + LoRA multi-task model.

    Combines pairwise ranking, contrastive learning, regression, and
    augmentation invariance losses. Optionally loads a pretrained MuQ
    backbone with LoRA adapters applied to the last few layers.

    Architecture:
    - Optional MuQ backbone with LoRA (when use_pretrained_muq=True)
    - Attention pooling over temporal frames
    - Shared encoder (2-layer MLP)
    - Projection head for contrastive learning
    - Comparator for pairwise ranking with per-dimension heads
    - Regression head with sigmoid for absolute quality prediction
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.5,
        lambda_invariance: float = 0.1,
        margin: float = 0.2,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        max_epochs: int = 200,
        use_pretrained_muq: bool = False,
        lora_rank: int = 16,
        lora_target_layers: tuple[int, ...] = (9, 10, 11, 12),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_regression = lambda_regression
        self.lambda_invariance = lambda_invariance
        self.max_epochs = max_epochs
        self.num_labels = num_labels

        # Optional MuQ backbone with LoRA
        self.backbone = None
        if use_pretrained_muq:
            from .lora import apply_lora_to_muq
            # Load MuQ and apply LoRA -- caller responsible for providing model
            raise NotImplementedError(
                "Pretrained MuQ loading not yet implemented. "
                "Pass use_pretrained_muq=False and provide pre-extracted embeddings."
            )

        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Shared encoder
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

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Comparison module for pairwise ranking
        # Takes [z_a; z_b; z_a - z_b; z_a * z_b]
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

        # Per-dimension ranking heads
        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        # Regression head for absolute quality prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        # Losses
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )

        self.val_outputs: list[dict] = []

    def pool(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attention-pool frame embeddings to sequence representation."""
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * w).sum(1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode and pool embeddings to [B, hidden_dim]."""
        pooled = self.pool(x, mask)
        return self.encoder(pooled)

    def compare(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compare two embeddings and produce per-dimension ranking logits."""
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for pairwise ranking.

        Returns dict with ranking_logits, z_a, z_b, proj_a, proj_b.
        """
        z_a = self.encode(emb_a, mask_a)
        z_b = self.encode(emb_b, mask_b)

        proj_a = self.projection(z_a)
        proj_b = self.projection(z_b)

        ranking_logits = self.compare(z_a, z_b)

        return {
            "ranking_logits": ranking_logits,
            "z_a": z_a,
            "z_b": z_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
        }

    def predict_scores(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict absolute quality scores in [0, 1] via the regression head.

        Args:
            x: Frame embeddings [B, T, D].
            mask: Attention mask [B, T].

        Returns:
            Predicted scores [B, num_labels] in [0, 1].
        """
        z = self.encode(x, mask)
        return self.regression_head(z)

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        # Ranking loss
        l_rank = self.ranking_loss(
            outputs["ranking_logits"],
            batch["labels_a"],
            batch["labels_b"],
        )

        # Contrastive loss
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids_a"], batch["piece_ids_b"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # Regression loss (predict absolute scores for both A and B)
        scores_a = self.regression_head(outputs["z_a"])
        scores_b = self.regression_head(outputs["z_b"])
        l_reg = (
            F.mse_loss(scores_a, batch["labels_a"])
            + F.mse_loss(scores_b, batch["labels_b"])
        ) / 2.0

        # Augmentation invariance loss (if augmented embeddings present)
        l_inv = torch.tensor(0.0, device=self.device)
        if "embeddings_aug_a" in batch:
            z_aug_a = self.encode(batch["embeddings_aug_a"], batch.get("mask_a"))
            l_inv = F.mse_loss(outputs["z_a"], z_aug_a)

        # Total loss
        loss = (
            l_rank
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
            + self.lambda_invariance * l_inv
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_contrast_loss", l_contrast)
        self.log("train_reg_loss", l_reg)
        self.log("train_inv_loss", l_inv)

        return loss

    def validation_step(self, batch: dict, idx: int) -> None:
        outputs = self(
            batch["embeddings_a"],
            batch["embeddings_b"],
            batch.get("mask_a"),
            batch.get("mask_b"),
        )

        l_rank = self.ranking_loss(
            outputs["ranking_logits"],
            batch["labels_a"],
            batch["labels_b"],
        )
        self.log("val_loss", l_rank, prog_bar=True)

        self.val_outputs.append({
            "logits": outputs["ranking_logits"].cpu(),
            "labels_a": batch["labels_a"].cpu(),
            "labels_b": batch["labels_b"].cpu(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self.val_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= 0.05

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


class MuQStagedModel(pl.LightningModule):
    """A2: Two-stage domain adaptation model.

    Stage 1 (self_supervised): Contrastive cross-performer learning +
    augmentation invariance using T3+T4 data. No labels required.

    Stage 2 (supervised): Same multi-task loss as A1 (ranking + contrastive +
    regression) using T1+T2+T3 data.

    Call switch_to_supervised() to transition between stages.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = 19,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.5,
        lambda_invariance: float = 0.5,
        margin: float = 0.2,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        max_epochs: int = 200,
        use_pretrained_muq: bool = False,
        stage: str = "self_supervised",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_regression = lambda_regression
        self.lambda_invariance = lambda_invariance
        self.max_epochs = max_epochs
        self.num_labels = num_labels
        self.stage = stage

        if stage not in ("self_supervised", "supervised"):
            raise ValueError(f"stage must be 'self_supervised' or 'supervised', got '{stage}'")

        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Shared encoder
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

        # Projection head for contrastive learning (used in both stages)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Comparison module (used in supervised stage)
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

        # Per-dimension ranking heads (supervised stage)
        self.ranking_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_labels)
        ])

        # Regression head (supervised stage)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

        # Losses
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )

        self.val_outputs: list[dict] = []

    def pool(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * w).sum(1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool(x, mask)
        return self.encoder(pooled)

    def compare(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        diff = z_a - z_b
        prod = z_a * z_b
        concat = torch.cat([z_a, z_b, diff, prod], dim=-1)
        comp = self.comparator(concat)
        logits = [head(comp) for head in self.ranking_heads]
        return torch.cat(logits, dim=-1)

    def switch_to_supervised(self) -> None:
        """Transition from self-supervised to supervised stage."""
        self.stage = "supervised"

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        if self.stage == "self_supervised":
            return self._self_supervised_step(batch)
        else:
            return self._supervised_step(batch)

    def _self_supervised_step(self, batch: dict) -> torch.Tensor:
        """Stage 1: Contrastive + invariance on unlabeled paired data."""
        z_clean = self.encode(batch["embeddings_clean"], batch.get("mask"))
        z_aug = self.encode(batch["embeddings_augmented"], batch.get("mask"))

        # Contrastive loss using piece membership
        proj_clean = self.projection(z_clean)
        proj_aug = self.projection(z_aug)
        all_proj = torch.cat([proj_clean, proj_aug], dim=0)
        all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # Augmentation invariance: clean and augmented should produce same embedding
        l_inv = F.mse_loss(z_clean, z_aug)

        loss = l_contrast + self.lambda_invariance * l_inv

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_contrast_loss", l_contrast)
        self.log("train_inv_loss", l_inv)

        return loss

    def _supervised_step(self, batch: dict) -> torch.Tensor:
        """Stage 2: Full multi-task ranking + regression."""
        z_a = self.encode(batch["embeddings_a"], batch.get("mask_a"))
        z_b = self.encode(batch["embeddings_b"], batch.get("mask_b"))

        ranking_logits = self.compare(z_a, z_b)

        # Ranking loss
        l_rank = self.ranking_loss(
            ranking_logits, batch["labels_a"], batch["labels_b"],
        )

        # Contrastive loss
        proj_a = self.projection(z_a)
        proj_b = self.projection(z_b)
        all_proj = torch.cat([proj_a, proj_b], dim=0)
        all_pieces = torch.cat([batch["piece_ids_a"], batch["piece_ids_b"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # Regression loss
        scores_a = self.regression_head(z_a)
        scores_b = self.regression_head(z_b)
        l_reg = (
            F.mse_loss(scores_a, batch["labels_a"])
            + F.mse_loss(scores_b, batch["labels_b"])
        ) / 2.0

        loss = (
            l_rank
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_contrast_loss", l_contrast)
        self.log("train_reg_loss", l_reg)

        return loss

    def validation_step(self, batch: dict, idx: int) -> None:
        if self.stage == "self_supervised":
            z_clean = self.encode(batch["embeddings_clean"], batch.get("mask"))
            z_aug = self.encode(batch["embeddings_augmented"], batch.get("mask"))
            l_inv = F.mse_loss(z_clean, z_aug)
            self.log("val_loss", l_inv, prog_bar=True)
        else:
            z_a = self.encode(batch["embeddings_a"], batch.get("mask_a"))
            z_b = self.encode(batch["embeddings_b"], batch.get("mask_b"))
            ranking_logits = self.compare(z_a, z_b)

            l_rank = self.ranking_loss(
                ranking_logits, batch["labels_a"], batch["labels_b"],
            )
            self.log("val_loss", l_rank, prog_bar=True)

            self.val_outputs.append({
                "logits": ranking_logits.cpu(),
                "labels_a": batch["labels_a"].cpu(),
                "labels_b": batch["labels_b"].cpu(),
            })

    def on_validation_epoch_end(self) -> None:
        if not self.val_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels_a = torch.cat([x["labels_a"] for x in self.val_outputs])
        all_labels_b = torch.cat([x["labels_b"] for x in self.val_outputs])

        true_ranking = (all_labels_a > all_labels_b).float()
        pred_ranking = (all_logits > 0).float()

        diff = (all_labels_a - all_labels_b).abs()
        non_ambiguous = diff >= 0.05

        if non_ambiguous.any():
            correct = (pred_ranking[non_ambiguous] == true_ranking[non_ambiguous]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.5

        self.log("val_pairwise_acc", accuracy, prog_bar=True)
        self.val_outputs.clear()

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
