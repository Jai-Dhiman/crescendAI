"""Audio encoder experiments: MuQ domain adaptation via LoRA, staged, and full unfreeze."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_improvement.heads import HeteroscedasticHead
from model_improvement.losses import DimensionWiseRankingLoss, gaussian_nll_loss, piece_based_infonce_loss
from model_improvement.taxonomy import NUM_DIMS


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
        num_labels: int = NUM_DIMS,
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
        warmup_epochs: int = 5,
        use_pretrained_muq: bool = False,
        lora_rank: int = 16,
        lora_target_layers: tuple[int, ...] = (9, 10, 11, 12),
        use_gaussian_head: bool = False,
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
        self.warmup_epochs = warmup_epochs
        self.num_labels = num_labels
        self.use_gaussian_head = use_gaussian_head

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

        # Regression head — scalar sigmoid or Gaussian (mu, sigma) depending on flag
        if use_gaussian_head:
            self.head = HeteroscedasticHead(hidden_dim, num_labels)
        else:
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict quality scores via the regression head.

        Args:
            x: Frame embeddings [B, T, D].
            mask: Attention mask [B, T].

        Returns:
            Scalar head: scores [B, num_labels] in [0, 1].
            Gaussian head: (mu [B, num_labels], sigma [B, num_labels]).
        """
        z = self.encode(x, mask)
        if self.use_gaussian_head:
            return self.head(z)
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

        # Regression loss
        if self.use_gaussian_head:
            mu_a, sigma_a = self.head(outputs["z_a"])
            mu_b, sigma_b = self.head(outputs["z_b"])
            l_reg = (
                gaussian_nll_loss(mu_a, sigma_a, batch["labels_a"])
                + gaussian_nll_loss(mu_b, sigma_b, batch["labels_b"])
            ) / 2.0
        else:
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
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class MuQLoRAMaxModel(MuQLoRAModel):
    """A1-Max: MuQ + LoRA with all Tier 1 improvements.

    Extends MuQLoRAModel with:
    - ListMLE ranking loss (per-dimension, grouped by piece within batch)
    - CCC regression loss (replaces MSE)
    - Embedding mixup (Beta distribution)
    - Configurable loss weights (ranking-dominant by default)

    Architecture is identical to MuQLoRAModel. Only training_step differs.
    """

    def __init__(
        self,
        *,
        lambda_listmle: float = 1.5,
        use_ccc: bool = True,
        mixup_alpha: float = 0.2,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.3,
        lambda_invariance: float = 0.1,
        use_gaussian_head: bool = False,
        **kwargs,
    ):
        # NOTE: Do NOT call self.save_hyperparameters() here -- parent
        # already calls it, and a second call overwrites parent hparams.
        super().__init__(
            lambda_contrastive=lambda_contrastive,
            lambda_regression=lambda_regression,
            lambda_invariance=lambda_invariance,
            use_gaussian_head=use_gaussian_head,
            **kwargs,
        )
        # Manually store extra hparams that parent doesn't know about
        self.hparams.update({
            "lambda_listmle": lambda_listmle,
            "use_ccc": use_ccc,
            "mixup_alpha": mixup_alpha,
            "use_gaussian_head": use_gaussian_head,
        })
        self.lambda_listmle = lambda_listmle
        self.use_ccc = use_ccc
        self.mixup_alpha = mixup_alpha

        from model_improvement.losses import ListMLELoss, ccc_loss
        self._listmle = ListMLELoss()
        self._ccc_loss_fn = ccc_loss

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        emb_a = batch["embeddings_a"]
        emb_b = batch["embeddings_b"]
        labels_a = batch["labels_a"]
        labels_b = batch["labels_b"]

        # Apply mixup on embeddings + labels (before encoding)
        if self.mixup_alpha > 0 and self.training:
            from model_improvement.data import apply_mixup
            emb_a, labels_a = apply_mixup(emb_a, labels_a, self.mixup_alpha)
            emb_b, labels_b = apply_mixup(emb_b, labels_b, self.mixup_alpha)

        outputs = self(
            emb_a, emb_b,
            batch.get("mask_a"), batch.get("mask_b"),
        )

        # 1. Pairwise ranking loss (BCE, from parent)
        l_rank = self.ranking_loss(
            outputs["ranking_logits"], labels_a, labels_b,
        )

        # 2. Contrastive loss
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids_a"], batch["piece_ids_b"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # 3. Regression loss (Gaussian NLL, CCC, or MSE)
        if self.use_gaussian_head:
            mu_a, sigma_a = self.head(outputs["z_a"])
            mu_b, sigma_b = self.head(outputs["z_b"])
            l_reg = (
                gaussian_nll_loss(mu_a, sigma_a, labels_a)
                + gaussian_nll_loss(mu_b, sigma_b, labels_b)
            ) / 2.0
            scores_a, scores_b = mu_a.detach(), mu_b.detach()
        else:
            scores_a = self.regression_head(outputs["z_a"])
            scores_b = self.regression_head(outputs["z_b"])
            if self.use_ccc:
                l_reg = (
                    self._ccc_loss_fn(scores_a, labels_a)
                    + self._ccc_loss_fn(scores_b, labels_b)
                ) / 2.0
            else:
                l_reg = (
                    F.mse_loss(scores_a, labels_a)
                    + F.mse_loss(scores_b, labels_b)
                ) / 2.0

        # 4. ListMLE on regression scores grouped by piece
        l_listmle = torch.tensor(0.0, device=self.device)
        if self.lambda_listmle > 0:
            all_scores = torch.cat([scores_a, scores_b], dim=0)
            all_labels = torch.cat([labels_a, labels_b], dim=0)
            piece_ids = torch.cat(
                [batch["piece_ids_a"], batch["piece_ids_b"]], dim=0
            )

            unique_pieces = piece_ids.unique()
            listmle_count = 0
            for pid in unique_pieces:
                pmask = piece_ids == pid
                if pmask.sum() < 2:
                    continue
                l_listmle = l_listmle + self._listmle(
                    all_scores[pmask], all_labels[pmask]
                )
                listmle_count += 1
            if listmle_count > 0:
                l_listmle = l_listmle / listmle_count

        # 5. Augmentation invariance loss
        l_inv = torch.tensor(0.0, device=self.device)
        if "embeddings_aug_a" in batch:
            z_aug_a = self.encode(
                batch["embeddings_aug_a"], batch.get("mask_a")
            )
            l_inv = F.mse_loss(outputs["z_a"], z_aug_a)

        # Total loss (ranking-dominant)
        loss = (
            l_rank
            + self.lambda_listmle * l_listmle
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
            + self.lambda_invariance * l_inv
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_listmle_loss", l_listmle)
        self.log("train_contrast_loss", l_contrast)
        reg_name = "train_ccc_loss" if self.use_ccc else "train_reg_loss"
        self.log(reg_name, l_reg)
        self.log("train_inv_loss", l_inv)

        return loss


class MuQFrozenProbeModel(MuQLoRAMaxModel):
    """Frozen MuQ backbone + MLP probe for ablation baseline.

    Same head architecture as A1-Max (attention pooling, 2-layer MLP,
    regression head) but with all backbone parameters frozen and MSE-only
    loss. Provides a fair baseline for measuring the contribution of
    LoRA adaptation and ranking losses.
    """

    def __init__(self, **kwargs):
        # Force MSE-only loss config
        kwargs.setdefault("lora_rank", 1)  # Minimal, will be frozen
        kwargs["lambda_listmle"] = 0.0
        kwargs["lambda_contrastive"] = 0.0
        kwargs["lambda_regression"] = 1.0
        kwargs["lambda_invariance"] = 0.0
        kwargs["use_ccc"] = False
        kwargs["mixup_alpha"] = 0.0
        super().__init__(**kwargs)

        # Freeze everything except the MLP heads
        # (attn, encoder, regression_head remain trainable)
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = False
        # Freeze comparator and ranking heads (not used in MSE-only)
        for param in self.comparator.parameters():
            param.requires_grad = False
        for param in self.ranking_heads.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        """MSE regression only -- no ranking or contrastive losses."""
        emb_a = batch["embeddings_a"]
        emb_b = batch["embeddings_b"]
        labels_a = batch["labels_a"]
        labels_b = batch["labels_b"]

        z_a = self.encode(emb_a, batch.get("mask_a"))
        z_b = self.encode(emb_b, batch.get("mask_b"))

        scores_a = self.regression_head(z_a)
        scores_b = self.regression_head(z_b)

        loss = (F.mse_loss(scores_a, labels_a) + F.mse_loss(scores_b, labels_b)) / 2.0

        self.log("train_loss", loss, prog_bar=True)
        return loss


class MuQFullUnfreezeModel(pl.LightningModule):
    """A3: Full unfreeze with gradual layer unfreezing and discriminative LR.

    Starts with all backbone layers frozen, then progressively unfreezes
    layers from top to bottom according to an epoch-based schedule. Each
    layer group gets a decayed learning rate (deeper = lower LR).

    Uses the same multi-task loss as A1 (ranking + contrastive + regression).

    Args:
        unfreeze_schedule: Dict mapping epoch -> list of layer indices to unfreeze
            at that epoch. E.g., {0: [12], 10: [11], 20: [10], 30: [9]}
        lr_decay_factor: Multiplier applied per layer depth. Layer k gets
            lr * (factor ^ (max_layer - k)).
        mock_num_layers: Number of mock transformer layers for testing
            (only used when use_pretrained_muq=False).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        num_labels: int = NUM_DIMS,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.5,
        margin: float = 0.2,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        max_epochs: int = 200,
        warmup_epochs: int = 5,
        use_pretrained_muq: bool = False,
        unfreeze_schedule: dict[int, list[int]] | None = None,
        lr_decay_factor: float = 0.8,
        mock_num_layers: int = 12,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.wd = weight_decay
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.lambda_regression = lambda_regression
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.num_labels = num_labels
        self.lr_decay_factor = lr_decay_factor
        self.unfreeze_schedule = unfreeze_schedule or {}
        self._unfrozen_layers: set[int] = set()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Backbone (MuQ or mock encoder for testing)
        if use_pretrained_muq:
            raise NotImplementedError(
                "Pretrained MuQ loading not yet implemented. "
                "Pass use_pretrained_muq=False for testing."
            )
        else:
            from .lora import create_mock_encoder
            self.backbone = create_mock_encoder(
                hidden_size=input_dim, num_layers=mock_num_layers
            )

        # Freeze all backbone layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Shared encoder on top of backbone output
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

        # Regression head
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

    def get_unfrozen_layers(self) -> set[int]:
        """Return the set of currently unfrozen backbone layer indices."""
        return set(self._unfrozen_layers)

    def on_train_epoch_start(self) -> None:
        """Check unfreeze schedule and unfreeze layers as needed."""
        self.unfreeze_for_epoch(self.current_epoch)

    def unfreeze_for_epoch(self, epoch: int) -> None:
        """Unfreeze backbone layers scheduled for the given epoch.

        Called automatically by on_train_epoch_start during training.
        Can also be called directly in tests.
        """
        if epoch in self.unfreeze_schedule:
            for layer_idx in self.unfreeze_schedule[epoch]:
                if layer_idx < len(self.backbone.layers):
                    for param in self.backbone.layers[layer_idx].parameters():
                        param.requires_grad = True
                    self._unfrozen_layers.add(layer_idx)

    def pool(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * w).sum(1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            hidden = x
            for layer in self.backbone.layers:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, use_reentrant=False
                )
            backbone_out = hidden
        else:
            backbone_out = self.backbone(x)
        pooled = self.pool(backbone_out, mask)
        return self.encoder(pooled)

    def compare(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
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

        # Regression loss
        scores_a = self.regression_head(outputs["z_a"])
        scores_b = self.regression_head(outputs["z_b"])
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
        """Build optimizer with discriminative learning rates per layer group.

        Unfrozen backbone layers get decayed LR (deeper = lower).
        Head layers (encoder, comparator, ranking/regression heads) get base LR.
        """
        param_groups = []

        # Backbone layer groups with discriminative LR
        if self._unfrozen_layers:
            max_layer = max(self._unfrozen_layers)
            for layer_idx in sorted(self._unfrozen_layers):
                layer_params = list(self.backbone.layers[layer_idx].parameters())
                trainable = [p for p in layer_params if p.requires_grad]
                if trainable:
                    depth = max_layer - layer_idx
                    layer_lr = self.lr * (self.lr_decay_factor ** depth)
                    param_groups.append({
                        "params": trainable,
                        "lr": layer_lr,
                        "name": f"backbone_layer_{layer_idx}",
                    })

        # Head parameters at base LR
        head_modules = [
            self.attn, self.encoder, self.projection,
            self.comparator, self.ranking_heads, self.regression_head,
        ]
        head_params = []
        for module in head_modules:
            head_params.extend(p for p in module.parameters() if p.requires_grad)

        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": self.lr,
                "name": "heads",
            })

        if not param_groups:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": self.lr}]

        opt = torch.optim.AdamW(param_groups, weight_decay=self.wd)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


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
        num_labels: int = NUM_DIMS,
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
        warmup_epochs: int = 5,
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
        self.warmup_epochs = warmup_epochs
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

            # Contrastive loss (same as training)
            proj_clean = self.projection(z_clean)
            proj_aug = self.projection(z_aug)
            all_proj = torch.cat([proj_clean, proj_aug], dim=0)
            all_pieces = torch.cat([batch["piece_ids"], batch["piece_ids"]], dim=0)
            l_contrast = piece_based_infonce_loss(
                all_proj, all_pieces, temperature=self.temperature
            )

            # Invariance loss
            l_inv = F.mse_loss(z_clean, z_aug)

            val_loss = l_contrast + self.lambda_invariance * l_inv
            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_contrast_loss", l_contrast)
            self.log("val_inv_loss", l_inv)
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
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
