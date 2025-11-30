"""
PyTorch Lightning module for piano performance evaluation.

Supports 3 fusion modes for comparison experiments:
- crossattn: Cross-attention fusion (baseline)
- gated: Gated Multimodal Unit fusion (research recommendation)
- concat: Simple concatenation fusion (simple baseline)

All modes use projection heads to align encoder representations.
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from typing import Dict, List, Optional, Any, Literal

from .audio_encoder import MERTEncoder
from .midi_encoder import MIDIBertEncoder
from .fusion_crossattn import CrossAttentionFusion
from .fusion_concat import ConcatenationFusion
from .fusion_gated import GatedFusion
from .projection import DualProjection
from .aggregation import HierarchicalAggregator
from .mtl_head import MultiTaskHead
from ..losses.uncertainty_loss import UncertaintyWeightedLoss
from ..losses.ranking_loss import RankingLoss
from ..losses.contrastive_loss import InfoNCELoss
from ..losses.lds import LDSWeighting, FDSFeatureSmoothing
from ..losses.bootstrap_loss import BootstrapLoss
from ..losses.coral_loss import CORALHead
from ..data.gpu_augmentation import GPUAudioAugmentation


FusionType = Literal["crossattn", "gated", "concat"]
ModalityType = Literal["audio", "midi", "fusion"]
TrainingMode = Literal["contrastive", "regression", "full"]


class PerformanceEvaluationModel(pl.LightningModule):
    """
    Complete PyTorch Lightning module for piano performance evaluation.

    Architecture:
    1. Audio Encoder: MERT-95M (pre-trained transformer) -> 768-dim
    2. MIDI Encoder: MIDIBert (OctupleMIDI tokenization) -> 256-dim
    3. Projection Heads: Align to shared 512-dim space
    4. Fusion: CrossAttention / Gated / Concatenation (configurable)
    5. Hierarchical Aggregation: BiLSTM + multi-head attention
    6. Multi-Task Head: 8 dimensions with uncertainty weighting

    Loss: Uncertainty-weighted MSE + Ranking + Contrastive
    """

    def __init__(
        self,
        # Model architecture
        audio_dim: int = 768,
        midi_dim: int = 256,
        shared_dim: int = 512,
        aggregator_dim: int = 512,
        num_dimensions: int = 8,
        dimension_names: Optional[List[str]] = None,
        # Modality and training mode options
        modality: ModalityType = "fusion",
        training_mode: TrainingMode = "full",
        # Fusion options
        fusion_type: FusionType = "gated",
        use_projection: bool = True,
        # Encoder options
        mert_model_name: str = "m-a-p/MERT-v1-95M",
        freeze_audio_encoder: bool = False,
        gradient_checkpointing: bool = True,
        midi_pretrained_checkpoint: Optional[str] = None,
        # Loss weights
        mse_weight: float = 1.0,
        ranking_weight: float = 0.2,
        contrastive_weight: float = 0.1,
        ranking_margin: float = 5.0,
        contrastive_temperature: float = 0.07,
        # Base loss options
        base_loss: Literal["mse", "huber", "mae"] = "mse",
        huber_delta: float = 1.0,
        # LDS options
        lds_enabled: bool = False,
        lds_num_bins: int = 100,
        lds_kernel_size: int = 5,
        lds_sigma: float = 2.0,
        lds_reweight_scale: float = 1.0,
        # Bootstrap loss options
        bootstrap_enabled: bool = False,
        bootstrap_beta: float = 0.8,
        bootstrap_warmup_epochs: int = 5,
        # CORAL ordinal regression options
        coral_enabled: bool = False,
        coral_num_classes: int = 20,
        coral_weight: float = 0.3,
        # FDS (Feature Distribution Smoothing) options
        fds_enabled: bool = False,
        fds_num_bins: int = 100,
        fds_momentum: float = 0.9,
        fds_kernel_sigma: float = 2.0,
        fds_start_epoch: int = 0,
        # GPU augmentation options (runs on GPU for speed)
        gpu_augmentation_enabled: bool = False,
        gpu_augmentation_config: Optional[Dict[str, Any]] = None,
        # Training options
        learning_rate: float = 1e-5,
        backbone_lr: float = 5e-6,
        heads_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_epochs: int = 50,
        **kwargs,
    ):
        """
        Initialize Performance Evaluation Model.

        Args:
            audio_dim: Audio encoder output dimension (768 for MERT-95M)
            midi_dim: MIDI encoder output dimension (256 for MIDIBert)
            shared_dim: Shared projection space dimension (512)
            aggregator_dim: Aggregator output dimension
            num_dimensions: Number of evaluation dimensions (8)
            dimension_names: Names of dimensions
            modality: 'audio' (audio-only), 'midi' (midi-only), or 'fusion' (both)
            training_mode: 'contrastive' (alignment only), 'regression' (MTL only), 'full' (both)
            fusion_type: 'crossattn', 'gated', or 'concat'
            use_projection: Whether to use projection heads before fusion
            mert_model_name: HuggingFace MERT model name
            freeze_audio_encoder: Whether to freeze audio encoder weights
            gradient_checkpointing: Enable gradient checkpointing
            midi_pretrained_checkpoint: Path to pretrained MIDI encoder checkpoint
            mse_weight: Weight for uncertainty-weighted MSE loss
            ranking_weight: Weight for ranking loss
            contrastive_weight: Weight for contrastive alignment loss
            ranking_margin: Margin for ranking loss
            contrastive_temperature: Temperature for InfoNCE
            base_loss: Base loss function ('mse', 'huber', 'mae')
            huber_delta: Delta parameter for Huber loss
            lds_enabled: Enable Label Distribution Smoothing
            lds_num_bins: Number of bins for LDS density estimation
            lds_kernel_size: Kernel size for LDS smoothing
            lds_sigma: Sigma for LDS Gaussian kernel
            lds_reweight_scale: Scale factor for LDS reweighting
            bootstrap_enabled: Enable bootstrap loss for noisy labels
            bootstrap_beta: Weight on original label vs prediction (0-1)
            bootstrap_warmup_epochs: Epochs before applying bootstrapping
            coral_enabled: Enable CORAL ordinal regression
            coral_num_classes: Number of ordinal classes (20 = 5-point resolution)
            coral_weight: Weight for CORAL loss component
            fds_enabled: Enable Feature Distribution Smoothing
            fds_num_bins: Number of bins for FDS target discretization
            fds_momentum: Momentum for FDS running statistics
            fds_kernel_sigma: Sigma for FDS Gaussian smoothing kernel
            fds_start_epoch: Epoch to start updating FDS statistics
            learning_rate: Default learning rate
            backbone_lr: Learning rate for backbone encoders
            heads_lr: Learning rate for task heads
            weight_decay: Weight decay for AdamW
            warmup_steps: Linear warmup steps
            max_epochs: Maximum training epochs
        """
        super().__init__()

        # Override dims based on modality
        if modality == "audio":
            midi_dim = 0
        elif modality == "midi":
            audio_dim = 0

        # Save hyperparameters (with potentially modified dims)
        self.save_hyperparameters()

        # Store training mode for loss computation
        self.training_mode = training_mode
        self.modality = modality

        # Default dimension names
        if dimension_names is None:
            dimension_names = [
                "note_accuracy",
                "rhythmic_stability",
                "articulation_clarity",
                "pedal_technique",
                "tone_quality",
                "dynamic_range",
                "musical_expression",
                "overall_interpretation",
            ]

        # Build encoders based on modality
        if audio_dim > 0:
            self.audio_encoder = MERTEncoder(
                model_name=mert_model_name,
                freeze_backbone=freeze_audio_encoder,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            self.audio_encoder = None

        if midi_dim > 0:
            self.midi_encoder = MIDIBertEncoder(
                hidden_size=midi_dim,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
                max_seq_length=512,  # Must match pretrained checkpoint
                pretrained_checkpoint=midi_pretrained_checkpoint,
            )
        else:
            self.midi_encoder = None

        # Projection heads (align to shared space)
        if use_projection:
            self.projection = DualProjection(
                audio_dim=audio_dim,
                midi_dim=midi_dim,
                shared_dim=shared_dim,
                dropout=0.1,
                normalize=False,  # Don't normalize for fusion, only for contrastive
            )
            fusion_input_dim = shared_dim
        else:
            self.projection = None
            fusion_input_dim = audio_dim if midi_dim == 0 else audio_dim + midi_dim

        # Build fusion module based on type
        self.fusion_type = fusion_type
        if fusion_type == "crossattn":
            # CrossAttention needs original dims if no projection
            if use_projection:
                self.fusion = CrossAttentionFusion(
                    audio_dim=shared_dim,
                    midi_dim=shared_dim,
                    num_heads=8,
                    dropout=0.1,
                )
                fusion_output_dim = shared_dim * 2
            else:
                self.fusion = CrossAttentionFusion(
                    audio_dim=audio_dim,
                    midi_dim=midi_dim,
                    num_heads=8,
                    dropout=0.1,
                )
                fusion_output_dim = audio_dim + midi_dim
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                audio_dim=shared_dim if use_projection else audio_dim,
                midi_dim=shared_dim if use_projection else midi_dim,
                output_dim=shared_dim,
                dropout=0.1,
            )
            fusion_output_dim = shared_dim
        elif fusion_type == "concat":
            self.fusion = ConcatenationFusion(
                audio_dim=shared_dim if use_projection else audio_dim,
                midi_dim=shared_dim if use_projection else midi_dim,
                fusion_dim=shared_dim,
                dropout=0.1,
            )
            fusion_output_dim = shared_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Aggregator
        self.aggregator = HierarchicalAggregator(
            input_dim=fusion_output_dim,
            lstm_hidden=256,
            lstm_layers=2,
            attention_heads=4,
            dropout=0.2,
            output_dim=aggregator_dim,
        )

        # Multi-task head
        self.mtl_head = MultiTaskHead(
            input_dim=aggregator_dim,
            shared_hidden=256,
            task_hidden=128,
            dimensions=dimension_names,
            dropout=0.1,
        )

        # Loss functions
        self.uncertainty_loss = UncertaintyWeightedLoss(
            num_tasks=num_dimensions,
            base_loss=base_loss,
            huber_delta=huber_delta,
        )
        self.ranking_loss = RankingLoss(margin=ranking_margin)
        self.contrastive_loss = InfoNCELoss(temperature=contrastive_temperature)

        # Bootstrap loss (for noisy label handling)
        self.bootstrap_loss = None
        if bootstrap_enabled:
            self.bootstrap_loss = BootstrapLoss(
                beta=bootstrap_beta,
                warmup_epochs=bootstrap_warmup_epochs,
                base_loss=base_loss,
                huber_delta=huber_delta,
            )

        # LDS weighting (for imbalanced regression)
        self.lds_weighting = None
        if lds_enabled:
            self.lds_weighting = LDSWeighting(
                num_bins=lds_num_bins,
                kernel_size=lds_kernel_size,
                sigma=lds_sigma,
                reweight_scale=lds_reweight_scale,
            )
            # Note: LDS must be fitted with training labels before use
            # This is done in fit_lds() method or externally

        # CORAL ordinal regression head
        self.coral_head = None
        if coral_enabled:
            self.coral_head = CORALHead(
                input_dim=aggregator_dim,
                num_dimensions=num_dimensions,
                num_classes=coral_num_classes,
                label_range=(0.0, 100.0),
            )

        # FDS feature smoothing (for imbalanced regression)
        self.fds_smoothing = None
        if fds_enabled:
            self.fds_smoothing = FDSFeatureSmoothing(
                feature_dim=aggregator_dim,
                num_bins=fds_num_bins,
                start_update_epoch=fds_start_epoch,
                momentum=fds_momentum,
                kernel_sigma=fds_kernel_sigma,
                label_range=(0.0, 100.0),
            )

        # GPU augmentation (runs on GPU for speed, replaces CPU augmentation)
        self.gpu_augmentation = None
        if gpu_augmentation_enabled:
            aug_config = gpu_augmentation_config or {}
            self.gpu_augmentation = GPUAudioAugmentation(
                sample_rate=24000,  # MERT sample rate
                gain_prob=aug_config.get('gain_prob', 0.4),
                noise_prob=aug_config.get('noise_prob', 0.3),
                time_mask_prob=aug_config.get('time_mask_prob', 0.3),
                pitch_shift_prob=aug_config.get('pitch_shift_prob', 0.3),
                time_stretch_prob=aug_config.get('time_stretch_prob', 0.3),
                max_augmentations=aug_config.get('max_augmentations', 3),
            )

        # Metrics
        self.dimension_names = self.mtl_head.get_dimension_names()
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics for each dimension and split."""
        for split in ["train", "val", "test"]:
            for dim_name in self.dimension_names:
                setattr(self, f"{split}_mae_{dim_name}", MeanAbsoluteError())
                setattr(self, f"{split}_pearson_{dim_name}", PearsonCorrCoef())
                setattr(self, f"{split}_spearman_{dim_name}", SpearmanCorrCoef())

    def fit_lds(self, labels: torch.Tensor) -> None:
        """
        Fit LDS weighting from training labels.

        Call this before training to enable LDS weighting.

        Args:
            labels: Training labels [num_samples, num_dims]
        """
        if self.lds_weighting is not None:
            self.lds_weighting.fit(labels)
            print(f"LDS fitted: {self.lds_weighting.get_density_stats()}")

    def forward(
        self,
        audio_waveform: torch.Tensor,
        midi_tokens: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            audio_waveform: Raw audio waveform [batch, num_samples] at 24kHz
            midi_tokens: OctupleMIDI tokens [batch, events, 8] (optional)
            audio_mask: Audio attention mask
            midi_mask: MIDI attention mask
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with 'scores', 'uncertainties', and optional 'attention'/'embeddings'
        """
        # Encode audio (skip if audio-only mode disabled)
        audio_features = None
        if self.hparams.audio_dim > 0 and self.audio_encoder is not None:
            audio_features, _ = self.audio_encoder(
                audio_waveform=audio_waveform,
                attention_mask=audio_mask,
            )

        # Encode MIDI (skip if midi-only mode disabled)
        midi_features = None
        if midi_tokens is not None and self.midi_encoder is not None:
            midi_features = self.midi_encoder(
                midi_tokens=midi_tokens,
                attention_mask=midi_mask,
            )

        # Validate we have at least one modality
        if audio_features is None and midi_features is None:
            import warnings
            warnings.warn("Skipping batch: No audio or MIDI features available")
            return None

        # Project to shared space
        audio_projected = audio_features
        midi_projected = midi_features
        if self.projection is not None:
            audio_projected, midi_projected = self.projection(
                audio_features=audio_features,
                midi_features=midi_features,
            )

        # Fuse modalities
        fused_features, fusion_info = self.fusion(
            audio_features=audio_projected,
            midi_features=midi_projected,
            audio_mask=audio_mask,
            midi_mask=midi_mask,
        )

        # Aggregate temporal information
        aggregated, aggregation_attention = self.aggregator(
            fused_features=fused_features,
            return_attention=return_attention,
        )

        # Multi-task prediction
        scores, uncertainties = self.mtl_head(
            features=aggregated,
            return_uncertainties=True,
        )

        result = {
            "scores": scores,
            "uncertainties": uncertainties,
        }

        if return_attention:
            result["attention"] = {
                "fusion": fusion_info,
                "aggregation": aggregation_attention,
            }

        if return_embeddings:
            result["embeddings"] = {
                "audio_raw": audio_features,
                "midi_raw": midi_features,
                "audio_proj": audio_projected,
                "midi_proj": midi_projected,
                "fused": fused_features,
                "aggregated": aggregated,
            }

        return result

    def _compute_combined_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        audio_embed: Optional[torch.Tensor] = None,
        midi_embed: Optional[torch.Tensor] = None,
        aggregated_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with all components.

        Args:
            predictions: Model predictions [batch, num_dims]
            targets: Target labels [batch, num_dims]
            audio_embed: Audio embeddings for contrastive [batch, dim]
            midi_embed: MIDI embeddings for contrastive [batch, dim]
            aggregated_features: Aggregated features for CORAL [batch, agg_dim]

        Returns:
            Dictionary with loss components
        """
        # Get effective targets (possibly bootstrapped)
        effective_targets = targets
        bootstrap_loss_val = torch.tensor(0.0, device=predictions.device)

        if self.bootstrap_loss is not None:
            # Update bootstrap epoch from trainer
            if self.trainer is not None:
                self.bootstrap_loss.set_epoch(self.current_epoch)
            # Compute bootstrap loss (also gives us softened targets implicitly)
            bootstrap_loss_val = self.bootstrap_loss(predictions, targets)

        # 1. Uncertainty-weighted loss (MSE/Huber/MAE based on config)
        mse_output = self.uncertainty_loss(
            predictions=predictions,
            targets=effective_targets,
            log_vars=self.mtl_head.log_vars,
        )
        mse_loss = mse_output["loss"]

        # Apply LDS reweighting if enabled and fitted
        lds_weight = torch.tensor(1.0, device=predictions.device)
        if self.lds_weighting is not None and self.lds_weighting._fitted:
            lds_weights = self.lds_weighting.get_weights(targets)
            lds_weight = lds_weights.mean()
            mse_loss = mse_loss * lds_weight

        # 2. Ranking loss
        rank_loss = self.ranking_loss(predictions, targets)

        # 3. Contrastive loss (if both embeddings available)
        if audio_embed is not None and midi_embed is not None:
            contrast_loss = self.contrastive_loss(audio_embed, midi_embed)
        else:
            contrast_loss = torch.tensor(0.0, device=predictions.device)

        # 4. CORAL ordinal regression loss (if enabled)
        coral_loss_val = torch.tensor(0.0, device=predictions.device)
        if self.coral_head is not None and aggregated_features is not None:
            coral_loss_val = self.coral_head(aggregated_features, targets)

        # Combined loss
        # If bootstrap enabled, use bootstrap loss as primary regression loss
        if self.bootstrap_loss is not None:
            regression_loss = bootstrap_loss_val
        else:
            regression_loss = mse_loss

        total_loss = (
            self.hparams.mse_weight * regression_loss
            + self.hparams.ranking_weight * rank_loss
            + self.hparams.contrastive_weight * contrast_loss
        )

        # Add CORAL loss if enabled
        if self.coral_head is not None:
            total_loss = total_loss + self.hparams.coral_weight * coral_loss_val

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "rank_loss": rank_loss,
            "contrast_loss": contrast_loss,
            "coral_loss": coral_loss_val,
            "bootstrap_loss": bootstrap_loss_val,
            "lds_weight": lds_weight,
            "task_losses": mse_output["task_losses"],
            "uncertainties": mse_output["uncertainties"],
        }

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        split: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Shared step for train/val/test."""
        audio_waveform = batch["audio_waveform"]
        midi_tokens = batch.get("midi_tokens", None)
        targets = batch.get("labels", None)

        # Apply GPU augmentation during training (faster than CPU augmentation)
        if self.training and self.gpu_augmentation is not None:
            audio_waveform = self.gpu_augmentation(audio_waveform)

        # Only request embeddings when needed for contrastive loss (training_mode="full")
        # or for FDS feature smoothing. Avoids keeping 6 extra tensors in memory.
        need_embeddings = (
            self.training_mode in ("full", "contrastive") or
            (self.fds_smoothing is not None and self.training)
        )

        # Forward pass
        output = self.forward(
            audio_waveform=audio_waveform,
            midi_tokens=midi_tokens,
            return_embeddings=need_embeddings,
        )

        if output is None:
            return None

        # Extract embeddings only if they were requested
        embeddings = output.get("embeddings", None)

        # Get pooled embeddings for contrastive loss (only if embeddings available)
        audio_embed = None
        midi_embed = None
        aggregated_features = None
        if embeddings is not None:
            if embeddings.get("audio_proj") is not None:
                audio_embed = embeddings["audio_proj"].mean(dim=1)
                audio_embed = F.normalize(audio_embed, p=2, dim=-1)
            if embeddings.get("midi_proj") is not None:
                midi_embed = embeddings["midi_proj"].mean(dim=1)
                midi_embed = F.normalize(midi_embed, p=2, dim=-1)
            aggregated_features = embeddings.get("aggregated", None)

        # Handle contrastive-only training mode
        if self.training_mode == "contrastive":
            return self._contrastive_step(audio_embed, midi_embed, split)

        # For regression and full modes, we need predictions and targets
        predictions = output["scores"]

        # FDS: Update statistics during training
        if self.fds_smoothing is not None and self.training and targets is not None:
            if aggregated_features is not None:
                current_epoch = self.current_epoch if self.trainer else 0
                self.fds_smoothing.update_statistics(
                    aggregated_features.detach(),
                    targets,
                    current_epoch,
                )

        # Compute combined loss
        loss_output = self._compute_combined_loss(
            predictions=predictions,
            targets=targets,
            audio_embed=audio_embed if self.training_mode == "full" else None,
            midi_embed=midi_embed if self.training_mode == "full" else None,
            aggregated_features=aggregated_features,
        )

        total_loss = loss_output["loss"]
        task_losses = loss_output["task_losses"]

        # Log losses
        self.log(f"{split}_loss", total_loss, prog_bar=True)
        self.log(f"{split}_mse_loss", loss_output["mse_loss"])
        self.log(f"{split}_rank_loss", loss_output["rank_loss"])
        self.log(f"{split}_contrast_loss", loss_output["contrast_loss"])

        # Log CORAL loss if enabled
        if self.coral_head is not None:
            self.log(f"{split}_coral_loss", loss_output["coral_loss"])

        # Log bootstrap and LDS metrics if enabled
        if self.bootstrap_loss is not None:
            self.log(f"{split}_bootstrap_loss", loss_output["bootstrap_loss"])
        if self.lds_weighting is not None:
            self.log(f"{split}_lds_weight", loss_output["lds_weight"])

        # Log FDS statistics periodically
        if self.fds_smoothing is not None and split == "val":
            fds_stats = self.fds_smoothing.get_statistics_summary()
            self.log("fds_bin_coverage", fds_stats["coverage"])

        # Log per-dimension metrics
        # NOTE: Pearson/Spearman metrics accumulate ALL predictions/targets until epoch end,
        # causing OOM on large training sets (7000+ batches). Only compute during val/test.
        for i, dim_name in enumerate(self.dimension_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]

            # MAE uses running mean - safe for training
            mae_metric = getattr(self, f"{split}_mae_{dim_name}")
            mae_metric(pred_i, target_i)
            self.log(f"{split}_mae_{dim_name}", mae_metric, on_step=False, on_epoch=True)
            self.log(f"{split}_task_loss_{dim_name}", task_losses[i])

            # Correlation metrics only for val/test (they accumulate all data)
            if split != "train":
                pearson_metric = getattr(self, f"{split}_pearson_{dim_name}")
                spearman_metric = getattr(self, f"{split}_spearman_{dim_name}")
                pearson_metric(pred_i, target_i)
                spearman_metric(pred_i, target_i)
                self.log(f"{split}_pearson_{dim_name}", pearson_metric, on_step=False, on_epoch=True)
                self.log(f"{split}_spearman_{dim_name}", spearman_metric, on_step=False, on_epoch=True)

        # Log uncertainties on validation
        if split == "val":
            for i, dim_name in enumerate(self.dimension_names):
                self.log(f"uncertainty_{dim_name}", loss_output["uncertainties"][i])

        return {
            "loss": total_loss,
            "predictions": predictions,
            "targets": targets,
        }

    def _contrastive_step(
        self,
        audio_embed: Optional[torch.Tensor],
        midi_embed: Optional[torch.Tensor],
        split: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Contrastive-only training step for MERT-MIDIBert alignment."""
        if audio_embed is None or midi_embed is None:
            return None

        # Compute contrastive loss
        contrast_loss = self.contrastive_loss(audio_embed, midi_embed)

        # Compute alignment score (cosine similarity of matched pairs)
        alignment_score = (audio_embed * midi_embed).sum(dim=-1).mean()

        # Compute contrastive accuracy (% of correct positives ranked above negatives)
        with torch.no_grad():
            similarity = torch.mm(audio_embed, midi_embed.t())  # [batch, batch]
            batch_size = similarity.shape[0]
            labels = torch.arange(batch_size, device=similarity.device)

            # Audio-to-MIDI accuracy
            a2m_correct = (similarity.argmax(dim=1) == labels).float().mean()
            # MIDI-to-Audio accuracy
            m2a_correct = (similarity.argmax(dim=0) == labels).float().mean()
            contrastive_acc = (a2m_correct + m2a_correct) / 2

        # Log metrics
        self.log(f"{split}_loss", contrast_loss, prog_bar=True)
        self.log(f"{split}_contrast_loss", contrast_loss)
        self.log(f"{split}_alignment_score", alignment_score, prog_bar=True)
        self.log(f"{split}_contrastive_acc", contrastive_acc)

        return {
            "loss": contrast_loss,
            "alignment_score": alignment_score,
            "contrastive_acc": contrastive_acc,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        result = self._shared_step(batch, "train")
        if result is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        # Extract loss and explicitly free predictions/targets to help GC
        loss = result["loss"]
        del result

        # Periodic garbage collection to prevent memory accumulation
        # Every 500 steps, force GC to reclaim any leaked objects
        if batch_idx > 0 and batch_idx % 500 == 0:
            gc.collect()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step with diagnostics."""
        result = self._shared_step(batch, "val")
        if result is None:
            return

        # Compute diagnostics on first batch
        if batch_idx == 0:
            diagnostics = self._compute_diagnostics(batch)
            for name, value in diagnostics.items():
                self.log(f"val_{name}", value, prog_bar=False)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        self._shared_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Clean up memory at end of each epoch."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        """Clean up memory at end of validation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_diagnostics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute fusion diagnostics for analysis."""
        diagnostics = {"fusion_type": hash(self.fusion_type) % 100}  # Encode fusion type

        with torch.no_grad():
            output = self.forward(
                audio_waveform=batch["audio_waveform"],
                midi_tokens=batch.get("midi_tokens", None),
                return_attention=True,
                return_embeddings=True,
            )

            if output is None:
                return diagnostics

            embeddings = output["embeddings"]

            # Cross-modal alignment
            if embeddings["audio_proj"] is not None and embeddings["midi_proj"] is not None:
                audio_pooled = F.normalize(embeddings["audio_proj"].mean(dim=1), p=2, dim=-1)
                midi_pooled = F.normalize(embeddings["midi_proj"].mean(dim=1), p=2, dim=-1)
                alignment = (audio_pooled * midi_pooled).sum(dim=-1).mean().item()
                diagnostics["cross_modal_alignment"] = alignment

            # Feature diversity
            if embeddings["audio_proj"] is not None and embeddings["audio_proj"].shape[0] > 1:
                audio_norm = F.normalize(embeddings["audio_proj"].mean(dim=1), p=2, dim=-1)
                sim_matrix = torch.mm(audio_norm, audio_norm.t())
                mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
                diagnostics["audio_diversity"] = 1.0 - sim_matrix[mask].mean().item()

            # Gated fusion: log gate values
            if self.fusion_type == "gated" and output["attention"]["fusion"] is not None:
                fusion_info = output["attention"]["fusion"]
                if "gate_mean" in fusion_info:
                    diagnostics["gate_mean"] = fusion_info["gate_mean"]

        return diagnostics

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers with differential learning rates."""
        backbone_params = []
        head_params = []

        # Backbone (encoders)
        if self.audio_encoder is not None and not self.hparams.freeze_audio_encoder:
            backbone_params.extend(list(self.audio_encoder.parameters()))
        if self.midi_encoder is not None:
            backbone_params.extend(list(self.midi_encoder.parameters()))

        # Heads (projection, fusion, aggregator, mtl)
        if self.projection is not None:
            head_params.extend(list(self.projection.parameters()))
        head_params.extend(list(self.fusion.parameters()))
        head_params.extend(list(self.aggregator.parameters()))
        head_params.extend(list(self.mtl_head.parameters()))

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": float(self.hparams.backbone_lr)})
        if head_params:
            param_groups.append({"params": head_params, "lr": float(self.hparams.heads_lr)})

        optimizer = AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_steps),
            eta_min=1e-6,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


if __name__ == "__main__":
    print("PyTorch Lightning module loaded successfully")
    print("Supports 3 fusion modes: crossattn, gated, concat")
    print("Combined loss: Uncertainty MSE + Ranking + Contrastive")
