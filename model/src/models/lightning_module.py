import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from typing import Dict, List, Optional, Tuple, Any

from .audio_encoder import MERTEncoder
from .midi_encoder import MIDIBertEncoder
from .fusion import CrossAttentionFusion
from .aggregation import HierarchicalAggregator
from .mtl_head import MultiTaskHead
from ..losses.uncertainty_loss import UncertaintyWeightedLoss


class PerformanceEvaluationModel(pl.LightningModule):
    """
    Complete PyTorch Lightning module for piano performance evaluation.

    Architecture:
    1. Audio Encoder: MERT-95M (pre-trained transformer)
    2. MIDI Encoder: MIDIBert (OctupleMIDI tokenization)
    3. Cross-Attention Fusion: Bidirectional attention (multi-modal required)
    4. Hierarchical Aggregation: BiLSTM + multi-head attention
    5. Multi-Task Head: 10 dimensions with uncertainty weighting

    Training:
    - Stage 2: Pseudo-label pre-training (MAESTRO)
    - Stage 3: Expert label fine-tuning (200-300 segments)
    """

    def __init__(
        self,
        # Model architecture
        audio_dim: int = 768,
        midi_dim: int = 256,
        fusion_dim: int = 1024,
        aggregator_dim: int = 512,
        num_dimensions: int = 10,
        dimension_names: Optional[List[str]] = None,
        # Encoder options
        mert_model_name: str = "m-a-p/MERT-v1-95M",
        freeze_audio_encoder: bool = False,
        gradient_checkpointing: bool = True,
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
            audio_dim: Audio encoder output dimension
            midi_dim: MIDI encoder output dimension
            fusion_dim: Fusion module output dimension
            aggregator_dim: Aggregator output dimension
            num_dimensions: Number of evaluation dimensions
            dimension_names: Names of dimensions
            mert_model_name: HuggingFace MERT model name
            freeze_audio_encoder: Whether to freeze audio encoder weights
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            learning_rate: Default learning rate
            backbone_lr: Learning rate for backbone encoders (MERT, MIDIBert)
            heads_lr: Learning rate for task heads
            weight_decay: Weight decay for AdamW optimizer
            warmup_steps: Linear warmup steps for learning rate scheduler
            max_epochs: Maximum training epochs
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Build model components
        self.audio_encoder = MERTEncoder(
            model_name=mert_model_name,
            freeze_backbone=freeze_audio_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Only create MIDI encoder if midi_dim > 0 (audio-only mode when midi_dim=0)
        if midi_dim > 0:
            self.midi_encoder = MIDIBertEncoder(
                hidden_size=midi_dim,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
            )
        else:
            self.midi_encoder = None

        self.fusion = CrossAttentionFusion(
            audio_dim=audio_dim,
            midi_dim=midi_dim,
            num_heads=8,
            dropout=0.1,
        )

        self.aggregator = HierarchicalAggregator(
            input_dim=fusion_dim,
            lstm_hidden=256,
            lstm_layers=2,
            attention_heads=4,
            dropout=0.2,
            output_dim=aggregator_dim,
        )

        self.mtl_head = MultiTaskHead(
            input_dim=aggregator_dim,
            shared_hidden=256,
            task_hidden=128,
            dimensions=dimension_names,
            dropout=0.1,
        )

        # Loss function (uncertainty-weighted only)
        self.loss_fn = UncertaintyWeightedLoss(
            num_tasks=num_dimensions,
        )

        # Metrics (per dimension)
        self.dimension_names = self.mtl_head.get_dimension_names()
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics for each dimension and split."""
        for split in ["train", "val", "test"]:
            for dim_name in self.dimension_names:
                # MAE
                setattr(self, f"{split}_mae_{dim_name}", MeanAbsoluteError())
                # Pearson correlation
                setattr(self, f"{split}_pearson_{dim_name}", PearsonCorrCoef())
                # Spearman correlation
                setattr(self, f"{split}_spearman_{dim_name}", SpearmanCorrCoef())

    def forward(
        self,
        audio_waveform: torch.Tensor,
        midi_tokens: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            audio_waveform: Raw audio waveform [batch, num_samples] at 24kHz
            midi_tokens: OctupleMIDI tokens [batch, events, 8] (optional)
            audio_mask: Audio attention mask [batch, sequence_length]
            midi_mask: MIDI attention mask [batch, events]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - 'scores': Predicted scores [batch, num_dimensions]
                - 'uncertainties': Uncertainty estimates [num_dimensions]
                - 'attention': Attention weights if requested
        """
        # Encode audio (MERT expects raw waveforms at 24kHz)
        # Skip audio encoding in MIDI-only mode (audio_dim=0)
        if self.hparams.audio_dim > 0:
            audio_features, _ = self.audio_encoder(
                audio_waveform=audio_waveform,
                attention_mask=audio_mask,
            )
        else:
            audio_features = None

        # Encode MIDI (if available and encoder exists)
        if midi_tokens is not None and self.midi_encoder is not None:
            midi_features = self.midi_encoder(
                midi_tokens=midi_tokens,
                attention_mask=midi_mask,
            )
        else:
            midi_features = None
            # Debug: Log why MIDI encoding was skipped
            if self.hparams.audio_dim == 0:  # MIDI-only mode
                if midi_tokens is None:
                    # Skip this batch - all MIDI files failed to load
                    # Return a dummy output to avoid breaking the training loop
                    import warnings
                    warnings.warn(
                        "Skipping batch in MIDI-only mode: all MIDI files failed to load. "
                        "This is expected if some MIDI files are corrupted."
                    )
                    # Return None to signal the batch should be skipped
                    return None
                if self.midi_encoder is None:
                    raise ValueError(
                        "MIDI-only mode requires MIDI encoder, but self.midi_encoder is None. "
                        f"Check that midi_dim ({self.hparams.midi_dim}) > 0."
                    )

        # Fuse modalities
        fused_features, fusion_attention = self.fusion(
            audio_features=audio_features,
            midi_features=midi_features,
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
                "fusion": fusion_attention,
                "aggregation": aggregation_attention,
            }

        return result

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        split: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for train/val/test.

        Args:
            batch: Batch dictionary (expects 'audio_waveform', 'labels', optional 'midi_tokens')
            split: 'train', 'val', or 'test'

        Returns:
            Dictionary with loss and predictions
        """
        # Extract batch data
        audio_waveform = batch["audio_waveform"]
        midi_tokens = batch.get("midi_tokens", None)
        targets = batch["labels"]

        # Forward pass
        output = self.forward(
            audio_waveform=audio_waveform,
            midi_tokens=midi_tokens,
        )

        # Check if batch was skipped (all MIDI failed in MIDI-only mode)
        if output is None:
            return None

        predictions = output["scores"]
        uncertainties = output["uncertainties"]

        # Compute loss
        loss_output = self.loss_fn(
            predictions=predictions,
            targets=targets,
            log_vars=self.mtl_head.log_vars,
        )

        total_loss = loss_output["loss"]
        task_losses = loss_output["task_losses"]

        # Log overall loss
        self.log(f"{split}_loss", total_loss, prog_bar=True)

        # Log per-dimension metrics
        for i, dim_name in enumerate(self.dimension_names):
            # Extract dimension predictions and targets
            pred_i = predictions[:, i]
            target_i = targets[:, i]

            # Update metrics
            mae_metric = getattr(self, f"{split}_mae_{dim_name}")
            pearson_metric = getattr(self, f"{split}_pearson_{dim_name}")
            spearman_metric = getattr(self, f"{split}_spearman_{dim_name}")

            mae_metric(pred_i, target_i)
            pearson_metric(pred_i, target_i)
            spearman_metric(pred_i, target_i)

            # Log metrics
            self.log(f"{split}_mae_{dim_name}", mae_metric, on_step=False, on_epoch=True)
            self.log(f"{split}_pearson_{dim_name}", pearson_metric, on_step=False, on_epoch=True)
            self.log(f"{split}_spearman_{dim_name}", spearman_metric, on_step=False, on_epoch=True)

            # Log task loss
            self.log(f"{split}_task_loss_{dim_name}", task_losses[i])

        # Log uncertainties
        if split == "val":
            for i, dim_name in enumerate(self.dimension_names):
                self.log(f"uncertainty_{dim_name}", uncertainties[i])

        return {
            "loss": total_loss,
            "predictions": predictions,
            "targets": targets,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        result = self._shared_step(batch, "train")
        if result is None:
            # Batch was skipped due to missing MIDI data
            # Return a zero loss to continue training
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return result["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        result = self._shared_step(batch, "val")
        # Skip if batch was None (all MIDI failed)
        if result is None:
            return

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        result = self._shared_step(batch, "test")
        # Skip if batch was None (all MIDI failed)
        if result is None:
            return

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Uses different learning rates for:
        - Backbone encoders (lower LR to preserve pre-training)
        - Task heads (higher LR for faster adaptation)
        """
        # Group parameters by learning rate
        backbone_params = []
        head_params = []

        # Audio encoder parameters
        if not self.hparams.freeze_audio_encoder:
            backbone_params.extend(list(self.audio_encoder.parameters()))

        # MIDI encoder parameters (if encoder exists)
        if self.midi_encoder is not None:
            backbone_params.extend(list(self.midi_encoder.parameters()))

        # Fusion, aggregator, and MTL head parameters
        head_params.extend(list(self.fusion.parameters()))
        head_params.extend(list(self.aggregator.parameters()))
        head_params.extend(list(self.mtl_head.parameters()))

        # Create parameter groups (only add non-empty groups)
        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": float(self.hparams.backbone_lr),
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": float(self.hparams.heads_lr),
            })

        # Optimizer
        optimizer = AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_steps,
            eta_min=1e-6,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    print("PyTorch Lightning module loaded successfully")
    print("Complete architecture:")
    print("- MERT-95M audio encoder")
    print("- MIDIBert symbolic encoder")
    print("- Cross-attention fusion")
    print("- BiLSTM hierarchical aggregation")
    print("- Multi-task head (10 dimensions)")
    print("- Uncertainty-weighted loss")
    print("- Metrics: MAE, Pearson r, Spearman ρ")
