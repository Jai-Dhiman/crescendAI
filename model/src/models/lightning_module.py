import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from typing import Dict, List, Optional, Tuple, Any

from .audio_encoder import create_audio_encoder
from .midi_encoder import create_midi_encoder
from .fusion import create_fusion_module
from .aggregation import create_aggregator
from .mtl_head import create_mtl_head
from ..losses.uncertainty_loss import create_loss_function


class PerformanceEvaluationModel(pl.LightningModule):
    """
    Complete PyTorch Lightning module for piano performance evaluation.

    Architecture:
    1. Audio Encoder: MERT-95M (or simplified CNN)
    2. MIDI Encoder: MIDIBert (or simplified LSTM)
    3. Cross-Attention Fusion: Bidirectional attention
    4. Hierarchical Aggregation: BiLSTM + multi-head attention
    5. Multi-Task Head: 10 dimensions with uncertainty

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
        use_mert: bool = True,
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

        # Loss options
        loss_type: str = 'uncertainty',
        label_smoothing: float = 0.1,

        **kwargs
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
            use_mert: Whether to use MERT (vs simplified CNN)
            mert_model_name: HuggingFace model name
            freeze_audio_encoder: Whether to freeze audio encoder
            gradient_checkpointing: Enable gradient checkpointing
            learning_rate: Default learning rate
            backbone_lr: Learning rate for backbone encoders
            heads_lr: Learning rate for task heads
            weight_decay: Weight decay for AdamW
            warmup_steps: Warmup steps for LR scheduler
            max_epochs: Maximum training epochs
            loss_type: Loss function type ('uncertainty' or 'weighted_mse')
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Build model components
        self.audio_encoder = create_audio_encoder(
            use_mert=use_mert,
            model_name=mert_model_name,
            freeze_backbone=freeze_audio_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.midi_encoder = create_midi_encoder(
            use_transformer=True,
            hidden_size=midi_dim,
            num_layers=6,
        )

        self.fusion = create_fusion_module(
            audio_dim=audio_dim,
            midi_dim=midi_dim,
            use_cross_attention=True,
            num_heads=8,
        )

        self.aggregator = create_aggregator(
            input_dim=fusion_dim,
            output_dim=aggregator_dim,
            use_lstm=True,
            lstm_hidden=256,
            lstm_layers=2,
        )

        self.mtl_head = create_mtl_head(
            input_dim=aggregator_dim,
            dimensions=dimension_names,
            use_shared=True,
        )

        # Loss function
        self.loss_fn = create_loss_function(
            loss_type=loss_type,
            num_tasks=num_dimensions,
            label_smoothing=label_smoothing,
        )

        # Metrics (per dimension)
        self.dimension_names = self.mtl_head.get_dimension_names()
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics for each dimension and split."""
        for split in ['train', 'val', 'test']:
            for dim_name in self.dimension_names:
                # MAE
                setattr(
                    self,
                    f'{split}_mae_{dim_name}',
                    MeanAbsoluteError()
                )
                # Pearson correlation
                setattr(
                    self,
                    f'{split}_pearson_{dim_name}',
                    PearsonCorrCoef()
                )
                # Spearman correlation
                setattr(
                    self,
                    f'{split}_spearman_{dim_name}',
                    SpearmanCorrCoef()
                )

    def forward(
        self,
        cqt: torch.Tensor,
        midi_tokens: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            cqt: CQT spectrogram [batch, 168, time_frames]
            midi_tokens: OctupleMIDI tokens [batch, events, 8] (optional)
            audio_mask: Audio attention mask [batch, time_frames]
            midi_mask: MIDI attention mask [batch, events]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - 'scores': Predicted scores [batch, num_dimensions]
                - 'uncertainties': Uncertainty estimates [num_dimensions]
                - 'attention': Attention weights if requested
        """
        # Encode audio
        audio_features, _ = self.audio_encoder(
            cqt=cqt,
            attention_mask=audio_mask,
        )

        # Encode MIDI (if available)
        if midi_tokens is not None:
            midi_features = self.midi_encoder(
                midi_tokens=midi_tokens,
                attention_mask=midi_mask,
            )
        else:
            midi_features = None

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
            'scores': scores,
            'uncertainties': uncertainties,
        }

        if return_attention:
            result['attention'] = {
                'fusion': fusion_attention,
                'aggregation': aggregation_attention,
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
            batch: Batch dictionary
            split: 'train', 'val', or 'test'

        Returns:
            Dictionary with loss and predictions
        """
        # Extract batch data
        cqt = batch['cqt']
        midi_tokens = batch.get('midi_tokens', None)
        targets = batch['labels']

        # Forward pass
        output = self.forward(
            cqt=cqt,
            midi_tokens=midi_tokens,
        )

        predictions = output['scores']
        uncertainties = output['uncertainties']

        # Compute loss
        loss_output = self.loss_fn(
            predictions=predictions,
            targets=targets,
            log_vars=self.mtl_head.log_vars,
        )

        total_loss = loss_output['loss']
        task_losses = loss_output['task_losses']

        # Log overall loss
        self.log(f'{split}_loss', total_loss, prog_bar=True)

        # Log per-dimension metrics
        for i, dim_name in enumerate(self.dimension_names):
            # Extract dimension predictions and targets
            pred_i = predictions[:, i]
            target_i = targets[:, i]

            # Update metrics
            mae_metric = getattr(self, f'{split}_mae_{dim_name}')
            pearson_metric = getattr(self, f'{split}_pearson_{dim_name}')
            spearman_metric = getattr(self, f'{split}_spearman_{dim_name}')

            mae_metric(pred_i, target_i)
            pearson_metric(pred_i, target_i)
            spearman_metric(pred_i, target_i)

            # Log metrics
            self.log(f'{split}_mae_{dim_name}', mae_metric, on_step=False, on_epoch=True)
            self.log(f'{split}_pearson_{dim_name}', pearson_metric, on_step=False, on_epoch=True)
            self.log(f'{split}_spearman_{dim_name}', spearman_metric, on_step=False, on_epoch=True)

            # Log task loss
            self.log(f'{split}_task_loss_{dim_name}', task_losses[i])

        # Log uncertainties
        if split == 'val':
            for i, dim_name in enumerate(self.dimension_names):
                self.log(f'uncertainty_{dim_name}', uncertainties[i])

        return {
            'loss': total_loss,
            'predictions': predictions,
            'targets': targets,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        result = self._shared_step(batch, 'train')
        return result['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        self._shared_step(batch, 'val')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        self._shared_step(batch, 'test')

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

        # MIDI encoder parameters
        backbone_params.extend(list(self.midi_encoder.parameters()))

        # Fusion, aggregator, and MTL head parameters
        head_params.extend(list(self.fusion.parameters()))
        head_params.extend(list(self.aggregator.parameters()))
        head_params.extend(list(self.mtl_head.parameters()))

        # Create parameter groups
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.hparams.backbone_lr,
                'name': 'backbone'
            },
            {
                'params': head_params,
                'lr': self.hparams.heads_lr,
                'name': 'heads'
            }
        ]

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
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
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
