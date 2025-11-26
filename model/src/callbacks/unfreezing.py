"""
Staged unfreezing callback for gradual encoder unfreezing.

Prevents catastrophic forgetting by:
1. Initially freezing pre-trained encoders (MERT, MIDI)
2. Gradually unfreezing layers as training progresses
3. Using lower learning rates for unfrozen backbone layers

Reference: ULMFiT (Howard & Ruder, 2018) for staged unfreezing strategy
"""

from typing import Dict, List, Optional, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class StagedUnfreezingCallback(Callback):
    """
    Callback for staged unfreezing of encoder layers.

    Implements a schedule where:
    - Phase 1: Only train projection heads, fusion, and task heads
    - Phase 2: Unfreeze top N encoder layers with reduced LR
    - Phase 3: Unfreeze remaining layers with further reduced LR

    Example schedule:
        [
            {'epoch': 0, 'unfreeze': [], 'freeze': ['audio_encoder', 'midi_encoder']},
            {'epoch': 5, 'unfreeze': ['audio_encoder.top_4'], 'lr_scale': 0.1},
            {'epoch': 10, 'unfreeze': ['audio_encoder.all', 'midi_encoder.top_2'], 'lr_scale': 0.01},
        ]
    """

    def __init__(
        self,
        schedule: List[Dict[str, Any]],
        verbose: bool = True,
    ):
        """
        Initialize staged unfreezing callback.

        Args:
            schedule: List of unfreezing steps, each containing:
                - epoch: Epoch to apply changes
                - unfreeze: List of components to unfreeze
                - freeze: List of components to freeze (optional)
                - lr_scale: Scale factor for backbone LR (optional)
            verbose: Whether to print unfreezing actions
        """
        super().__init__()
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
        self.verbose = verbose
        self._applied_epochs = set()

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Check if we need to unfreeze layers at this epoch."""
        current_epoch = trainer.current_epoch

        for step in self.schedule:
            step_epoch = step['epoch']

            # Skip if already applied or not yet reached
            if step_epoch in self._applied_epochs:
                continue
            if current_epoch < step_epoch:
                continue

            # Apply this step
            self._apply_schedule_step(pl_module, step, trainer)
            self._applied_epochs.add(step_epoch)

    def _apply_schedule_step(
        self,
        pl_module: pl.LightningModule,
        step: Dict[str, Any],
        trainer: pl.Trainer,
    ) -> None:
        """Apply a single schedule step."""
        epoch = step['epoch']

        if self.verbose:
            print(f"\n[StagedUnfreezing] Epoch {epoch}:")

        # Handle freezing
        freeze_targets = step.get('freeze', [])
        for target in freeze_targets:
            self._freeze_component(pl_module, target)

        # Handle unfreezing
        unfreeze_targets = step.get('unfreeze', [])
        for target in unfreeze_targets:
            self._unfreeze_component(pl_module, target)

        # Handle learning rate scaling
        lr_scale = step.get('lr_scale')
        if lr_scale is not None:
            self._scale_backbone_lr(trainer, pl_module, lr_scale)

    def _freeze_component(
        self,
        pl_module: pl.LightningModule,
        target: str,
    ) -> None:
        """Freeze a model component."""
        if target == 'audio_encoder':
            if hasattr(pl_module, 'audio_encoder'):
                for param in pl_module.audio_encoder.parameters():
                    param.requires_grad = False
                if self.verbose:
                    print(f"  Froze: audio_encoder")

        elif target == 'midi_encoder':
            if hasattr(pl_module, 'midi_encoder') and pl_module.midi_encoder is not None:
                for param in pl_module.midi_encoder.parameters():
                    param.requires_grad = False
                if self.verbose:
                    print(f"  Froze: midi_encoder")

        elif target == 'projection':
            if hasattr(pl_module, 'projection') and pl_module.projection is not None:
                for param in pl_module.projection.parameters():
                    param.requires_grad = False
                if self.verbose:
                    print(f"  Froze: projection")

        else:
            if self.verbose:
                print(f"  Warning: Unknown freeze target '{target}'")

    def _unfreeze_component(
        self,
        pl_module: pl.LightningModule,
        target: str,
    ) -> None:
        """Unfreeze a model component (or part of it)."""
        # Audio encoder targets
        if target == 'audio_encoder.all':
            if hasattr(pl_module, 'audio_encoder'):
                for param in pl_module.audio_encoder.parameters():
                    param.requires_grad = True
                if self.verbose:
                    print(f"  Unfroze: audio_encoder (all layers)")

        elif target.startswith('audio_encoder.top_'):
            num_layers = int(target.split('_')[-1])
            if hasattr(pl_module, 'audio_encoder'):
                # MERT encoder structure - unfreeze top N layers
                self._unfreeze_top_layers(pl_module.audio_encoder, num_layers, 'audio')
                if self.verbose:
                    print(f"  Unfroze: audio_encoder (top {num_layers} layers)")

        # MIDI encoder targets
        elif target == 'midi_encoder.all':
            if hasattr(pl_module, 'midi_encoder') and pl_module.midi_encoder is not None:
                for param in pl_module.midi_encoder.parameters():
                    param.requires_grad = True
                if self.verbose:
                    print(f"  Unfroze: midi_encoder (all layers)")

        elif target.startswith('midi_encoder.top_'):
            num_layers = int(target.split('_')[-1])
            if hasattr(pl_module, 'midi_encoder') and pl_module.midi_encoder is not None:
                pl_module.midi_encoder.unfreeze_top_layers(num_layers)
                if self.verbose:
                    print(f"  Unfroze: midi_encoder (top {num_layers} layers)")

        # Projection head
        elif target == 'projection':
            if hasattr(pl_module, 'projection') and pl_module.projection is not None:
                for param in pl_module.projection.parameters():
                    param.requires_grad = True
                if self.verbose:
                    print(f"  Unfroze: projection")

        else:
            if self.verbose:
                print(f"  Warning: Unknown unfreeze target '{target}'")

    def _unfreeze_top_layers(
        self,
        encoder: Any,
        num_layers: int,
        encoder_type: str,
    ) -> None:
        """Unfreeze the top N layers of an encoder."""
        if encoder_type == 'audio':
            # MERT encoder - access transformer layers
            if hasattr(encoder, 'mert') and hasattr(encoder.mert, 'encoder'):
                layers = encoder.mert.encoder.layers
                total_layers = len(layers)
                for i in range(total_layers - num_layers, total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True

                # Also unfreeze the final layer norm
                if hasattr(encoder.mert.encoder, 'layer_norm'):
                    for param in encoder.mert.encoder.layer_norm.parameters():
                        param.requires_grad = True

    def _scale_backbone_lr(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        scale: float,
    ) -> None:
        """Scale the learning rate for backbone parameters."""
        optimizer = trainer.optimizers[0]

        # Find backbone param group (usually the first one)
        for i, param_group in enumerate(optimizer.param_groups):
            if 'backbone' in str(param_group.get('name', '')).lower() or i == 0:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * scale
                if self.verbose:
                    print(f"  Scaled backbone LR: {old_lr:.2e} -> {param_group['lr']:.2e}")
                break


def create_default_unfreezing_schedule(
    freeze_epochs: int = 5,
    partial_unfreeze_epochs: int = 5,
    audio_top_layers: int = 4,
    midi_top_layers: int = 2,
) -> List[Dict[str, Any]]:
    """
    Create a default staged unfreezing schedule.

    Args:
        freeze_epochs: Epochs to keep encoders frozen
        partial_unfreeze_epochs: Epochs with partial unfreezing
        audio_top_layers: Number of audio encoder layers to unfreeze first
        midi_top_layers: Number of MIDI encoder layers to unfreeze first

    Returns:
        Schedule list for StagedUnfreezingCallback
    """
    return [
        # Phase 1: Freeze encoders, train only heads
        {
            'epoch': 0,
            'freeze': ['audio_encoder', 'midi_encoder'],
            'unfreeze': ['projection'],
        },
        # Phase 2: Unfreeze top layers with reduced LR
        {
            'epoch': freeze_epochs,
            'unfreeze': [f'audio_encoder.top_{audio_top_layers}', f'midi_encoder.top_{midi_top_layers}'],
            'lr_scale': 0.1,
        },
        # Phase 3: Unfreeze all with further reduced LR
        {
            'epoch': freeze_epochs + partial_unfreeze_epochs,
            'unfreeze': ['audio_encoder.all', 'midi_encoder.all'],
            'lr_scale': 0.1,  # Cumulative: 0.1 * 0.1 = 0.01 of original
        },
    ]


if __name__ == "__main__":
    print("Staged Unfreezing Callback")
    print("- Prevents catastrophic forgetting")
    print("- Gradually unfreezes encoder layers")
    print("- Uses differential learning rates")

    # Example schedule
    schedule = create_default_unfreezing_schedule()
    print("\nDefault schedule:")
    for step in schedule:
        print(f"  Epoch {step['epoch']}: {step}")
