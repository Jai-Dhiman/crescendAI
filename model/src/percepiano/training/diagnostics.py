"""
Comprehensive diagnostics for PercePiano training.

This module provides callbacks and utilities to diagnose why the hierarchical
components (beat/measure attention) might not be contributing to model performance.

Key diagnostics:
1. Index validation - Check beat/measure indices for issues
2. Activation statistics - Track variance through the hierarchy
3. Attention analysis - Detect collapsed or uniform attention
4. Gradient flow - Verify gradients reach all components
5. Ablation testing - Compare with/without hierarchy
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


@dataclass
class DiagnosticStats:
    """Container for diagnostic statistics."""

    # Index statistics
    beat_idx_min: float = 0.0
    beat_idx_max: float = 0.0
    measure_idx_min: float = 0.0
    measure_idx_max: float = 0.0
    negative_zero_shifted_count: int = 0

    # Activation variances (std)
    input_std: float = 0.0
    x_embedded_std: float = 0.0
    note_out_std: float = 0.0
    voice_out_std: float = 0.0
    hidden_out_std: float = 0.0
    beat_nodes_std: float = 0.0
    beat_out_std: float = 0.0
    beat_spanned_std: float = 0.0
    measure_nodes_std: float = 0.0
    measure_out_std: float = 0.0
    measure_spanned_std: float = 0.0
    total_note_cat_std: float = 0.0
    contracted_std: float = 0.0
    aggregated_std: float = 0.0
    logits_std: float = 0.0
    predictions_std: float = 0.0

    # Attention statistics
    beat_attention_entropy: float = 0.0
    measure_attention_entropy: float = 0.0
    final_attention_entropy: float = 0.0

    # Gradient norms (populated during training)
    grad_note_lstm: float = 0.0
    grad_beat_attention: float = 0.0
    grad_measure_attention: float = 0.0
    grad_performance_contractor: float = 0.0
    grad_final_attention: float = 0.0
    grad_prediction_head: float = 0.0

    # Contribution analysis
    hidden_out_contribution: float = 0.0  # Fraction of total_note_cat variance from hidden_out
    beat_spanned_contribution: float = 0.0
    measure_spanned_contribution: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items()}


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention weights.

    Low entropy = attention collapsed to few positions (bad)
    High entropy = attention spread evenly (might be too uniform)
    Ideal = moderate entropy with meaningful peaks

    Args:
        attention_weights: Tensor of shape (B, T, ...) with softmaxed weights

    Returns:
        Mean entropy across batch
    """
    # Flatten to (B, T) if needed
    if attention_weights.dim() > 2:
        attention_weights = attention_weights.mean(dim=tuple(range(2, attention_weights.dim())))

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attention_weights = attention_weights.clamp(min=eps)

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)

    # Normalize by max possible entropy (uniform distribution)
    max_entropy = torch.log(torch.tensor(attention_weights.shape[-1], dtype=torch.float))
    normalized_entropy = entropy / max_entropy

    return normalized_entropy.mean().item()


def analyze_indices(
    beat_numbers: torch.Tensor,
    measure_numbers: torch.Tensor,
) -> Dict[str, Any]:
    """
    Analyze beat and measure indices for potential issues.

    Note: Padding positions have value 0. Valid positions have value >= 1.
    This function analyzes ONLY valid positions (> 0) to avoid confusion.

    Returns dict with:
    - Index ranges (excluding padding zeros)
    - Gaps in indices
    - Negative values after zero-shifting (only in valid positions)
    - Boundary detection issues
    """
    results = {}

    # Only consider non-padding values (> 0) for range
    valid_beat_mask = beat_numbers > 0
    valid_measure_mask = measure_numbers > 0

    if valid_beat_mask.any():
        results['beat_min'] = beat_numbers[valid_beat_mask].min().item()
        results['beat_max'] = beat_numbers[valid_beat_mask].max().item()
    else:
        results['beat_min'] = 0
        results['beat_max'] = 0

    if valid_measure_mask.any():
        results['measure_min'] = measure_numbers[valid_measure_mask].min().item()
        results['measure_max'] = measure_numbers[valid_measure_mask].max().item()
    else:
        results['measure_min'] = 0
        results['measure_max'] = 0

    # Count negative zero-shifted ONLY in valid positions
    # This is the real issue indicator - if valid positions become negative after zero-shifting
    negative_beat_count = 0
    negative_measure_count = 0
    beat_zero_shifted_min = float('inf')
    beat_zero_shifted_max = float('-inf')
    measure_zero_shifted_min = float('inf')
    measure_zero_shifted_max = float('-inf')

    for i in range(beat_numbers.shape[0]):
        # Beat analysis - only valid positions
        valid_mask = beat_numbers[i] > 0
        if valid_mask.any():
            valid_beats = beat_numbers[i][valid_mask]
            # Zero-shift relative to first valid beat
            zero_shifted = valid_beats - valid_beats[0]
            negative_beat_count += (zero_shifted < 0).sum().item()
            beat_zero_shifted_min = min(beat_zero_shifted_min, zero_shifted.min().item())
            beat_zero_shifted_max = max(beat_zero_shifted_max, zero_shifted.max().item())

        # Measure analysis - only valid positions
        valid_mask = measure_numbers[i] > 0
        if valid_mask.any():
            valid_measures = measure_numbers[i][valid_mask]
            zero_shifted = valid_measures - valid_measures[0]
            negative_measure_count += (zero_shifted < 0).sum().item()
            measure_zero_shifted_min = min(measure_zero_shifted_min, zero_shifted.min().item())
            measure_zero_shifted_max = max(measure_zero_shifted_max, zero_shifted.max().item())

    results['beat_zero_shifted_min'] = beat_zero_shifted_min if beat_zero_shifted_min != float('inf') else 0
    results['beat_zero_shifted_max'] = beat_zero_shifted_max if beat_zero_shifted_max != float('-inf') else 0
    results['negative_beat_count'] = negative_beat_count

    results['measure_zero_shifted_min'] = measure_zero_shifted_min if measure_zero_shifted_min != float('inf') else 0
    results['measure_zero_shifted_max'] = measure_zero_shifted_max if measure_zero_shifted_max != float('-inf') else 0
    results['negative_measure_count'] = negative_measure_count

    # Check if indices are sequential (after densification)
    # For each batch item, check if unique values are 0,1,2,3...
    sequential_issues = 0
    for i in range(beat_numbers.shape[0]):
        valid_mask = beat_numbers[i] > 0
        if valid_mask.any():
            valid_beats = beat_numbers[i][valid_mask]
            unique_beats = torch.unique(valid_beats)
            expected = torch.arange(unique_beats.min(), unique_beats.max() + 1, device=unique_beats.device)
            if len(unique_beats) != len(expected):
                sequential_issues += 1
    results['non_sequential_samples'] = sequential_issues

    # Check boundary detection (excluding padding-to-valid transitions)
    diff = beat_numbers[:, 1:] - beat_numbers[:, :-1]
    # Only count transitions between valid positions
    valid_transitions = (beat_numbers[:, :-1] > 0) & (beat_numbers[:, 1:] > 0)
    results['diff_equals_1_count'] = ((diff == 1) & valid_transitions).sum().item()
    results['diff_greater_0_count'] = ((diff > 0) & valid_transitions).sum().item()
    results['diff_less_0_count'] = ((diff < 0) & valid_transitions).sum().item()

    return results


class DiagnosticCallback(Callback):
    """
    PyTorch Lightning callback for comprehensive training diagnostics.

    Captures:
    - Activation statistics at each level of the hierarchy
    - Attention weight distributions
    - Gradient flow through components
    - Index validation

    Usage:
        callback = DiagnosticCallback(log_every_n_steps=100)
        trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        detailed_analysis_every_n_epochs: int = 5,
        save_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detailed_analysis_every_n_epochs = detailed_analysis_every_n_epochs
        self.save_dir = Path(save_dir) if save_dir else None

        self.step_stats: List[DiagnosticStats] = []
        self.epoch_stats: List[Dict[str, Any]] = []

        # Gradient hooks
        self._gradient_norms: Dict[str, float] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Register gradient hooks."""
        if stage == "fit":
            self._register_gradient_hooks(pl_module)

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Remove gradient hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _register_gradient_hooks(self, pl_module: pl.LightningModule) -> None:
        """Register hooks to capture gradient norms."""

        def make_hook(name: str):
            def hook(grad):
                if grad is not None:
                    self._gradient_norms[name] = grad.norm().item()
            return hook

        # Register hooks on key parameters
        model = pl_module
        if hasattr(model, 'han_encoder'):
            han = model.han_encoder

            # Note LSTM
            for name, param in han.note_lstm.named_parameters():
                if param.requires_grad:
                    h = param.register_hook(make_hook(f'note_lstm.{name}'))
                    self._hooks.append(h)
                    break  # Just one parameter is enough

            # Beat attention context vector
            if hasattr(han, 'beat_attention'):
                h = han.beat_attention.context_vector.register_hook(make_hook('beat_attention.context_vector'))
                self._hooks.append(h)

            # Measure attention context vector
            if hasattr(han, 'measure_attention'):
                h = han.measure_attention.context_vector.register_hook(make_hook('measure_attention.context_vector'))
                self._hooks.append(h)

        # Performance contractor
        if hasattr(model, 'performance_contractor'):
            for name, param in model.performance_contractor.named_parameters():
                if param.requires_grad:
                    h = param.register_hook(make_hook(f'performance_contractor.{name}'))
                    self._hooks.append(h)
                    break

        # Final attention
        if hasattr(model, 'final_attention'):
            for name, param in model.final_attention.named_parameters():
                if param.requires_grad:
                    h = param.register_hook(make_hook(f'final_attention.{name}'))
                    self._hooks.append(h)
                    break

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Log diagnostics periodically during training."""
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            # Log gradient norms
            for name, norm in self._gradient_norms.items():
                pl_module.log(f'grad/{name}', norm, on_step=True, on_epoch=False)

            # Log gradient summary
            if self._gradient_norms:
                mean_grad = np.mean(list(self._gradient_norms.values()))
                pl_module.log('grad/mean_norm', mean_grad, on_step=True, on_epoch=False)

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run detailed diagnostics at start of validation."""
        current_epoch = trainer.current_epoch

        if current_epoch % self.detailed_analysis_every_n_epochs == 0:
            # Get a batch from validation dataloader
            val_loader = trainer.val_dataloaders
            if val_loader is None:
                return

            try:
                batch = next(iter(val_loader))
            except StopIteration:
                return

            # Move batch to device
            device = pl_module.device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Run diagnostic forward pass
            stats = self._run_diagnostic_forward(pl_module, batch)

            # Log to tensorboard/wandb
            for key, value in stats.to_dict().items():
                pl_module.log(f'diag/{key}', value, on_epoch=True)

            # Store for later analysis
            self.epoch_stats.append({
                'epoch': current_epoch,
                **stats.to_dict()
            })

            # Print summary
            self._print_diagnostic_summary(stats, current_epoch)

    def _run_diagnostic_forward(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
    ) -> DiagnosticStats:
        """Run forward pass with detailed activation capture.

        Supports multiple model types:
        - PercePianoHAN: has han_encoder attribute
        - PercePianoBiLSTMBaseline: has lstm, note_attention, note_contractor
        - PercePianoBaselinePlusBeat: has lstm, beat_attention, beat_lstm, note_contractor
        """
        stats = DiagnosticStats()

        pl_module.eval()
        device = pl_module.device

        with torch.no_grad():
            # Extract inputs
            input_features = batch['input_features']
            note_locations = {
                'beat': batch['note_locations_beat'].to(device),
                'measure': batch['note_locations_measure'].to(device),
                'voice': batch['note_locations_voice'].to(device),
            }

            # Handle PackedSequence
            from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
            if isinstance(input_features, PackedSequence):
                input_features, _ = pad_packed_sequence(input_features, batch_first=True)

            # Ensure input is on correct device
            input_features = input_features.to(device)

            # Index analysis (can stay on CPU)
            idx_stats = analyze_indices(
                note_locations['beat'].cpu(),
                note_locations['measure'].cpu(),
            )
            stats.beat_idx_min = idx_stats['beat_min']
            stats.beat_idx_max = idx_stats['beat_max']
            stats.measure_idx_min = idx_stats['measure_min']
            stats.measure_idx_max = idx_stats['measure_max']
            stats.negative_zero_shifted_count = idx_stats['negative_beat_count']

            # Input stats
            stats.input_std = input_features.std().item()

            # Detect model type and dispatch to appropriate handler
            model_type = self._detect_model_type(pl_module)

            if model_type == 'han':
                # Full HAN model with han_encoder
                stats = self._diagnostic_han(pl_module, input_features, note_locations, stats)
            elif model_type == 'note_voice_beat':
                # True incremental: note + voice + beat
                stats = self._diagnostic_note_voice_beat(pl_module, input_features, note_locations, stats)
            elif model_type == 'note_voice':
                # True incremental: note + voice (parallel)
                stats = self._diagnostic_note_voice(pl_module, input_features, note_locations, stats)
            elif model_type == 'note_only':
                # True incremental: note only
                stats = self._diagnostic_note_only(pl_module, input_features, note_locations, stats)
            elif model_type == 'baseline_beat':
                # Old-style Baseline + Beat hierarchy model
                stats = self._diagnostic_baseline_beat(pl_module, input_features, note_locations, stats)
            elif model_type == 'baseline':
                # Simple 7-layer BiLSTM baseline
                stats = self._diagnostic_baseline(pl_module, input_features, note_locations, stats)
            else:
                # Unknown model type - log warning
                print(f"[DiagnosticCallback] WARNING: Unknown model type '{model_type}', skipping detailed diagnostics")

        pl_module.train()
        return stats

    def _detect_model_type(self, pl_module: pl.LightningModule) -> str:
        """Detect model type based on attributes.

        Returns one of:
        - 'han': Full HAN with han_encoder
        - 'note_voice_beat': True incremental with note + voice + beat
        - 'note_voice': True incremental with note + voice (parallel)
        - 'note_only': True incremental with only note LSTM
        - 'baseline_beat': Old-style baseline + beat hierarchy
        - 'baseline': Simple 7-layer BiLSTM baseline
        - 'unknown': Unrecognized model type
        """
        if hasattr(pl_module, 'han_encoder'):
            return 'han'
        # True incremental models (Round 17) - check in order of specificity
        elif hasattr(pl_module, 'beat_attention') and hasattr(pl_module, 'voice_lstm'):
            return 'note_voice_beat'
        elif hasattr(pl_module, 'voice_lstm') and hasattr(pl_module, 'note_lstm'):
            return 'note_voice'
        elif hasattr(pl_module, 'note_lstm') and not hasattr(pl_module, 'voice_lstm'):
            return 'note_only'
        # Old incremental models
        elif hasattr(pl_module, 'beat_attention') and hasattr(pl_module, 'beat_lstm'):
            return 'baseline_beat'
        elif hasattr(pl_module, 'lstm') and hasattr(pl_module, 'note_attention'):
            return 'baseline'
        return 'unknown'

    def _diagnostic_baseline(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoBiLSTMBaseline."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import compute_actual_lengths

        # Embed input
        x_embedded = pl_module.note_embedder(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]

        # LSTM
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = pl_module.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = lstm_out.std().item()

        # Contractor
        contracted = pl_module.note_contractor(lstm_out)
        stats.contracted_std = contracted.std().item()

        # Final attention
        aggregated = pl_module.note_attention(contracted, diagnose=False)
        stats.aggregated_std = aggregated.std().item()

        # Prediction head
        logits = pl_module.out_fc(aggregated)
        stats.logits_std = logits.std().item()
        predictions = torch.sigmoid(logits)
        stats.predictions_std = predictions.std().item()

        return stats

    def _diagnostic_note_only(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoNoteOnly (true incremental)."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import compute_actual_lengths

        # Embed input
        x_embedded = pl_module.note_embedder(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]

        # Note LSTM (2-layer, not 7-layer like baseline)
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        note_out, _ = pl_module.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = note_out.std().item()

        # Contractor (512 -> 512)
        contracted = pl_module.note_contractor(note_out)
        stats.contracted_std = contracted.std().item()

        # Final attention
        aggregated = pl_module.note_attention(contracted, diagnose=False)
        stats.aggregated_std = aggregated.std().item()

        # Prediction head
        logits = pl_module.out_fc(aggregated)
        stats.logits_std = logits.std().item()
        predictions = torch.sigmoid(logits)
        stats.predictions_std = predictions.std().item()

        return stats

    def _diagnostic_note_voice(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoNoteVoice (true incremental with parallel voice)."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import compute_actual_lengths

        # Embed input
        x_embedded = pl_module.note_embedder(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]
        batch_size = input_features.shape[0]

        # Note LSTM (processes same input as voice)
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        note_out, _ = pl_module.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = note_out.std().item()

        # Voice LSTM (processes same embedded input in PARALLEL - key architectural insight)
        voice_numbers = note_locations['voice']
        max_voice = voice_numbers.max().item()

        # Run voice LSTM
        voice_out = pl_module.run_voice_lstm(x_embedded, voice_numbers, max_voice, batch_size, seq_len)
        stats.voice_out_std = voice_out.std().item()

        # Concatenate note and voice (forms 1024-dim hidden_out)
        hidden_out = torch.cat([note_out, voice_out], dim=2)
        stats.hidden_out_std = hidden_out.std().item()

        # Contractor (1024 -> 512)
        contracted = pl_module.note_contractor(hidden_out)
        stats.contracted_std = contracted.std().item()

        # Final attention
        aggregated = pl_module.note_attention(contracted, diagnose=False)
        stats.aggregated_std = aggregated.std().item()

        # Prediction head
        logits = pl_module.out_fc(aggregated)
        stats.logits_std = logits.std().item()
        predictions = torch.sigmoid(logits)
        stats.predictions_std = predictions.std().item()

        return stats

    def _diagnostic_note_voice_beat(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoNoteVoiceBeat (true incremental with beat hierarchy)."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import (
            compute_actual_lengths, make_higher_node,
            run_hierarchy_lstm_with_pack, span_beat_to_note_num
        )

        # Embed input
        x_embedded = pl_module.note_embedder(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]
        batch_size = input_features.shape[0]

        # Note LSTM
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        note_out, _ = pl_module.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = note_out.std().item()

        # Voice LSTM (parallel with note)
        voice_numbers = note_locations['voice']
        max_voice = voice_numbers.max().item()
        voice_out = pl_module.run_voice_lstm(x_embedded, voice_numbers, max_voice, batch_size, seq_len)
        stats.voice_out_std = voice_out.std().item()

        # Concatenate note and voice (1024-dim)
        hidden_out = torch.cat([note_out, voice_out], dim=2)
        stats.hidden_out_std = hidden_out.std().item()

        # Beat hierarchy
        beat_numbers = note_locations['beat']

        # Create beat nodes via attention
        beat_nodes, beat_attn = make_higher_node(
            hidden_out, beat_numbers, pl_module.beat_attention,
            return_attention=True
        )
        stats.beat_nodes_std = beat_nodes.std().item()

        # Beat attention entropy
        if beat_attn is not None and len(beat_attn) > 0:
            entropies = []
            for attn in beat_attn:
                if attn is not None and attn.numel() > 1:
                    attn_flat = attn.flatten()
                    attn_flat = attn_flat[attn_flat > 1e-10]
                    if len(attn_flat) > 1:
                        entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum()
                        max_entropy = torch.log(torch.tensor(len(attn_flat), dtype=torch.float))
                        normalized_entropy = entropy / max_entropy
                        entropies.append(normalized_entropy.item())
            if entropies:
                stats.beat_attention_entropy = sum(entropies) / len(entropies)

        # Beat LSTM
        max_beat = beat_numbers.max().item()
        beat_out = run_hierarchy_lstm_with_pack(
            beat_nodes, pl_module.beat_lstm, batch_size, max_beat
        )
        stats.beat_out_std = beat_out.std().item()

        # Span back to note level
        beat_spanned = span_beat_to_note_num(beat_out, beat_numbers, seq_len)
        stats.beat_spanned_std = beat_spanned.std().item()

        # Concatenate for final output (1024 + 512 = 1536)
        total_note_cat = torch.cat([hidden_out, beat_spanned], dim=2)
        stats.total_note_cat_std = total_note_cat.std().item()

        # Contractor (1536 -> 512)
        contracted = pl_module.note_contractor(total_note_cat)
        stats.contracted_std = contracted.std().item()

        # Final attention
        aggregated = pl_module.note_attention(contracted, diagnose=False)
        stats.aggregated_std = aggregated.std().item()

        # Prediction head
        logits = pl_module.out_fc(aggregated)
        stats.logits_std = logits.std().item()
        predictions = torch.sigmoid(logits)
        stats.predictions_std = predictions.std().item()

        return stats

    def _diagnostic_baseline_beat(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoBaselinePlusBeat."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import (
            compute_actual_lengths, make_higher_node,
            run_hierarchy_lstm_with_pack, span_beat_to_note_num
        )

        # Embed input
        x_embedded = pl_module.note_embedder(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]
        batch_size = input_features.shape[0]

        # LSTM
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = pl_module.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = lstm_out.std().item()
        # For baseline+beat, hidden_out is just lstm_out
        stats.hidden_out_std = lstm_out.std().item()

        # Beat hierarchy
        beat_nodes = make_higher_node(
            lstm_out, pl_module.beat_attention,
            note_locations['beat'], note_locations['beat'],
            lower_is_note=True, actual_lengths=actual_lengths
        )
        stats.beat_nodes_std = beat_nodes.std().item()

        beat_out = run_hierarchy_lstm_with_pack(beat_nodes, pl_module.beat_lstm)
        stats.beat_out_std = beat_out.std().item()

        beat_spanned = span_beat_to_note_num(beat_out, note_locations['beat'], actual_lengths)
        if beat_spanned.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - beat_spanned.shape[1], beat_spanned.shape[2],
                device=beat_spanned.device
            )
            beat_spanned = torch.cat([beat_spanned, padding], dim=1)
        stats.beat_spanned_std = beat_spanned.std().item()

        # Beat attention entropy
        beat_similarity = pl_module.beat_attention.get_attention(lstm_out)
        beat_attention_weights = torch.softmax(beat_similarity, dim=1)
        stats.beat_attention_entropy = compute_attention_entropy(beat_attention_weights)

        # Contribution analysis (lstm vs beat for this model)
        total_var = lstm_out.var() + beat_spanned.var()
        if total_var > 0:
            stats.hidden_out_contribution = (lstm_out.var() / total_var).item()
            stats.beat_spanned_contribution = (beat_spanned.var() / total_var).item()

        # Concatenate and contract
        combined = torch.cat([lstm_out, beat_spanned], dim=-1)
        stats.total_note_cat_std = combined.std().item()

        contracted = pl_module.note_contractor(combined)
        stats.contracted_std = contracted.std().item()

        # Final attention
        aggregated = pl_module.note_attention(contracted, diagnose=False)
        stats.aggregated_std = aggregated.std().item()

        # Prediction head
        logits = pl_module.out_fc(aggregated)
        stats.logits_std = logits.std().item()
        predictions = torch.sigmoid(logits)
        stats.predictions_std = predictions.std().item()

        return stats

    def _diagnostic_han(
        self,
        pl_module: pl.LightningModule,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        stats: DiagnosticStats,
    ) -> DiagnosticStats:
        """Diagnostic path for PercePianoHAN (full hierarchical model)."""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from ..models.hierarchy_utils import (
            compute_actual_lengths, make_higher_node,
            run_hierarchy_lstm_with_pack, span_beat_to_note_num
        )

        han = pl_module.han_encoder

        # Project input
        x_embedded = han.note_fc(input_features)
        stats.x_embedded_std = x_embedded.std().item()

        # Compute actual lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])
        seq_len = input_features.shape[1]

        # Note LSTM
        x_packed = pack_padded_sequence(
            x_embedded,
            actual_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        note_out, _ = han.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=seq_len)
        stats.note_out_std = note_out.std().item()

        # Voice processing
        voice_out = han._run_voice_processing(x_embedded, note_locations['voice'], actual_lengths)
        stats.voice_out_std = voice_out.std().item()

        # Combined hidden
        hidden_out = torch.cat([note_out, voice_out], dim=-1)
        stats.hidden_out_std = hidden_out.std().item()

        # Beat aggregation
        beat_nodes = make_higher_node(
            hidden_out,
            han.beat_attention,
            note_locations['beat'],
            note_locations['beat'],
            lower_is_note=True,
            actual_lengths=actual_lengths,
        )
        stats.beat_nodes_std = beat_nodes.std().item()

        beat_out = run_hierarchy_lstm_with_pack(beat_nodes, han.beat_lstm)
        stats.beat_out_std = beat_out.std().item()

        beat_spanned = span_beat_to_note_num(beat_out, note_locations['beat'], actual_lengths)
        # Pad if needed
        if beat_spanned.shape[1] < hidden_out.shape[1]:
            padding = torch.zeros(
                beat_spanned.shape[0],
                hidden_out.shape[1] - beat_spanned.shape[1],
                beat_spanned.shape[2],
                device=beat_spanned.device,
            )
            beat_spanned = torch.cat([beat_spanned, padding], dim=1)
        stats.beat_spanned_std = beat_spanned.std().item()

        # Measure aggregation
        measure_nodes = make_higher_node(
            beat_out,
            han.measure_attention,
            note_locations['beat'],
            note_locations['measure'],
            actual_lengths=actual_lengths,
        )
        stats.measure_nodes_std = measure_nodes.std().item()

        measure_out = run_hierarchy_lstm_with_pack(measure_nodes, han.measure_lstm)
        stats.measure_out_std = measure_out.std().item()

        measure_spanned = span_beat_to_note_num(measure_out, note_locations['measure'], actual_lengths)
        if measure_spanned.shape[1] < hidden_out.shape[1]:
            padding = torch.zeros(
                measure_spanned.shape[0],
                hidden_out.shape[1] - measure_spanned.shape[1],
                measure_spanned.shape[2],
                device=measure_spanned.device,
            )
            measure_spanned = torch.cat([measure_spanned, padding], dim=1)
        stats.measure_spanned_std = measure_spanned.std().item()

        # Contribution analysis
        total_var = hidden_out.var() + beat_spanned.var() + measure_spanned.var()
        if total_var > 0:
            stats.hidden_out_contribution = (hidden_out.var() / total_var).item()
            stats.beat_spanned_contribution = (beat_spanned.var() / total_var).item()
            stats.measure_spanned_contribution = (measure_spanned.var() / total_var).item()

        # Total note cat
        total_note_cat = torch.cat([hidden_out, beat_spanned, measure_spanned], dim=-1)
        stats.total_note_cat_std = total_note_cat.std().item()

        # Beat attention entropy
        beat_similarity = han.beat_attention.get_attention(hidden_out)
        beat_attention_weights = torch.softmax(beat_similarity, dim=1)
        stats.beat_attention_entropy = compute_attention_entropy(beat_attention_weights)

        # Measure attention entropy
        measure_similarity = han.measure_attention.get_attention(beat_out)
        measure_attention_weights = torch.softmax(measure_similarity, dim=1)
        stats.measure_attention_entropy = compute_attention_entropy(measure_attention_weights)

        # Performance contractor
        if hasattr(pl_module, 'performance_contractor'):
            contracted = pl_module.performance_contractor(total_note_cat)
            stats.contracted_std = contracted.std().item()

        # Final attention
        if hasattr(pl_module, 'final_attention'):
            attention_mask = note_locations['beat'] > 0
            aggregated = pl_module.final_attention(contracted, mask=attention_mask)
            stats.aggregated_std = aggregated.std().item()

        # Prediction head
        if hasattr(pl_module, 'prediction_head'):
            logits = pl_module.prediction_head(aggregated)
            stats.logits_std = logits.std().item()
            predictions = torch.sigmoid(logits)
            stats.predictions_std = predictions.std().item()

        return stats

    def _print_diagnostic_summary(self, stats: DiagnosticStats, epoch: int) -> None:
        """Print formatted diagnostic summary."""
        print("\n" + "=" * 70)
        print(f"DIAGNOSTIC SUMMARY - Epoch {epoch}")
        print("=" * 70)

        # Index issues
        print("\n[1] INDEX ANALYSIS:")
        print(f"  Beat range: [{stats.beat_idx_min:.0f}, {stats.beat_idx_max:.0f}]")
        print(f"  Measure range: [{stats.measure_idx_min:.0f}, {stats.measure_idx_max:.0f}]")
        if stats.negative_zero_shifted_count > 0:
            print(f"  [WARNING] Negative zero-shifted values: {stats.negative_zero_shifted_count}")
        else:
            print(f"  Zero-shifted values: OK (no negatives)")

        # Activation variances
        print("\n[2] ACTIVATION VARIANCES (std):")
        print(f"  {'Component':<25} {'Std':>10} {'Status':<15}")
        print(f"  {'-'*25} {'-'*10} {'-'*15}")

        activations = [
            ('input', stats.input_std, 0.3, 0.6),
            ('x_embedded', stats.x_embedded_std, 0.15, 0.4),
            ('note_out (LSTM)', stats.note_out_std, 0.1, 0.4),
            ('voice_out', stats.voice_out_std, 0.1, 0.4),
            ('hidden_out', stats.hidden_out_std, 0.1, 0.4),
            ('beat_nodes', stats.beat_nodes_std, 0.05, 0.3),
            ('beat_out', stats.beat_out_std, 0.05, 0.3),
            ('beat_spanned', stats.beat_spanned_std, 0.05, 0.3),
            ('measure_nodes', stats.measure_nodes_std, 0.05, 0.3),
            ('measure_out', stats.measure_out_std, 0.05, 0.3),
            ('measure_spanned', stats.measure_spanned_std, 0.05, 0.3),
            ('total_note_cat', stats.total_note_cat_std, 0.05, 0.3),
            ('contracted', stats.contracted_std, 0.05, 0.3),
            ('aggregated', stats.aggregated_std, 0.1, 0.5),
            ('logits', stats.logits_std, 0.5, 3.0),
            ('predictions', stats.predictions_std, 0.1, 0.25),
        ]

        for name, std, low_thresh, high_thresh in activations:
            if std < low_thresh:
                status = "[LOW - ISSUE]"
            elif std > high_thresh:
                status = "[HIGH]"
            else:
                status = "[OK]"
            print(f"  {name:<25} {std:>10.4f} {status:<15}")

        # Attention entropy
        print("\n[3] ATTENTION ENTROPY (0=collapsed, 1=uniform):")
        print(f"  Beat attention:    {stats.beat_attention_entropy:.3f}")
        print(f"  Measure attention: {stats.measure_attention_entropy:.3f}")

        # Contribution analysis
        print("\n[4] HIERARCHY CONTRIBUTION (fraction of total variance):")
        print(f"  hidden_out (Bi-LSTM):  {stats.hidden_out_contribution:.1%}")
        print(f"  beat_spanned:          {stats.beat_spanned_contribution:.1%}")
        print(f"  measure_spanned:       {stats.measure_spanned_contribution:.1%}")

        if stats.beat_spanned_contribution < 0.1:
            print("  [WARNING] Beat hierarchy contributing < 10% - may not be learning!")
        if stats.measure_spanned_contribution < 0.05:
            print("  [WARNING] Measure hierarchy contributing < 5% - may not be learning!")

        print("=" * 70 + "\n")

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run comprehensive diagnostics at the end of training.

        This provides a final summary of the trained model's internal state,
        helping verify that:
        1. All hierarchy levels are contributing appropriately
        2. Attention mechanisms are not collapsed or uniform
        3. Activation variances are in healthy ranges
        """
        print("\n" + "=" * 70)
        print("END OF TRAINING DIAGNOSTICS")
        print("=" * 70)

        model_type = self._detect_model_type(pl_module)
        print(f"\nModel type: {model_type}")
        print(f"Total epochs trained: {trainer.current_epoch + 1}")

        # Get validation dataloader
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            print("[WARNING] No validation dataloader available for final diagnostics")
            return

        # Run diagnostics on multiple batches for more stable statistics
        all_stats = []
        num_batches = min(5, len(val_loader))  # Use up to 5 batches

        print(f"\nRunning diagnostics on {num_batches} validation batches...")

        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break

            # Move batch to device
            device = pl_module.device
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            try:
                stats = self._run_diagnostic_forward(pl_module, batch_on_device)
                all_stats.append(stats)
            except Exception as e:
                print(f"  [WARNING] Batch {i} diagnostic failed: {e}")

        if not all_stats:
            print("[ERROR] No successful diagnostic runs")
            return

        # Aggregate statistics across batches
        avg_stats = DiagnosticStats()
        for field_name in [
            'input_std', 'x_embedded_std', 'note_out_std', 'voice_out_std',
            'hidden_out_std', 'beat_nodes_std', 'beat_out_std', 'beat_spanned_std',
            'measure_nodes_std', 'measure_out_std', 'measure_spanned_std',
            'total_note_cat_std', 'contracted_std', 'aggregated_std',
            'logits_std', 'predictions_std', 'beat_attention_entropy',
            'measure_attention_entropy', 'hidden_out_contribution',
            'beat_spanned_contribution', 'measure_spanned_contribution',
        ]:
            values = [getattr(s, field_name) for s in all_stats if getattr(s, field_name) > 0]
            if values:
                setattr(avg_stats, field_name, sum(values) / len(values))

        # Use first batch for index stats (they should be consistent)
        avg_stats.beat_idx_min = all_stats[0].beat_idx_min
        avg_stats.beat_idx_max = all_stats[0].beat_idx_max
        avg_stats.measure_idx_min = all_stats[0].measure_idx_min
        avg_stats.measure_idx_max = all_stats[0].measure_idx_max
        avg_stats.negative_zero_shifted_count = all_stats[0].negative_zero_shifted_count

        # Print comprehensive summary
        self._print_diagnostic_summary(avg_stats, trainer.current_epoch)

        # Print model-specific insights
        self._print_model_insights(model_type, avg_stats)

        # Save stats if save_dir is configured
        if self.save_dir:
            self.epoch_stats.append({
                'epoch': trainer.current_epoch,
                'is_final': True,
                **{k: getattr(avg_stats, k) for k in dir(avg_stats) if not k.startswith('_')}
            })
            self.save_stats()

    def _print_model_insights(self, model_type: str, stats: DiagnosticStats) -> None:
        """Print model-specific diagnostic insights."""
        print("\n[5] MODEL-SPECIFIC INSIGHTS:")

        if model_type == 'note_only':
            print("  Model: PercePianoNoteOnly (2-layer note LSTM)")
            print("  Expected R2: ~0.10")
            print(f"  Note LSTM output std: {stats.note_out_std:.4f}")
            if stats.note_out_std < 0.05:
                print("  [WARNING] Note LSTM output variance is low - may indicate vanishing signal")

        elif model_type == 'note_voice':
            print("  Model: PercePianoNoteVoice (2-layer note + 2-layer voice)")
            print("  Expected R2: ~0.15 (gain ~+0.05 from voice)")
            print(f"  Note LSTM output std: {stats.note_out_std:.4f}")
            print(f"  Voice LSTM output std: {stats.voice_out_std:.4f}")
            print(f"  Combined hidden_out std: {stats.hidden_out_std:.4f}")
            if stats.voice_out_std < 0.05:
                print("  [WARNING] Voice LSTM output variance is low - voice may not be learning")

        elif model_type == 'note_voice_beat':
            print("  Model: PercePianoNoteVoiceBeat (note + voice + beat hierarchy)")
            print("  Expected R2: ~0.25-0.30 (gain ~+0.10 from beat)")
            print(f"  Beat attention entropy: {stats.beat_attention_entropy:.3f}")
            if stats.beat_attention_entropy > 0.9:
                print("  [WARNING] Beat attention is near-uniform - attention may not be learning")
            elif stats.beat_attention_entropy < 0.3:
                print("  [WARNING] Beat attention is too focused - may be overfitting")
            else:
                print("  [OK] Beat attention entropy is in healthy range (0.3-0.8)")
            print(f"  Beat contribution: {stats.beat_spanned_contribution:.1%}")

        elif model_type == 'han':
            print("  Model: PercePianoHAN (full hierarchical)")
            print("  Expected R2: ~0.35-0.40 (target: 0.397)")
            print(f"  Beat attention entropy: {stats.beat_attention_entropy:.3f}")
            print(f"  Measure attention entropy: {stats.measure_attention_entropy:.3f}")
            print(f"  Beat contribution: {stats.beat_spanned_contribution:.1%}")
            print(f"  Measure contribution: {stats.measure_spanned_contribution:.1%}")

        elif model_type == 'baseline':
            print("  Model: PercePianoBiLSTMBaseline (7-layer single LSTM)")
            print("  Expected R2: ~0.19 (matches VirtuosoNetSingle)")

        elif model_type == 'baseline_beat':
            print("  Model: PercePianoBaselinePlusBeat (7-layer + beat)")
            print("  [NOTE] This is the OLD incremental approach - consider using note_voice_beat instead")

        print()

    def save_stats(self, path: Optional[Path] = None) -> None:
        """Save collected statistics to JSON."""
        save_path = path or self.save_dir
        if save_path is None:
            return

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'diagnostic_stats.json', 'w') as f:
            json.dump(self.epoch_stats, f, indent=2)


class HierarchyAblationCallback(Callback):
    """
    Callback to measure model performance with and without hierarchy.

    Supports two ablation modes:

    1. "true_baseline" (RECOMMENDED): Uses a pre-trained PercePianoBiLSTMBaseline model
       that matches the original VirtuosoNetSingle architecture (7-layer single LSTM).
       This gives the TRUE hierarchy gain as reported in the paper (+0.21 R2).

    2. "zeroed_hierarchy" (DEPRECATED): Zeros out beat_spanned and measure_spanned
       but still uses HAN's note+voice LSTMs. This does NOT match the original
       baseline and gives misleading hierarchy gain measurements.

    The original PercePiano paper compares:
    - VirtuosoNetSingle (7-layer LSTM): R2 = 0.187
    - VirtuosoNetMultiLevel (HAN): R2 = 0.397
    - Hierarchy gain: +0.21

    Usage:
        # True baseline comparison (requires pre-trained baseline checkpoint)
        callback = HierarchyAblationCallback(
            baseline_checkpoint="/path/to/bilstm_baseline.ckpt",
            mode="true_baseline"
        )

        # Legacy mode (not recommended, gives misleading results)
        callback = HierarchyAblationCallback(mode="zeroed_hierarchy")
    """

    def __init__(
        self,
        run_every_n_epochs: int = 10,
        baseline_checkpoint: Optional[str] = None,
        mode: str = "zeroed_hierarchy",
    ):
        super().__init__()
        self.run_every_n_epochs = run_every_n_epochs
        self.baseline_checkpoint = baseline_checkpoint
        self.mode = mode
        self.ablation_results: List[Dict[str, float]] = []
        self._baseline_model: Optional[pl.LightningModule] = None

        if mode == "true_baseline" and baseline_checkpoint is None:
            print("[ABLATION] WARNING: true_baseline mode requires baseline_checkpoint. "
                  "Falling back to zeroed_hierarchy mode.")
            self.mode = "zeroed_hierarchy"

        if mode == "zeroed_hierarchy":
            print("[ABLATION] WARNING: Using zeroed_hierarchy mode. This does NOT match "
                  "the original VirtuosoNetSingle baseline and gives misleading results. "
                  "For accurate comparison, train PercePianoBiLSTMBaseline separately "
                  "and use mode='true_baseline' with baseline_checkpoint.")

    def _load_baseline_model(self, device: torch.device) -> Optional[pl.LightningModule]:
        """Load the pre-trained baseline model."""
        if self._baseline_model is not None:
            return self._baseline_model

        if self.baseline_checkpoint is None:
            return None

        try:
            from ..models.percepiano_replica import PercePianoBiLSTMBaseline
            self._baseline_model = PercePianoBiLSTMBaseline.load_from_checkpoint(
                self.baseline_checkpoint
            )
            self._baseline_model.to(device)
            self._baseline_model.eval()
            print(f"[ABLATION] Loaded baseline model from {self.baseline_checkpoint}")
            return self._baseline_model
        except Exception as e:
            print(f"[ABLATION] Failed to load baseline model: {e}")
            return None

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run ablation at end of validation."""
        current_epoch = trainer.current_epoch

        if current_epoch % self.run_every_n_epochs != 0:
            return

        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return

        mode_str = "TRUE BASELINE" if self.mode == "true_baseline" else "ZEROED HIERARCHY (deprecated)"
        print(f"\n[ABLATION] Running {mode_str} test at epoch {current_epoch}...")

        # Collect predictions with full model and ablated model
        full_preds = []
        ablated_preds = []
        targets = []

        pl_module.eval()
        device = pl_module.device

        # Load baseline model if using true_baseline mode
        baseline_model = None
        if self.mode == "true_baseline":
            baseline_model = self._load_baseline_model(device)
            if baseline_model is None:
                print("[ABLATION] Falling back to zeroed_hierarchy mode")
                self.mode = "zeroed_hierarchy"

        with torch.no_grad():
            from torch.nn.utils.rnn import PackedSequence

            for batch in val_loader:
                # Move tensors to device, handling PackedSequence properly
                batch_on_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_on_device[k] = v.to(device)
                    elif isinstance(v, PackedSequence):
                        # Move PackedSequence data to device (batch_sizes MUST stay on CPU)
                        batch_on_device[k] = PackedSequence(
                            v.data.to(device),
                            v.batch_sizes,  # Must stay on CPU per PyTorch requirement
                            v.sorted_indices.to(device) if v.sorted_indices is not None else None,
                            v.unsorted_indices.to(device) if v.unsorted_indices is not None else None,
                        )
                    else:
                        batch_on_device[k] = v

                # Full model prediction
                note_locations = {
                    'beat': batch_on_device['note_locations_beat'],
                    'measure': batch_on_device['note_locations_measure'],
                    'voice': batch_on_device['note_locations_voice'],
                }

                outputs = pl_module(
                    batch_on_device['input_features'],
                    note_locations,
                    batch_on_device.get('attention_mask'),
                    batch_on_device.get('lengths'),
                )
                full_preds.append(outputs['predictions'].cpu())
                targets.append(batch_on_device['scores'].cpu())

                # Ablated prediction
                if self.mode == "true_baseline" and baseline_model is not None:
                    ablated_outputs = baseline_model(
                        batch_on_device['input_features'],
                        note_locations,
                        batch_on_device.get('attention_mask'),
                        batch_on_device.get('lengths'),
                    )
                    ablated_preds.append(ablated_outputs['predictions'].cpu())
                else:
                    # Legacy: zero out hierarchy
                    ablated_outputs = self._forward_zeroed_hierarchy(
                        pl_module, batch_on_device, note_locations
                    )
                    ablated_preds.append(ablated_outputs.cpu())

        # Compute R2 scores
        from sklearn.metrics import r2_score

        full_preds = torch.cat(full_preds).numpy()
        ablated_preds = torch.cat(ablated_preds).numpy()
        targets = torch.cat(targets).numpy()

        full_r2 = r2_score(targets, full_preds)
        ablated_r2 = r2_score(targets, ablated_preds)
        hierarchy_gain = full_r2 - ablated_r2

        result = {
            'epoch': current_epoch,
            'full_r2': full_r2,
            'ablated_r2': ablated_r2,
            'hierarchy_gain': hierarchy_gain,
            'mode': self.mode,
        }
        self.ablation_results.append(result)

        # Log
        pl_module.log('ablation/full_r2', full_r2)
        pl_module.log('ablation/ablated_r2', ablated_r2)
        pl_module.log('ablation/hierarchy_gain', hierarchy_gain)

        baseline_name = "Bi-LSTM 7-layer" if self.mode == "true_baseline" else "zeroed hierarchy"
        print(f"[ABLATION] Full HAN R2: {full_r2:.4f}")
        print(f"[ABLATION] Baseline ({baseline_name}) R2: {ablated_r2:.4f}")
        print(f"[ABLATION] Hierarchy gain: {hierarchy_gain:+.4f}")

        if self.mode == "true_baseline":
            expected_gain = 0.21
            if abs(hierarchy_gain - expected_gain) > 0.10:
                print(f"[ABLATION] NOTE: Expected gain ~+0.21, got {hierarchy_gain:+.4f}")
        elif hierarchy_gain < 0.01:
            print("[ABLATION] WARNING: Hierarchy contributing < 0.01 R2!")
            print("[ABLATION] NOTE: This uses zeroed_hierarchy which may underestimate gain.")

        pl_module.train()

    def _forward_zeroed_hierarchy(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        note_locations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with hierarchy zeroed out (DEPRECATED approach).

        WARNING: This does NOT match the original VirtuosoNetSingle baseline!
        - Uses 4-layer split LSTMs (note + voice) instead of 7-layer single LSTM
        - Still includes voice processing which baseline doesn't have
        - Feeds 2048-dim (half zeros) to contractor instead of 512-dim dense

        For accurate comparison, use mode='true_baseline' with a trained
        PercePianoBiLSTMBaseline checkpoint.
        """
        from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

        device = pl_module.device
        input_features = batch['input_features']
        if isinstance(input_features, PackedSequence):
            input_features, _ = pad_packed_sequence(input_features, batch_first=True)

        # Ensure input is on correct device
        input_features = input_features.to(device)

        han = pl_module.han_encoder

        # Get hidden_out only (Bi-LSTM part)
        from ..models.hierarchy_utils import compute_actual_lengths
        actual_lengths = compute_actual_lengths(note_locations['beat'])

        x_embedded = han.note_fc(input_features)

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        x_packed = pack_padded_sequence(
            x_embedded,
            actual_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        note_out, _ = han.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=x_embedded.shape[1])

        voice_out = han._run_voice_processing(x_embedded, note_locations['voice'], actual_lengths)
        hidden_out = torch.cat([note_out, voice_out], dim=-1)

        # Zero out beat and measure (ablation)
        batch_size, seq_len = hidden_out.shape[:2]
        beat_spanned = torch.zeros(batch_size, seq_len, han.beat_size * 2, device=hidden_out.device)
        measure_spanned = torch.zeros(batch_size, seq_len, han.measure_size * 2, device=hidden_out.device)

        # Concatenate
        total_note_cat = torch.cat([hidden_out, beat_spanned, measure_spanned], dim=-1)

        # Rest of forward
        contracted = pl_module.performance_contractor(total_note_cat)
        attention_mask = note_locations['beat'] > 0
        aggregated = pl_module.final_attention(contracted, mask=attention_mask)
        logits = pl_module.prediction_head(aggregated)
        predictions = torch.sigmoid(logits)

        return predictions


def run_full_diagnostics(
    model: pl.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 5,
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostics on a trained model.

    Args:
        model: Trained PercePiano model
        dataloader: Validation/test dataloader
        device: Device to run on
        num_batches: Number of batches to analyze

    Returns:
        Dictionary with all diagnostic results
    """
    model.eval()
    model.to(device)

    all_stats = []
    all_index_stats = []

    callback = DiagnosticCallback()

    with torch.no_grad():
        from torch.nn.utils.rnn import PackedSequence

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Move tensors to device, handling PackedSequence properly
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                elif isinstance(v, PackedSequence):
                    # Move PackedSequence data to device (batch_sizes MUST stay on CPU)
                    batch_on_device[k] = PackedSequence(
                        v.data.to(device),
                        v.batch_sizes,  # Must stay on CPU per PyTorch requirement
                        v.sorted_indices.to(device) if v.sorted_indices is not None else None,
                        v.unsorted_indices.to(device) if v.unsorted_indices is not None else None,
                    )
                else:
                    batch_on_device[k] = v

            stats = callback._run_diagnostic_forward(model, batch_on_device)
            all_stats.append(stats.to_dict())

            # Index analysis (on CPU)
            idx_stats = analyze_indices(
                batch['note_locations_beat'].cpu() if isinstance(batch['note_locations_beat'], torch.Tensor) else batch['note_locations_beat'],
                batch['note_locations_measure'].cpu() if isinstance(batch['note_locations_measure'], torch.Tensor) else batch['note_locations_measure'],
            )
            all_index_stats.append(idx_stats)

    # Aggregate stats
    aggregated = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    # Aggregate index stats
    index_summary = {}
    for key in all_index_stats[0].keys():
        values = [s[key] for s in all_index_stats]
        index_summary[key] = {
            'mean': np.mean(values),
            'total': np.sum(values) if 'count' in key else np.mean(values),
        }

    return {
        'activation_stats': aggregated,
        'index_stats': index_summary,
        'raw_stats': all_stats,
    }
