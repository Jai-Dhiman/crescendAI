"""
PercePiano Replica Model - SOTA Reproduction.

This module implements the exact architecture from:
"PercePiano: A Benchmark for Perceptual Evaluation of Piano Performance"
Park et al., ISMIR 2024 / Nature Scientific Reports 2024
GitHub: https://github.com/JonghoKimSNU/PercePiano

Architecture: Bi-LSTM + Score Alignment + Hierarchical Attention Network (HAN)
Published R-squared: 0.397 (piece-split), 0.285 (performer-split)

Key differences from our score_aligned_module.py:
1. Bi-LSTM instead of Transformer for sequence encoding (~10x fewer params)
2. Smaller hidden dimensions (256 vs 768)
3. Exact PercePiano hyperparameters (lr=2.5e-5, dropout=0.2, etc.)

Attribution:
    If you use this code, please cite the PercePiano paper:
    @article{park2024percepiano,
        title={PercePiano: A Benchmark for Perceptual Evaluation of Piano Performance},
        author={Park, Jongho and Kim, Dasaem and others},
        journal={Nature Scientific Reports},
        year={2024}
    }
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .context_attention import ContextAttention
from .hierarchy_utils import (
    make_higher_node,
    span_beat_to_note_num,
    run_hierarchy_lstm_with_pack,
    compute_actual_lengths,
)


# All 19 PercePiano dimensions
PERCEPIANO_DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamic_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "sophistication",
    "interpretation",
]


class PercePianoHAN(nn.Module):
    """
    Hierarchical Attention Network encoder matching PercePiano's architecture.

    Architecture (from han_bigger256_concat.yml):
        Note LSTM (256, 2 layers, bidirectional)
            -> Voice LSTM (256, 2 layers, bidirectional)
            -> Beat Attention + LSTM (256, 2 layers, bidirectional)
            -> Measure Attention + LSTM (256, 1 layer, bidirectional)
            -> Concatenate all levels (total_note_cat)

    Parameters matched exactly to PercePiano:
        - note_size: 256
        - voice_size: 256
        - beat_size: 256
        - measure_size: 256
        - num_attention_heads: 8
        - dropout: 0.2
    """

    def __init__(
        self,
        input_size: int = 84,  # VirtuosoNet feature dimension (79 base + 5 unnorm)
        note_size: int = 256,
        voice_size: int = 256,
        beat_size: int = 256,
        measure_size: int = 256,
        note_layers: int = 2,
        voice_layers: int = 2,
        beat_layers: int = 2,
        measure_layers: int = 1,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.note_size = note_size
        self.voice_size = voice_size
        self.beat_size = beat_size
        self.measure_size = measure_size

        # Input projection
        self.note_fc = nn.Linear(input_size, note_size)

        # Note-level Bi-LSTM
        self.note_lstm = nn.LSTM(
            note_size, note_size, note_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if note_layers > 1 else 0,
        )

        # Voice-level Bi-LSTM (processes notes grouped by voice)
        # CRITICAL: Input is note_size (256), NOT note_size * 2 (512)
        # Original PercePiano processes the SAME 256-dim embeddings through both
        # note_lstm and voice_net IN PARALLEL (encoder_score.py:496-516)
        self.voice_lstm = nn.LSTM(
            note_size, voice_size, voice_layers,  # note_size=256, NOT note_size*2
            batch_first=True, bidirectional=True,
            dropout=dropout if voice_layers > 1 else 0,
        )

        # Combined dimension after note + voice (both bidirectional)
        combined_dim = (note_size + voice_size) * 2

        # Beat-level: Attention aggregation + Bi-LSTM
        self.beat_attention = ContextAttention(combined_dim, num_attention_heads)
        self.beat_lstm = nn.LSTM(
            combined_dim, beat_size, beat_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if beat_layers > 1 else 0,
        )

        # Measure-level: Attention aggregation + Bi-LSTM
        self.measure_attention = ContextAttention(beat_size * 2, num_attention_heads)
        self.measure_lstm = nn.LSTM(
            beat_size * 2, measure_size, measure_layers,
            batch_first=True, bidirectional=True,
        )

        # Output dimension: note+voice + beat + measure (all bidirectional)
        self.output_dim = combined_dim + beat_size * 2 + measure_size * 2

    def forward(
        self,
        x: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HAN encoder.

        Args:
            x: Input features [batch, seq_len, input_size]
            note_locations: Dict with 'beat', 'measure', 'voice' tensors

        Returns:
            Dict with hierarchical representations
        """
        batch_size, seq_len, _ = x.shape
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        voice_numbers = note_locations['voice']

        # Compute actual sequence lengths
        actual_lengths = compute_actual_lengths(beat_numbers)

        # Project input to 256-dim embeddings
        x_embedded = self.note_fc(x)  # [B, T, 256]

        # Note-level LSTM (processes 256-dim embeddings)
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        note_out, _ = self.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=seq_len)
        # note_out: [B, T, 512] (bidirectional)

        # Voice-level LSTM (processes the SAME 256-dim embeddings, NOT note_out)
        # CRITICAL FIX: Original PercePiano runs voice_net on x_embedded IN PARALLEL
        # with note_lstm, not on note_lstm output (encoder_score.py:515-516)
        voice_out = self._run_voice_processing(x_embedded, voice_numbers, actual_lengths)
        # voice_out: [B, T, 512] (bidirectional)

        # Concatenate note and voice outputs
        hidden_out = torch.cat([note_out, voice_out], dim=-1)

        # Beat aggregation + LSTM
        beat_nodes = make_higher_node(
            hidden_out, self.beat_attention, beat_numbers, beat_numbers,
            lower_is_note=True, actual_lengths=actual_lengths
        )
        beat_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_lstm)
        beat_spanned = span_beat_to_note_num(beat_out, beat_numbers, actual_lengths)

        # Ensure beat_spanned matches sequence length
        if beat_spanned.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - beat_spanned.shape[1], beat_spanned.shape[2],
                device=beat_spanned.device, dtype=beat_spanned.dtype
            )
            beat_spanned = torch.cat([beat_spanned, padding], dim=1)

        # Measure aggregation + LSTM
        measure_nodes = make_higher_node(
            beat_out, self.measure_attention, beat_numbers, measure_numbers,
            actual_lengths=actual_lengths
        )
        measure_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_lstm)
        measure_spanned = span_beat_to_note_num(measure_out, measure_numbers, actual_lengths)

        # Ensure measure_spanned matches sequence length
        if measure_spanned.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - measure_spanned.shape[1], measure_spanned.shape[2],
                device=measure_spanned.device, dtype=measure_spanned.dtype
            )
            measure_spanned = torch.cat([measure_spanned, padding], dim=1)

        # Concatenate all levels (PercePiano's total_note_cat)
        total_note_cat = torch.cat([hidden_out, beat_spanned, measure_spanned], dim=-1)

        return {
            'note': note_out,
            'voice': hidden_out,
            'beat': beat_out,
            'measure': measure_out,
            'beat_spanned': beat_spanned,
            'measure_spanned': measure_spanned,
            'total_note_cat': total_note_cat,
        }

    def _run_voice_processing(
        self,
        x_embedded: torch.Tensor,
        voice_numbers: torch.Tensor,
        actual_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process notes through voice LSTM, grouped by voice.

        Matches original PercePiano encoder_score.py:526-549 exactly:
        - Process each voice separately through LSTM
        - Scatter results back using batched torch.bmm (not Python loops)
        - This preserves voice-specific temporal patterns

        CRITICAL: This receives the 256-dim projected embeddings (x_embedded),
        NOT the 512-dim note LSTM output. The original PercePiano processes
        x through voice_net IN PARALLEL with note_lstm.

        Args:
            x_embedded: Projected embeddings [batch, seq_len, note_size] (256-dim)
            voice_numbers: Voice assignment per note [batch, seq_len]
            actual_lengths: Actual sequence lengths [batch]

        Returns:
            Voice-processed features [batch, seq_len, voice_hidden*2] (512-dim)
        """
        batch_size, num_notes, hidden_dim = x_embedded.shape
        output = torch.zeros(
            batch_size, num_notes, self.voice_size * 2,
            device=x_embedded.device, dtype=x_embedded.dtype
        )

        # Get max voice across batch (voice numbers start at 1)
        max_voice = voice_numbers.max().item()
        if max_voice == 0:
            return output

        # Process each voice separately (matching original exactly)
        for voice_idx in range(1, int(max_voice) + 1):
            # Create mask for this voice [batch, seq_len]
            voice_x_bool = (voice_numbers == voice_idx)
            num_voice_notes = torch.sum(voice_x_bool)
            num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)  # [batch]

            if num_voice_notes > 0:
                # Extract notes belonging to this voice from each batch item
                # (matching original: list comprehension with placeholder for empty batches)
                voice_notes = [
                    x_embedded[i, voice_x_bool[i]] if torch.sum(voice_x_bool[i]) > 0
                    else torch.zeros(1, hidden_dim, device=x_embedded.device, dtype=x_embedded.dtype)
                    for i in range(batch_size)
                ]

                # Pad to same length
                voice_x = pad_sequence(voice_notes, batch_first=True)

                # Pack and process through voice LSTM
                pack_voice_x = pack_padded_sequence(
                    voice_x,
                    [len(v) for v in voice_notes],
                    batch_first=True,
                    enforce_sorted=False
                )
                ith_voice_out, _ = self.voice_lstm(pack_voice_x)
                ith_voice_out, _ = pad_packed_sequence(ith_voice_out, batch_first=True)

                # Scatter results back using batched torch.bmm (matching original exactly)
                span_mat = torch.zeros(batch_size, num_notes, voice_x.shape[1], device=x_embedded.device)
                voice_where = torch.nonzero(voice_x_bool)
                span_mat[
                    voice_where[:, 0],
                    voice_where[:, 1],
                    torch.cat([torch.arange(num_batch_voice_notes[i].item(), device=x_embedded.device)
                               for i in range(batch_size)])
                ] = 1

                output += torch.bmm(span_mat, ith_voice_out)

        return output


class PercePianoReplicaModule(pl.LightningModule):
    """
    PyTorch Lightning module replicating PercePiano SOTA.

    This is a faithful reproduction of the PercePiano paper's best model
    (Bi-LSTM + SA + HAN) with exact hyperparameters from their config.

    Target performance: R-squared = 0.35-0.40 (piece-split)

    Key hyperparameters (from han_bigger256_concat.yml):
        - learning_rate: 2.5e-5
        - hidden_size: 256 (all levels)
        - dropout: 0.2
        - attention_heads: 8
        - loss: MSE
    """

    def __init__(
        self,
        # Input dimensions
        score_note_features: int = 20,
        score_global_features: int = 12,
        # HAN dimensions (matched to PercePiano)
        hidden_size: int = 256,
        note_layers: int = 2,
        voice_layers: int = 2,
        beat_layers: int = 2,
        measure_layers: int = 1,
        num_attention_heads: int = 8,
        # Final head
        final_hidden: int = 128,
        # Training (PercePiano defaults)
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        dropout: float = 0.2,
        # Task
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)

        # Global feature encoder (simple MLP)
        self.global_encoder = nn.Sequential(
            nn.Linear(score_global_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
        )

        # Tempo curve encoder (1D CNN)
        self.tempo_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # HAN encoder for note-level features
        han_input_size = score_note_features + hidden_size  # note features + global context
        self.han_encoder = PercePianoHAN(
            input_size=han_input_size,
            note_size=hidden_size,
            voice_size=hidden_size,
            beat_size=hidden_size,
            measure_size=hidden_size,
            note_layers=note_layers,
            voice_layers=voice_layers,
            beat_layers=beat_layers,
            measure_layers=measure_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        # Performance contractor: Contract total_note_cat from 2048 -> 512
        # (Critical layer from original PercePiano VirtuosoNetMultiLevel)
        encoder_output_size = hidden_size * 2  # 512 for hidden_size=256
        self.performance_contractor = nn.Linear(
            self.han_encoder.output_dim,  # 2048
            encoder_output_size,  # 512
        )

        # Final attention aggregation (over contracted 512-dim, not raw 2048-dim)
        self.final_attention = ContextAttention(
            encoder_output_size, num_attention_heads
        )

        # Prediction head (matching original PercePiano out_fc structure)
        # Original: Dropout -> Linear -> GELU -> Dropout -> Linear
        head_input_dim = encoder_output_size + hidden_size  # 512 + 256 for global context
        self.prediction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, encoder_output_size),  # -> 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, self.num_dimensions),  # 512 -> 19
        )

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        score_note_features: torch.Tensor,
        score_global_features: torch.Tensor,
        score_tempo_curve: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            score_note_features: [batch, num_notes, note_features]
            score_global_features: [batch, global_features]
            score_tempo_curve: [batch, num_segments]
            note_locations: Dict with 'beat', 'measure', 'voice' tensors
            attention_mask: [batch, num_notes] optional mask

        Returns:
            Dict with 'predictions' [batch, num_dimensions]
        """
        batch_size, num_notes, _ = score_note_features.shape

        # Encode global features
        global_encoded = self.global_encoder(score_global_features)  # [B, H/2]

        # Encode tempo curve
        tempo_input = score_tempo_curve.unsqueeze(1)  # [B, 1, T]
        tempo_encoded = self.tempo_encoder(tempo_input)  # [B, H/2]

        # Combine global context
        global_context = torch.cat([global_encoded, tempo_encoded], dim=-1)  # [B, H]

        # Broadcast global context to all notes
        global_broadcast = global_context.unsqueeze(1).expand(-1, num_notes, -1)  # [B, N, H]

        # Concatenate note features with global context
        han_input = torch.cat([score_note_features, global_broadcast], dim=-1)  # [B, N, F+H]

        # Run HAN encoder
        han_outputs = self.han_encoder(han_input, note_locations)
        total_note_cat = han_outputs['total_note_cat']  # [B, N, 2048]

        # Contract from 2048 -> 512 (critical step from original PercePiano)
        contracted = self.performance_contractor(total_note_cat)  # [B, N, 512]

        # Aggregate to single vector using attention (over 512-dim, not 2048)
        aggregated = self.final_attention(contracted)  # [B, 512]

        # Combine with global context
        combined = torch.cat([aggregated, global_context], dim=-1)  # [B, 512 + H]

        # Predict scores - apply sigmoid to match [0, 1] label range
        # (Original PercePiano always applies sigmoid before loss computation)
        predictions = torch.sigmoid(self.prediction_head(combined))  # [B, num_dims]

        return {
            'predictions': predictions,
            'han_outputs': han_outputs,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MSE loss (matching PercePiano).
        """
        total_loss = nn.functional.mse_loss(predictions, targets, reduction='mean')

        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction='mean'
            )

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract note_locations from batch."""
        return {
            'beat': batch['note_locations_beat'],
            'measure': batch['note_locations_measure'],
            'voice': batch['note_locations_voice'],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch['score_note_features'],
            batch['score_global_features'],
            batch['score_tempo_curve'],
            note_locations,
            batch.get('score_attention_mask'),
        )

        loss, per_dim_losses = self.compute_loss(outputs['predictions'], batch['scores'])

        self.log('train/loss', loss, prog_bar=True)
        self.training_step_outputs.append({'loss': loss.detach()})

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch['score_note_features'],
            batch['score_global_features'],
            batch['score_tempo_curve'],
            note_locations,
            batch.get('score_attention_mask'),
        )

        loss, _ = self.compute_loss(outputs['predictions'], batch['scores'])

        result = {
            'val_loss': loss.detach(),
            'predictions': outputs['predictions'].detach(),
            'targets': batch['scores'].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log('val/loss', loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

        # Convert to numpy for sklearn
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        # Use sklearn's r2_score for consistency with original PercePiano (train_m2pf.py:222)
        # This uses uniform_average by default for multioutput
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log('val/mean_r2', mean_r2, prog_bar=True)

        # Also compute and log per-dimension R2
        r2_values = []
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            r2_values.append(dim_r2)
            self.log(f'val/r2_{dim}', dim_r2)

        # Log key dimensions
        if 'tempo' in self.dimensions:
            self.log('val/tempo_r2', r2_values[self.dimensions.index('tempo')], prog_bar=True)
        if 'timing' in self.dimensions:
            self.log('val/timing_r2', r2_values[self.dimensions.index('timing')], prog_bar=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer with PercePiano settings."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Optional: learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/mean_r2',
            }
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PercePianoVNetModule(pl.LightningModule):
    """
    Faithful PercePiano replica using VirtuosoNet 84-dim features.

    This version matches the original PercePiano architecture exactly:
    - Input: 84-dim VirtuosoNet features (79 base + 5 unnorm for key augmentation)
    - HAN encoder: Note -> Voice -> Beat -> Measure hierarchy
    - Output: 19-dim scores

    Feature layout:
    - Indices 0-78: Base VirtuosoNet features (z-score normalized where applicable)
    - Index 79: midi_pitch_unnorm (raw MIDI pitch 21-108, for key augmentation)
    - Index 80-83: duration_unnorm, beat_importance_unnorm, measure_length_unnorm, following_rest_unnorm

    Key difference from PercePianoReplicaModule:
    - No global_encoder or tempo_encoder (these aren't in original PercePiano)
    - HAN input is exactly the VirtuosoNet features (no global context concatenation)
    - Simpler, more faithful to published architecture

    Training settings matched to original PercePiano:
    - learning_rate: 1e-4 (parser.py:119)
    - weight_decay: 1e-5 (parser.py:135)
    - batch_size: 32 (parser.py:107)
    - gradient_clip_val: 2.0 (parser.py:159) - SET IN TRAINER
    - scheduler: StepLR(step_size=3000, gamma=0.98)
    """

    # Recommended gradient clipping value (set in Lightning Trainer)
    GRADIENT_CLIP_VAL = 2.0

    def __init__(
        self,
        # Input dimension (VirtuosoNet features: 79 base + 5 unnorm)
        input_size: int = 84,
        # HAN dimensions (matched to PercePiano han_bigger256_concat.yml)
        hidden_size: int = 256,
        note_layers: int = 2,
        voice_layers: int = 2,
        beat_layers: int = 2,
        measure_layers: int = 1,
        num_attention_heads: int = 8,
        # Final head
        final_hidden: int = 128,
        # Training (matched to original PercePiano han_bigger256_concat.yml)
        learning_rate: float = 1e-4,   # Original: 1e-4 (was 2.5e-5 - 4x too low!)
        weight_decay: float = 1e-5,    # Original: 1e-5 (was 0.01 - 100x too high!)
        dropout: float = 0.2,
        # Task
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)

        # HAN encoder - takes 84-dim input directly (no global context concatenation)
        self.han_encoder = PercePianoHAN(
            input_size=input_size,  # 84-dim VirtuosoNet features (79 base + 5 unnorm)
            note_size=hidden_size,
            voice_size=hidden_size,
            beat_size=hidden_size,
            measure_size=hidden_size,
            note_layers=note_layers,
            voice_layers=voice_layers,
            beat_layers=beat_layers,
            measure_layers=measure_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        # Performance contractor: Contract total_note_cat from 2048 -> 512
        # (Critical layer from original PercePiano VirtuosoNetMultiLevel)
        encoder_output_size = hidden_size * 2  # 512 for hidden_size=256
        self.performance_contractor = nn.Linear(
            self.han_encoder.output_dim,  # 2048
            encoder_output_size,  # 512
        )

        # Final attention aggregation (over contracted 512-dim, not raw 2048-dim)
        self.final_attention = ContextAttention(
            encoder_output_size, num_attention_heads
        )

        # Prediction head (matching original PercePiano out_fc structure)
        # Original: Dropout -> Linear -> GELU -> Dropout -> Linear
        self.prediction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, encoder_output_size),  # 512 -> 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, self.num_dimensions),  # 512 -> 19
        )

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with VirtuosoNet features.

        Args:
            input_features: [batch, num_notes, 84] VirtuosoNet features (79 base + 5 unnorm)
            note_locations: Dict with 'beat', 'measure', 'voice' tensors
            attention_mask: [batch, num_notes] optional mask

        Returns:
            Dict with 'predictions' [batch, num_dimensions]
        """
        # Run HAN encoder directly on 84-dim features (no global context)
        han_outputs = self.han_encoder(input_features, note_locations)
        total_note_cat = han_outputs['total_note_cat']  # [B, N, 2048]

        # Contract from 2048 -> 512 (critical step from original PercePiano)
        contracted = self.performance_contractor(total_note_cat)  # [B, N, 512]

        # Aggregate to single vector using attention (over 512-dim, not 2048)
        aggregated = self.final_attention(contracted)  # [B, 512]

        # Predict scores - apply sigmoid to match [0, 1] label range
        # (Original PercePiano always applies sigmoid before loss computation)
        predictions = torch.sigmoid(self.prediction_head(aggregated))  # [B, num_dims]

        return {
            'predictions': predictions,
            'han_outputs': han_outputs,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MSE loss (matching PercePiano)."""
        total_loss = nn.functional.mse_loss(predictions, targets, reduction='mean')

        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction='mean'
            )

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract note_locations from batch."""
        return {
            'beat': batch['note_locations_beat'],
            'measure': batch['note_locations_measure'],
            'voice': batch['note_locations_voice'],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch['input_features'],
            note_locations,
            batch.get('attention_mask'),
        )

        loss, per_dim_losses = self.compute_loss(outputs['predictions'], batch['scores'])

        self.log('train/loss', loss, prog_bar=True)
        self.training_step_outputs.append({'loss': loss.detach()})

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch['input_features'],
            note_locations,
            batch.get('attention_mask'),
        )

        loss, _ = self.compute_loss(outputs['predictions'], batch['scores'])

        result = {
            'val_loss': loss.detach(),
            'predictions': outputs['predictions'].detach(),
            'targets': batch['scores'].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log('val/loss', loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])

        # Convert to numpy for sklearn
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        # Use sklearn's r2_score for consistency with original PercePiano (train_m2pf.py:222)
        # This uses uniform_average by default for multioutput
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log('val/mean_r2', mean_r2, prog_bar=True)

        # Also compute and log per-dimension R2
        r2_values = []
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            r2_values.append(dim_r2)
            self.log(f'val/r2_{dim}', dim_r2)

        # Log key dimensions
        if 'tempo' in self.dimensions:
            self.log('val/tempo_r2', r2_values[self.dimensions.index('tempo')], prog_bar=True)
        if 'timing' in self.dimensions:
            self.log('val/timing_r2', r2_values[self.dimensions.index('timing')], prog_bar=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure optimizer with PercePiano settings (matched to original).

        NOTE: Gradient clipping (grad_clip=2) should be set in the Lightning Trainer:
            trainer = pl.Trainer(gradient_clip_val=2.0, ...)
        Original PercePiano uses grad_clip=2 (parser.py:159, train_m2pf.py:173).
        """
        # Original uses Adam (not AdamW) with very light weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Original uses StepLR with step_size=3000, gamma=0.98
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.98
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # StepLR operates per step, not per epoch
            }
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("PercePiano Replica Model")
    print("=" * 60)
    print("Reference: Park et al., 'PercePiano', ISMIR/Nature 2024")
    print("GitHub: https://github.com/JonghoKimSNU/PercePiano")
    print("=" * 60)

    # Test model creation
    model = PercePianoReplicaModule()
    num_params = model.count_parameters()

    print(f"Total parameters: {num_params:,}")
    print(f"Target R-squared: 0.35-0.40 (piece-split)")
    print(f"Dimensions: {len(model.dimensions)}")
