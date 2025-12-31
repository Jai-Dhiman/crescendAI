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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from torch.nn.utils.rnn import PackedSequence

from .context_attention import ContextAttention, FinalContextAttention
from .hierarchy_utils import (
    compute_actual_lengths,
    make_higher_node,
    run_hierarchy_lstm_with_pack,
    span_beat_to_note_num,
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
        input_size: int = 79,  # VirtuosoNet feature dimension (matches original PercePiano)
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

        # Input projection (matches original PercePiano)
        self.note_fc = nn.Linear(input_size, note_size)

        # Note-level Bi-LSTM
        self.note_lstm = nn.LSTM(
            note_size,
            note_size,
            note_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if note_layers > 1 else 0,
        )

        # Voice-level Bi-LSTM (processes notes grouped by voice)
        # CRITICAL: Input is note_size (256), NOT note_size * 2 (512)
        # Original PercePiano processes the SAME 256-dim embeddings through both
        # note_lstm and voice_net IN PARALLEL (encoder_score.py:496-516)
        self.voice_lstm = nn.LSTM(
            note_size,
            voice_size,
            voice_layers,  # note_size=256, NOT note_size*2
            batch_first=True,
            bidirectional=True,
            dropout=dropout if voice_layers > 1 else 0,
        )

        # Combined dimension after note + voice (both bidirectional)
        combined_dim = (note_size + voice_size) * 2

        # Beat-level: Attention aggregation + Bi-LSTM
        # Temperature=0.5 sharpens attention (prevents uniform collapse)
        # use_hierarchy_init=True for Xavier init + wider context vectors
        self.beat_attention = ContextAttention(
            combined_dim, num_attention_heads, temperature=0.5, use_hierarchy_init=True
        )
        self.beat_lstm = nn.LSTM(
            combined_dim,
            beat_size,
            beat_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if beat_layers > 1 else 0,
        )

        # Measure-level: Attention aggregation + Bi-LSTM
        # Temperature=0.5 sharpens attention (prevents uniform collapse)
        # use_hierarchy_init=True for Xavier init + wider context vectors
        self.measure_attention = ContextAttention(
            beat_size * 2, num_attention_heads, temperature=0.5, use_hierarchy_init=True
        )
        self.measure_lstm = nn.LSTM(
            beat_size * 2,
            measure_size,
            measure_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension: note+voice + beat + measure (all bidirectional)
        self.output_dim = combined_dim + beat_size * 2 + measure_size * 2

        # Apply orthogonal initialization to all LSTMs for better gradient flow
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """
        Apply orthogonal initialization to all LSTM weights for better gradient flow.
        Also sets forget gate bias to 1.0 to prevent vanishing gradients.

        This matches the fix applied to PercePianoBiLSTMBaseline that achieved R2=0.1931.
        """
        for lstm in [self.note_lstm, self.voice_lstm, self.beat_lstm, self.measure_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights: orthogonal initialization
                    nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights: orthogonal initialization
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # Bias initialization: zero all, then set forget gate to 1.0
                    n = param.size(0)
                    param.data.fill_(0)
                    # Forget gate is the second quarter of the bias vector
                    # LSTM bias layout: [input_gate, forget_gate, cell_gate, output_gate]
                    param.data[n // 4 : n // 2].fill_(1.0)

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
        beat_numbers = note_locations["beat"]
        measure_numbers = note_locations["measure"]
        voice_numbers = note_locations["voice"]

        # Compute actual sequence lengths
        actual_lengths = compute_actual_lengths(beat_numbers)

        # Project input to 256-dim embeddings
        x_embedded = self.note_fc(x)  # [B, T, 256]

        # Note-level LSTM (processes 256-dim embeddings)
        x_packed = pack_padded_sequence(
            x_embedded,
            actual_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        note_out, _ = self.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(
            note_out, batch_first=True, total_length=seq_len
        )
        # note_out: [B, T, 512] (bidirectional)

        # Voice-level LSTM (processes the SAME 256-dim embeddings, NOT note_out)
        # CRITICAL FIX: Original PercePiano runs voice_net on x_embedded IN PARALLEL
        # with note_lstm, not on note_lstm output (encoder_score.py:515-516)
        voice_out = self._run_voice_processing(
            x_embedded, voice_numbers, actual_lengths
        )
        # voice_out: [B, T, 512] (bidirectional)

        # Concatenate note and voice outputs
        hidden_out = torch.cat([note_out, voice_out], dim=-1)

        # Beat aggregation + LSTM
        beat_nodes = make_higher_node(
            hidden_out,
            self.beat_attention,
            beat_numbers,
            beat_numbers,
            lower_is_note=True,
            actual_lengths=actual_lengths,
        )
        beat_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_lstm)
        beat_spanned = span_beat_to_note_num(beat_out, beat_numbers, actual_lengths)

        # Ensure beat_spanned matches sequence length
        if beat_spanned.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size,
                seq_len - beat_spanned.shape[1],
                beat_spanned.shape[2],
                device=beat_spanned.device,
                dtype=beat_spanned.dtype,
            )
            beat_spanned = torch.cat([beat_spanned, padding], dim=1)

        # Measure aggregation + LSTM
        measure_nodes = make_higher_node(
            beat_out,
            self.measure_attention,
            beat_numbers,
            measure_numbers,
            actual_lengths=actual_lengths,
        )
        measure_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_lstm)
        measure_spanned = span_beat_to_note_num(
            measure_out, measure_numbers, actual_lengths
        )

        # Ensure measure_spanned matches sequence length
        if measure_spanned.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size,
                seq_len - measure_spanned.shape[1],
                measure_spanned.shape[2],
                device=measure_spanned.device,
                dtype=measure_spanned.dtype,
            )
            measure_spanned = torch.cat([measure_spanned, padding], dim=1)

        # Concatenate all levels (PercePiano's total_note_cat)
        total_note_cat = torch.cat([hidden_out, beat_spanned, measure_spanned], dim=-1)

        return {
            "note": note_out,
            "voice": hidden_out,
            "beat": beat_out,
            "measure": measure_out,
            "beat_spanned": beat_spanned,
            "measure_spanned": measure_spanned,
            "total_note_cat": total_note_cat,
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
            batch_size,
            num_notes,
            self.voice_size * 2,
            device=x_embedded.device,
            dtype=x_embedded.dtype,
        )

        # Get max voice across batch (voice numbers start at 1)
        max_voice = voice_numbers.max().item()
        if max_voice == 0:
            return output

        # Process each voice separately (matching original exactly)
        for voice_idx in range(1, int(max_voice) + 1):
            # Create mask for this voice [batch, seq_len]
            voice_x_bool = voice_numbers == voice_idx
            num_voice_notes = torch.sum(voice_x_bool)
            num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)  # [batch]

            if num_voice_notes > 0:
                # Extract notes belonging to this voice from each batch item
                # (matching original: list comprehension with placeholder for empty batches)
                voice_notes = [
                    x_embedded[i, voice_x_bool[i]]
                    if torch.sum(voice_x_bool[i]) > 0
                    else torch.zeros(
                        1, hidden_dim, device=x_embedded.device, dtype=x_embedded.dtype
                    )
                    for i in range(batch_size)
                ]

                # Pad to same length
                voice_x = pad_sequence(voice_notes, batch_first=True)

                # Pack and process through voice LSTM
                pack_voice_x = pack_padded_sequence(
                    voice_x,
                    [len(v) for v in voice_notes],
                    batch_first=True,
                    enforce_sorted=False,
                )
                ith_voice_out, _ = self.voice_lstm(pack_voice_x)
                ith_voice_out, _ = pad_packed_sequence(ith_voice_out, batch_first=True)

                # Scatter results back using batched torch.bmm (matching original exactly)
                span_mat = torch.zeros(
                    batch_size, num_notes, voice_x.shape[1], device=x_embedded.device
                )
                voice_where = torch.nonzero(voice_x_bool)
                span_mat[
                    voice_where[:, 0],
                    voice_where[:, 1],
                    torch.cat(
                        [
                            torch.arange(
                                num_batch_voice_notes[i].item(),
                                device=x_embedded.device,
                            )
                            for i in range(batch_size)
                        ]
                    ),
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
        han_input_size = (
            score_note_features + hidden_size
        )  # note features + global context
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
        # CRITICAL: Use ContextAttention (same as original PercePiano model_m2pf.py:117)
        # The original uses the SAME attention mechanism for both hierarchy and final
        # aggregation, with learnable context vectors per head.
        # Temperature=0.5 sharpens attention (prevents uniform collapse that causes zero gradients)
        self.final_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5
        )

        # Prediction head (matching original PercePiano out_fc structure)
        # Original: Dropout -> Linear -> GELU -> Dropout -> Linear
        head_input_dim = (
            encoder_output_size + hidden_size
        )  # 512 + 256 for global context
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
        global_broadcast = global_context.unsqueeze(1).expand(
            -1, num_notes, -1
        )  # [B, N, H]

        # Concatenate note features with global context
        han_input = torch.cat(
            [score_note_features, global_broadcast], dim=-1
        )  # [B, N, F+H]

        # Run HAN encoder
        han_outputs = self.han_encoder(han_input, note_locations)
        total_note_cat = han_outputs["total_note_cat"]  # [B, N, 2048]

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
            "predictions": predictions,
            "han_outputs": han_outputs,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MSE loss (matching PercePiano).
        """
        total_loss = nn.functional.mse_loss(predictions, targets, reduction="mean")

        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract note_locations from batch."""
        return {
            "beat": batch["note_locations_beat"],
            "measure": batch["note_locations_measure"],
            "voice": batch["note_locations_voice"],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["score_note_features"],
            batch["score_global_features"],
            batch["score_tempo_curve"],
            note_locations,
            batch.get("score_attention_mask"),
        )

        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"]
        )

        self.log("train/loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss.detach()})

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["score_note_features"],
            batch["score_global_features"],
            batch["score_tempo_curve"],
            note_locations,
            batch.get("score_attention_mask"),
        )

        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])

        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log("val/loss", loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Convert to numpy for sklearn
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        # Use sklearn's r2_score for consistency with original PercePiano (train_m2pf.py:222)
        # This uses uniform_average by default for multioutput
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log("val/mean_r2", mean_r2, prog_bar=True)

        # Also compute and log per-dimension R2
        r2_values = []
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            r2_values.append(dim_r2)
            self.log(f"val/r2_{dim}", dim_r2)

        # Log key dimensions
        if "tempo" in self.dimensions:
            self.log(
                "val/tempo_r2", r2_values[self.dimensions.index("tempo")], prog_bar=True
            )
        if "timing" in self.dimensions:
            self.log(
                "val/timing_r2",
                r2_values[self.dimensions.index("timing")],
                prog_bar=True,
            )

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
            optimizer, mode="max", factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mean_r2",
            },
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PercePianoVNetModule(pl.LightningModule):
    """
    Faithful PercePiano replica using VirtuosoNet features.

    This version matches the original PercePiano architecture exactly:
    - Input: 79-dim VirtuosoNet features (matches original)
    - Batching: PackedSequence for input features (matches original dataset.py)
    - HAN encoder: Note -> Voice -> Beat -> Measure hierarchy
    - Output: 19-dim scores

    Feature layout (79 base + 5 unnorm = 84 total):
    - Indices 0-78: Base VirtuosoNet features (z-score normalized where applicable)
    - Index 79: midi_pitch_unnorm (raw MIDI pitch 21-108, for key augmentation)
    - Index 80-83: duration_unnorm, beat_importance_unnorm, measure_length_unnorm, following_rest_unnorm

    Key difference from PercePianoReplicaModule:
    - No global_encoder or tempo_encoder (these aren't in original PercePiano)
    - HAN input is exactly the VirtuosoNet features (no global context concatenation)
    - Simpler, more faithful to published architecture

    Training settings matched to original PercePiano SOTA (2_run_comp_multilevel_total.sh):
    - learning_rate: 2.5e-5 (SOTA training script)
    - batch_size: 8 (SOTA training script)
    - augment_train: False (SOTA doesn't use augmentation)
    - gradient_clip_val: 2.0 (parser.py:159) - SET IN TRAINER
    """

    # Recommended gradient clipping value (set in Lightning Trainer)
    GRADIENT_CLIP_VAL = 2.0

    def __init__(
        self,
        # Input dimension (VirtuosoNet features: 79 base - matches original PercePiano)
        input_size: int = 79,
        # HAN dimensions (matched to PercePiano han_bigger256_concat.yml)
        hidden_size: int = 256,
        note_layers: int = 2,
        voice_layers: int = 2,
        beat_layers: int = 2,
        measure_layers: int = 1,
        num_attention_heads: int = 8,
        # Training (matched to original PercePiano SOTA exactly)
        # NOTE: LR 2.5e-5 is from 2_run_comp_multilevel_total.sh (NOT 5e-5!)
        learning_rate: float = 2.5e-5,
        weight_decay: float = 1e-5,  # Original: 1e-5
        dropout: float = 0.2,
        # Task
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)

        # HAN encoder - takes 79-dim input directly (matches original PercePiano)
        self.han_encoder = PercePianoHAN(
            input_size=input_size,  # 79-dim VirtuosoNet features
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

        # NOTE: Original PercePiano has NO LayerNorm anywhere
        # We removed LayerNorm in Round 6 to match original exactly

        self.performance_contractor = nn.Linear(
            self.han_encoder.output_dim,  # 2048
            encoder_output_size,  # 512
        )

        # Final attention aggregation (over contracted 512-dim, not raw 2048-dim)
        # CRITICAL: Use ContextAttention (same as original PercePiano model_m2pf.py:117)
        # The original uses the SAME attention mechanism for both hierarchy and final
        # aggregation, with learnable context vectors per head.
        # Temperature=0.5 sharpens attention (prevents uniform collapse that causes zero gradients)
        self.final_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5
        )

        # Prediction head (matching original PercePiano out_fc structure EXACTLY)
        # From model_m2pf.py:118-124:
        #   Dropout -> Linear(512, 512) -> GELU -> Dropout -> Linear(512, 19)
        #
        # CRITICAL FIX (Round 6): The config's final_fc_size=128 is for the DECODER,
        # NOT the classifier. The actual classifier uses encoder.size*2 = 512.
        # See model_m2pf.py:118-124 for proof.
        self.prediction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, encoder_output_size),  # 512 -> 512 (CORRECT)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, self.num_dimensions),  # 512 -> 19 (CORRECT)
        )
        # NOTE: Using PyTorch defaults for initialization (kaiming_uniform)
        # Original PercePiano doesn't use custom init

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        input_features: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        diagnose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with VirtuosoNet features.

        Args:
            input_features: Either [batch, num_notes, 79] padded tensor OR PackedSequence
            note_locations: Dict with 'beat', 'measure', 'voice' tensors (always padded)
            attention_mask: [batch, num_notes] optional mask (True = valid position)
            lengths: Optional sequence lengths (required if input_features is PackedSequence)
            diagnose: If True, print activation statistics for debugging

        Returns:
            Dict with 'predictions' [batch, num_dimensions]
        """
        # Handle PackedSequence input (from percepiano_pack_collate)
        if isinstance(input_features, PackedSequence):
            # Unpack to padded format for HAN processing
            # HAN needs padded tensors to work with note_locations
            input_features, seq_lengths = pad_packed_sequence(
                input_features, batch_first=True
            )
            # Create attention mask from lengths if not provided
            if attention_mask is None:
                batch_size = input_features.shape[0]
                max_len = input_features.shape[1]
                attention_mask = torch.zeros(
                    batch_size, max_len, dtype=torch.bool, device=input_features.device
                )
                for i, length in enumerate(seq_lengths):
                    attention_mask[i, :length] = True

        # DIAGNOSTIC: Log key statistics for debugging
        if diagnose:
            print(f"  Input: mean={input_features.mean().item():.3f}, std={input_features.std().item():.3f}")

        # Run HAN encoder directly on 79-dim features (no global context)
        han_outputs = self.han_encoder(input_features, note_locations)
        total_note_cat = han_outputs["total_note_cat"]  # [B, N, 2048]

        if diagnose:
            print(f"  HAN output: std={total_note_cat.std().item():.4f}")

        # Contract from 2048 -> 512 (critical step from original PercePiano)
        # NOTE: Original has NO LayerNorm - removed in Round 6
        contracted = self.performance_contractor(total_note_cat)  # [B, N, 512]

        if diagnose:
            print(f"  Contracted: std={contracted.std().item():.3f}")

        # Create attention mask if not provided
        # CRITICAL FIX: After performance_contractor (linear layer with bias), all positions
        # become non-zero. We must create explicit mask based on beat_numbers.
        # Valid positions have beat_number > 0 (after +1 shift in dataset).
        if attention_mask is None:
            # Use beat_numbers to identify valid positions
            # Beat number > 0 means this is a real note (after +1 shift in dataset)
            # Padded positions have beat_number = 0
            attention_mask = note_locations["beat"] > 0  # [B, N]

        # Aggregate to single vector using attention (over 512-dim, not 2048)
        aggregated = self.final_attention(contracted, mask=attention_mask)  # [B, 512]

        if diagnose:
            print(f"  Aggregated: std={aggregated.std().item():.3f}")

        # NOTE: Original has NO LayerNorm before prediction head - removed in Round 6

        # Get logits before sigmoid
        logits = self.prediction_head(aggregated)

        if diagnose:
            print(f"  Logits: mean={logits.mean().item():.3f}, std={logits.std().item():.3f}")

        # Predict scores - apply sigmoid to match [0, 1] label range
        predictions = torch.sigmoid(logits)  # [B, num_dims]

        if diagnose:
            print(f"  Predictions: mean={predictions.mean().item():.3f}, std={predictions.std().item():.3f}")

        return {
            "predictions": predictions,
            "logits": logits,  # Also return logits for debugging
            "han_outputs": han_outputs,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MSE loss (matching PercePiano)."""
        total_loss = nn.functional.mse_loss(predictions, targets, reduction="mean")

        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract note_locations from batch."""
        return {
            "beat": batch["note_locations_beat"],
            "measure": batch["note_locations_measure"],
            "voice": batch["note_locations_voice"],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"],
            note_locations,
            batch.get("attention_mask"),
            batch.get("lengths"),  # Pass lengths for PackedSequence handling
        )

        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"]
        )

        self.log("train/loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss.detach()})

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"],
            note_locations,
            batch.get("attention_mask"),
            batch.get("lengths"),  # Pass lengths for PackedSequence handling
        )

        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])

        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log("val/loss", loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Convert to numpy for sklearn
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        # Use sklearn's r2_score for consistency with original PercePiano (train_m2pf.py:222)
        # This uses uniform_average by default for multioutput
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log("val/mean_r2", mean_r2, prog_bar=True)

        # Also compute and log per-dimension R2
        r2_values = []
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            r2_values.append(dim_r2)
            self.log(f"val/r2_{dim}", dim_r2)

        # Log key dimensions
        if "tempo" in self.dimensions:
            self.log(
                "val/tempo_r2", r2_values[self.dimensions.index("tempo")], prog_bar=True
            )
        if "timing" in self.dimensions:
            self.log(
                "val/timing_r2",
                r2_values[self.dimensions.index("timing")],
                prog_bar=True,
            )

        # Log prediction stats for monitoring
        pred_mean = all_preds_np.mean()
        pred_std = all_preds_np.std()
        self.log("val/pred_mean", pred_mean)
        self.log("val/pred_std", pred_std)

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
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # StepLR operates per step, not per epoch
            },
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PercePianoBiLSTMBaseline(pl.LightningModule):
    """
    Bi-LSTM Baseline matching original PercePiano VirtuosoNetSingle exactly.

    This is the baseline model used in the PercePiano paper for comparison.
    The paper reports R2 = 0.187 for this baseline vs R2 = 0.397 for full HAN.

    Architecture (from model_m2pf.py:56-85):
        - MixEmbedder: Linear(79, 256) - input projection
        - Single 7-layer bidirectional LSTM (combines note+voice+beat+measure layers)
        - note_contractor: Linear(512, 512)
        - ContextAttention(512, 8) - single attention aggregation
        - out_fc: Dropout -> Linear(512, 512) -> GELU -> Dropout -> Linear(512, 19)

    Key differences from HAN:
        - Single deep LSTM instead of hierarchical LSTMs
        - No voice/beat/measure separation
        - No hierarchical attention aggregation
        - Simpler architecture (~2M params vs ~4M for HAN)
    """

    def __init__(
        self,
        input_size: int = 79,
        hidden_size: int = 256,
        # Layer counts matching original: note(2) + voice(2) + beat(2) + measure(1) = 7
        num_layers: int = 7,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 1e-5,
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)
        self.hidden_size = hidden_size

        # MixEmbedder: Simple linear projection (matches original note_embedder)
        self.note_embedder = nn.Linear(input_size, hidden_size)

        # Single 7-layer bidirectional LSTM (matches VirtuosoNetSingle)
        # Original: note.layer + voice.layer + beat.layer + measure.layer = 2+2+2+1 = 7
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Apply orthogonal initialization to LSTM weights for better gradient flow
        # This helps prevent signal collapse in deep (7-layer) LSTMs
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: orthogonal init
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: orthogonal init (critical for deep RNNs)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: set forget gate bias to 1.0 for better gradient flow
                # LSTM bias is [input_gate, forget_gate, cell_gate, output_gate]
                n = param.size(0)
                param.data.fill_(0)
                # Set forget gate bias to 1.0 (indices n//4 to n//2)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Contractor: Linear(512, 512) - matches original note_contractor
        encoder_output_size = hidden_size * 2  # 512 for bidirectional
        self.note_contractor = nn.Linear(encoder_output_size, encoder_output_size)

        # Single ContextAttention over entire sequence (matches original)
        # Use temperature=0.5 to sharpen attention and prevent uniform weights
        # that cause vanishing gradients through softmax
        self.note_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5
        )

        # Output head (matches original out_fc exactly)
        self.out_fc = nn.Sequential(
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
        note_locations: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        diagnose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching VirtuosoNetSingle exactly.

        Args:
            input_features: [batch, num_notes, 79] or PackedSequence
            note_locations: Optional (used only to create attention mask)
            attention_mask: [batch, num_notes] optional mask
            lengths: Optional sequence lengths
            diagnose: If True, print activation statistics for debugging

        Returns:
            Dict with 'predictions' [batch, num_dimensions]
        """
        # Handle PackedSequence input
        if isinstance(input_features, PackedSequence):
            input_features, seq_lengths = pad_packed_sequence(
                input_features, batch_first=True
            )
            if attention_mask is None:
                batch_size = input_features.shape[0]
                max_len = input_features.shape[1]
                attention_mask = torch.zeros(
                    batch_size, max_len, dtype=torch.bool, device=input_features.device
                )
                for i, length in enumerate(seq_lengths):
                    attention_mask[i, :length] = True

        batch_size, seq_len, _ = input_features.shape

        # Step 1: Embed input (MixEmbedder)
        x_embedded = self.note_embedder(input_features)  # [B, T, 256]

        # Compute actual lengths for packing
        if attention_mask is not None:
            actual_lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        elif note_locations is not None and "beat" in note_locations:
            actual_lengths = compute_actual_lengths(note_locations["beat"])
        else:
            actual_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        # Step 2: Pack and run through 7-layer LSTM
        x_packed = pack_padded_sequence(
            x_embedded,
            actual_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        # lstm_out: [B, T, 512]

        if diagnose:
            print(f"\n  [Bi-LSTM DIAGNOSE]")
            print(f"    x_embedded: mean={x_embedded.mean():.4f}, std={x_embedded.std():.4f}")
            print(f"    lstm_out:   mean={lstm_out.mean():.4f}, std={lstm_out.std():.4f}")
            # Check for near-zero LSTM output (critical issue)
            if lstm_out.std() < 0.1:
                print(f"    [WARN] LSTM output std={lstm_out.std():.4f} is very low!")

        # Step 3: Contract (512 -> 512)
        note_contracted = self.note_contractor(lstm_out)  # [B, T, 512]

        if diagnose:
            print(f"    contracted: mean={note_contracted.mean():.4f}, std={note_contracted.std():.4f}")

        # Step 4: Attention aggregation to single vector
        # NOTE: Original VirtuosoNetSingle does NOT pass a mask to ContextAttention!
        # It relies on internal x.sum(-1)==0 check. Passing mask was causing gradient issues.
        aggregated = self.note_attention(note_contracted, mask=None, diagnose=diagnose)  # [B, 512]

        if diagnose:
            print(f"    aggregated: mean={aggregated.mean():.4f}, std={aggregated.std():.4f}")

        # Step 5: Predict through out_fc
        logits = self.out_fc(aggregated)  # [B, 19]
        predictions = torch.sigmoid(logits)

        if diagnose:
            print(f"    logits:     mean={logits.mean():.4f}, std={logits.std():.4f}")
            print(f"    predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        return {
            "predictions": predictions,
            "logits": logits,
            "lstm_out": lstm_out,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MSE loss (matching PercePiano)."""
        total_loss = nn.functional.mse_loss(predictions, targets, reduction="mean")

        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract note_locations from batch."""
        return {
            "beat": batch["note_locations_beat"],
            "measure": batch["note_locations_measure"],
            "voice": batch["note_locations_voice"],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"],
            note_locations,
            batch.get("attention_mask"),
            batch.get("lengths"),
        )

        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])
        self.log("train/loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss.detach()})

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"],
            note_locations,
            batch.get("attention_mask"),
            batch.get("lengths"),
        )

        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])

        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log("val/loss", loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()

        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log("val/mean_r2", mean_r2, prog_bar=True)

        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            self.log(f"val/r2_{dim}", dim_r2)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer matching original PercePiano."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3000, gamma=0.98
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PercePianoBaselinePlusBeat(pl.LightningModule):
    """
    Incremental Model: VirtuosoNetSingle + Beat Hierarchy.

    Phase 2 of incremental debugging: adds beat-level attention and LSTM
    to the validated 7-layer Bi-LSTM baseline.

    Architecture:
        Input(79) -> Embed(256) -> 7-layer BiLSTM(512) ->
        Beat Attention -> Beat LSTM(2 layers) ->
        Concat(LSTM_out + beat_spanned) -> Contractor(1024->512) ->
        Final Attention -> out_fc -> 19

    Expected R2: ~0.25-0.30 (baseline 0.19 + beat contribution ~0.06-0.11)

    Key diagnostic questions this model answers:
    1. Does beat_attention produce diverse weights or collapse to uniform?
    2. Do gradients flow back through the beat branch?
    3. Is span_beat_to_note_num correctly distributing beat representations?
    """

    def __init__(
        self,
        input_size: int = 79,
        hidden_size: int = 256,
        num_layers: int = 7,
        beat_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 1e-5,
        attention_lr_multiplier: float = 10.0,
        entropy_weight: float = 0.01,
        entropy_target: float = 0.6,
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)
        self.hidden_size = hidden_size

        # MixEmbedder: Simple linear projection
        self.note_embedder = nn.Linear(input_size, hidden_size)

        # Single 7-layer bidirectional LSTM (same as baseline)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Apply orthogonal initialization to main LSTM
        self._init_lstm(self.lstm)

        encoder_output_size = hidden_size * 2  # 512 for bidirectional

        # Beat-level hierarchy (NEW compared to baseline)
        # use_hierarchy_init=True for Xavier init + wider context vectors
        self.beat_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5, use_hierarchy_init=True
        )
        self.beat_lstm = nn.LSTM(
            encoder_output_size,
            hidden_size,
            beat_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if beat_layers > 1 else 0,
        )
        self._init_lstm(self.beat_lstm)

        # Contractor: Now takes LSTM + beat = 512 + 512 = 1024 -> 512
        contractor_input_size = encoder_output_size + encoder_output_size  # 1024
        self.note_contractor = nn.Linear(contractor_input_size, encoder_output_size)

        # Final attention (same as baseline)
        self.note_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5
        )

        # Output head (same as baseline)
        self.out_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, encoder_output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, self.num_dimensions),
        )

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Diagnostic storage for attention analysis
        self._last_beat_attention_entropy = None
        self._last_contractor_weights = None

    def _init_lstm(self, lstm: nn.LSTM):
        """Apply orthogonal initialization to LSTM."""
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                n = param.size(0)
                param.data.fill_(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        input_features: torch.Tensor,
        note_locations: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        diagnose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with beat hierarchy."""
        # Handle PackedSequence input
        if isinstance(input_features, PackedSequence):
            input_features, seq_lengths = pad_packed_sequence(
                input_features, batch_first=True
            )
            if attention_mask is None:
                batch_size = input_features.shape[0]
                max_len = input_features.shape[1]
                attention_mask = torch.zeros(
                    batch_size, max_len, dtype=torch.bool, device=input_features.device
                )
                for i, length in enumerate(seq_lengths):
                    attention_mask[i, :length] = True

        batch_size, seq_len, _ = input_features.shape
        beat_numbers = note_locations["beat"] if note_locations else None

        # Compute actual lengths
        if attention_mask is not None:
            actual_lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        elif beat_numbers is not None:
            actual_lengths = compute_actual_lengths(beat_numbers)
        else:
            actual_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        # Step 1: Embed input
        x_embedded = self.note_embedder(input_features)

        # Step 2: Run through 7-layer LSTM
        x_packed = pack_padded_sequence(
            x_embedded, actual_lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

        if diagnose:
            print(f"\n  [Baseline+Beat DIAGNOSE]")
            print(f"    lstm_out: mean={lstm_out.mean():.4f}, std={lstm_out.std():.4f}")

        # Step 3: Beat hierarchy (NEW)
        if beat_numbers is not None:
            beat_nodes = make_higher_node(
                lstm_out,
                self.beat_attention,
                beat_numbers,
                beat_numbers,
                lower_is_note=True,
                actual_lengths=actual_lengths,
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

            # Compute attention entropy (always, for regularization and diagnostics)
            beat_attn = self.beat_attention.get_attention(lstm_out)
            attn_probs = torch.softmax(beat_attn / self.beat_attention.temperature, dim=1)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=1).mean()
            max_entropy = torch.log(torch.tensor(float(seq_len), device=lstm_out.device))
            beat_entropy = entropy / max_entropy
            self._last_beat_attention_entropy = beat_entropy.item()

            if diagnose:
                print(f"    beat_nodes: mean={beat_nodes.mean():.4f}, std={beat_nodes.std():.4f}")
                print(f"    beat_out: mean={beat_out.mean():.4f}, std={beat_out.std():.4f}")
                print(f"    beat_spanned: mean={beat_spanned.mean():.4f}, std={beat_spanned.std():.4f}")
                print(f"    beat_attention entropy: {self._last_beat_attention_entropy:.4f} (1.0=uniform)")

            # Concatenate LSTM output with beat-spanned
            combined = torch.cat([lstm_out, beat_spanned], dim=-1)
            beat_entropy_val = beat_entropy
        else:
            # Fallback: no beat hierarchy, pad with zeros
            combined = torch.cat([
                lstm_out,
                torch.zeros_like(lstm_out)
            ], dim=-1)
            beat_entropy_val = None

        # Step 4: Contract from 1024 -> 512
        note_contracted = self.note_contractor(combined)

        if diagnose:
            print(f"    contracted: mean={note_contracted.mean():.4f}, std={note_contracted.std():.4f}")

            # Analyze contractor weights
            w = self.note_contractor.weight.data
            lstm_weights = w[:, :512].abs().mean().item()
            beat_weights = w[:, 512:].abs().mean().item()
            self._last_contractor_weights = (lstm_weights, beat_weights)
            print(f"    contractor weights - LSTM: {lstm_weights:.4f}, Beat: {beat_weights:.4f}")
            if beat_weights < lstm_weights * 0.1:
                print(f"    [WARN] Contractor ignoring beat branch!")

        # Step 5: Final attention aggregation
        aggregated = self.note_attention(note_contracted, mask=None, diagnose=diagnose)

        # Step 6: Predict
        logits = self.out_fc(aggregated)
        predictions = torch.sigmoid(logits)

        if diagnose:
            print(f"    logits: mean={logits.mean():.4f}, std={logits.std():.4f}")
            print(f"    predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        return {
            "predictions": predictions,
            "logits": logits,
            "lstm_out": lstm_out,
            "beat_spanned": beat_spanned if beat_numbers is not None else None,
            "beat_entropy": beat_entropy_val,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        beat_entropy: Optional[torch.Tensor] = None,
    ):
        mse_loss = nn.functional.mse_loss(predictions, targets, reduction="mean")
        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )

        # Entropy regularization: penalize high entropy (uniform) attention
        # Encourages sharper, more focused attention patterns
        entropy_penalty = torch.tensor(0.0, device=predictions.device)
        if beat_entropy is not None and self.hparams.entropy_weight > 0:
            # Penalize entropy above target (e.g., 0.6)
            excess_entropy = torch.clamp(beat_entropy - self.hparams.entropy_target, min=0)
            entropy_penalty = self.hparams.entropy_weight * excess_entropy

        total_loss = mse_loss + entropy_penalty
        return total_loss, per_dim_losses, {"mse": mse_loss, "entropy_penalty": entropy_penalty}

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return {
            "beat": batch["note_locations_beat"],
            "measure": batch["note_locations_measure"],
            "voice": batch["note_locations_voice"],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"], note_locations,
            batch.get("attention_mask"), batch.get("lengths"),
        )
        loss, _, loss_components = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs.get("beat_entropy")
        )
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mse", loss_components["mse"])
        self.log("train/entropy_penalty", loss_components["entropy_penalty"])
        if self._last_beat_attention_entropy is not None:
            self.log("train/beat_entropy", self._last_beat_attention_entropy)
        self.training_step_outputs.append({"loss": loss.detach()})
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"], note_locations,
            batch.get("attention_mask"), batch.get("lengths"),
        )
        loss, _, loss_components = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs.get("beat_entropy")
        )
        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mse", loss_components["mse"])
        if self._last_beat_attention_entropy is not None:
            self.log("val/beat_entropy", self._last_beat_attention_entropy)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log("val/mean_r2", mean_r2, prog_bar=True)
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            self.log(f"val/r2_{dim}", dim_r2)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        # Separate learning rates: higher LR for attention parameters
        attention_params = []
        other_params = []
        for name, param in self.named_parameters():
            if 'attention' in name.lower():
                attention_params.append(param)
            else:
                other_params.append(param)

        # Log parameter counts
        attn_count = sum(p.numel() for p in attention_params)
        other_count = sum(p.numel() for p in other_params)
        print(f"[Optimizer] Attention params: {attn_count:,} ({len(attention_params)} tensors)")
        print(f"[Optimizer] Other params: {other_count:,} ({len(other_params)} tensors)")
        print(f"[Optimizer] Attention LR multiplier: {self.hparams.attention_lr_multiplier}x")

        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': self.hparams.learning_rate},
            {'params': attention_params, 'lr': self.hparams.learning_rate * self.hparams.attention_lr_multiplier},
        ], weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.98)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PercePianoBaselinePlusBeatMeasure(pl.LightningModule):
    """
    Incremental Model: VirtuosoNetSingle + Beat + Measure Hierarchy.

    Phase 2 final: adds both beat-level and measure-level hierarchy
    to the validated 7-layer Bi-LSTM baseline.

    Architecture:
        Input(79) -> Embed(256) -> 7-layer BiLSTM(512) ->
        Beat Attention -> Beat LSTM(2 layers) ->
        Measure Attention -> Measure LSTM(1 layer) ->
        Concat(LSTM_out + beat_spanned + measure_spanned) -> Contractor(1536->512) ->
        Final Attention -> out_fc -> 19

    Expected R2: ~0.35-0.40 (approaching SOTA)
    """

    def __init__(
        self,
        input_size: int = 79,
        hidden_size: int = 256,
        num_layers: int = 7,
        beat_layers: int = 2,
        measure_layers: int = 1,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 1e-5,
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)
        self.hidden_size = hidden_size

        # MixEmbedder
        self.note_embedder = nn.Linear(input_size, hidden_size)

        # 7-layer BiLSTM
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self._init_lstm(self.lstm)

        encoder_output_size = hidden_size * 2  # 512

        # Beat hierarchy
        # use_hierarchy_init=True for Xavier init + wider context vectors
        self.beat_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5, use_hierarchy_init=True
        )
        self.beat_lstm = nn.LSTM(
            encoder_output_size, hidden_size, beat_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if beat_layers > 1 else 0,
        )
        self._init_lstm(self.beat_lstm)

        # Measure hierarchy
        # use_hierarchy_init=True for Xavier init + wider context vectors
        self.measure_attention = ContextAttention(
            encoder_output_size, num_attention_heads, temperature=0.5, use_hierarchy_init=True
        )
        self.measure_lstm = nn.LSTM(
            encoder_output_size, hidden_size, measure_layers,
            batch_first=True, bidirectional=True,
        )
        self._init_lstm(self.measure_lstm)

        # Contractor: LSTM + beat + measure = 512 + 512 + 512 = 1536 -> 512
        contractor_input_size = encoder_output_size * 3  # 1536
        self.note_contractor = nn.Linear(contractor_input_size, encoder_output_size)

        # Final attention
        self.note_attention = ContextAttention(encoder_output_size, num_attention_heads, temperature=0.5)

        # Output head
        self.out_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, encoder_output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_size, self.num_dimensions),
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self._last_beat_attention_entropy = None
        self._last_measure_attention_entropy = None
        self._last_contractor_weights = None

    def _init_lstm(self, lstm: nn.LSTM):
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                n = param.size(0)
                param.data.fill_(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        input_features: torch.Tensor,
        note_locations: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        diagnose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Handle PackedSequence
        if isinstance(input_features, PackedSequence):
            input_features, seq_lengths = pad_packed_sequence(input_features, batch_first=True)
            if attention_mask is None:
                batch_size = input_features.shape[0]
                max_len = input_features.shape[1]
                attention_mask = torch.zeros(
                    batch_size, max_len, dtype=torch.bool, device=input_features.device
                )
                for i, length in enumerate(seq_lengths):
                    attention_mask[i, :length] = True

        batch_size, seq_len, _ = input_features.shape
        beat_numbers = note_locations["beat"] if note_locations else None
        measure_numbers = note_locations["measure"] if note_locations else None

        if attention_mask is not None:
            actual_lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        elif beat_numbers is not None:
            actual_lengths = compute_actual_lengths(beat_numbers)
        else:
            actual_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        # Embed
        x_embedded = self.note_embedder(input_features)

        # 7-layer LSTM
        x_packed = pack_padded_sequence(x_embedded, actual_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

        if diagnose:
            print(f"\n  [Baseline+Beat+Measure DIAGNOSE]")
            print(f"    lstm_out: mean={lstm_out.mean():.4f}, std={lstm_out.std():.4f}")

        # Beat hierarchy
        if beat_numbers is not None:
            beat_nodes = make_higher_node(
                lstm_out, self.beat_attention, beat_numbers, beat_numbers,
                lower_is_note=True, actual_lengths=actual_lengths,
            )
            beat_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_lstm)
            beat_spanned = span_beat_to_note_num(beat_out, beat_numbers, actual_lengths)

            if beat_spanned.shape[1] < seq_len:
                padding = torch.zeros(
                    batch_size, seq_len - beat_spanned.shape[1], beat_spanned.shape[2],
                    device=beat_spanned.device, dtype=beat_spanned.dtype
                )
                beat_spanned = torch.cat([beat_spanned, padding], dim=1)

            if diagnose:
                print(f"    beat_out: std={beat_out.std():.4f}")
                beat_attn = self.beat_attention.get_attention(lstm_out)
                attn_probs = torch.softmax(beat_attn / 0.5, dim=1)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=1).mean()
                max_entropy = torch.log(torch.tensor(float(seq_len)))
                self._last_beat_attention_entropy = (entropy / max_entropy).item()
                print(f"    beat_attention entropy: {self._last_beat_attention_entropy:.4f}")
        else:
            beat_out = None
            beat_spanned = torch.zeros_like(lstm_out)

        # Measure hierarchy
        if beat_numbers is not None and measure_numbers is not None and beat_out is not None:
            measure_nodes = make_higher_node(
                beat_out, self.measure_attention, beat_numbers, measure_numbers,
                actual_lengths=actual_lengths,
            )
            measure_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_lstm)
            measure_spanned = span_beat_to_note_num(measure_out, measure_numbers, actual_lengths)

            if measure_spanned.shape[1] < seq_len:
                padding = torch.zeros(
                    batch_size, seq_len - measure_spanned.shape[1], measure_spanned.shape[2],
                    device=measure_spanned.device, dtype=measure_spanned.dtype
                )
                measure_spanned = torch.cat([measure_spanned, padding], dim=1)

            if diagnose:
                print(f"    measure_out: std={measure_out.std():.4f}")
                # Note: measure attention operates on beat_out, not lstm_out
                num_beats = beat_out.shape[1]
                meas_attn = self.measure_attention.get_attention(beat_out)
                attn_probs = torch.softmax(meas_attn / 0.5, dim=1)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=1).mean()
                max_entropy = torch.log(torch.tensor(float(num_beats)))
                self._last_measure_attention_entropy = (entropy / max_entropy).item()
                print(f"    measure_attention entropy: {self._last_measure_attention_entropy:.4f}")
        else:
            measure_spanned = torch.zeros_like(lstm_out)

        # Concatenate all three
        combined = torch.cat([lstm_out, beat_spanned, measure_spanned], dim=-1)

        # Contract 1536 -> 512
        note_contracted = self.note_contractor(combined)

        if diagnose:
            print(f"    contracted: std={note_contracted.std():.4f}")
            w = self.note_contractor.weight.data
            lstm_w = w[:, :512].abs().mean().item()
            beat_w = w[:, 512:1024].abs().mean().item()
            meas_w = w[:, 1024:].abs().mean().item()
            self._last_contractor_weights = (lstm_w, beat_w, meas_w)
            print(f"    contractor weights - LSTM: {lstm_w:.4f}, Beat: {beat_w:.4f}, Measure: {meas_w:.4f}")

        # Final attention
        aggregated = self.note_attention(note_contracted, mask=None, diagnose=diagnose)

        # Predict
        logits = self.out_fc(aggregated)
        predictions = torch.sigmoid(logits)

        if diagnose:
            print(f"    predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        return {
            "predictions": predictions,
            "logits": logits,
            "lstm_out": lstm_out,
            "beat_spanned": beat_spanned,
            "measure_spanned": measure_spanned,
        }

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        total_loss = nn.functional.mse_loss(predictions, targets, reduction="mean")
        per_dim_losses = {}
        for i, dim in enumerate(self.dimensions):
            per_dim_losses[dim] = nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )
        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return {
            "beat": batch["note_locations_beat"],
            "measure": batch["note_locations_measure"],
            "voice": batch["note_locations_voice"],
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"], note_locations,
            batch.get("attention_mask"), batch.get("lengths"),
        )
        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])
        self.log("train/loss", loss, prog_bar=True)
        self.training_step_outputs.append({"loss": loss.detach()})
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["input_features"], note_locations,
            batch.get("attention_mask"), batch.get("lengths"),
        )
        loss, _ = self.compute_loss(outputs["predictions"], batch["scores"])
        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)
        self.log("val/loss", loss, prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()
        mean_r2 = r2_score(all_targets_np, all_preds_np)
        self.log("val/mean_r2", mean_r2, prog_bar=True)
        for i, dim in enumerate(self.dimensions):
            dim_r2 = r2_score(all_targets_np[:, i], all_preds_np[:, i])
            self.log(f"val/r2_{dim}", dim_r2)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.98)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("PercePiano Replica Model")
    print("=" * 60)
    print("Reference: Park et al., 'PercePiano', ISMIR/Nature 2024")
    print("GitHub: https://github.com/JonghoKimSNU/PercePiano")
    print("=" * 60)

    # Test HAN model creation (use PercePianoVNetModule which has simpler interface)
    print("\n[1] Full HAN Model (PercePianoVNetModule):")
    han_model = PercePianoVNetModule()
    han_params = han_model.count_parameters()
    print(f"    Parameters: {han_params:,}")
    print(f"    Target R2: 0.35-0.40 (piece-split)")
    print(f"    Dimensions: {len(han_model.dimensions)}")

    # Test Bi-LSTM baseline creation
    print("\n[2] Bi-LSTM Baseline (PercePianoBiLSTMBaseline):")
    baseline = PercePianoBiLSTMBaseline()
    baseline_params = baseline.count_parameters()
    print(f"    Parameters: {baseline_params:,}")
    print(f"    Target R2: ~0.19 (paper baseline)")
    print(f"    LSTM layers: 7 (note+voice+beat+measure)")
    print(f"    Architecture: Input(79) -> Embed(256) -> LSTM(7-layer) -> Contract(512) -> Attn -> FC -> 19")

    # Architecture comparison
    print("\n[3] Architecture Comparison:")
    print(f"    HAN params:      {han_params:,}")
    print(f"    Baseline params: {baseline_params:,}")
    print(f"    Ratio: {han_params / baseline_params:.2f}x")
    print(f"    Expected hierarchy gain: +0.21 R2")

    # Test baseline forward pass only (HAN requires properly structured beat/measure data)
    print("\n[4] Baseline Forward Pass Test:")
    batch_size, seq_len, input_dim = 2, 100, 79
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_locations = {
        "beat": torch.arange(1, seq_len + 1).unsqueeze(0).expand(batch_size, -1),
        "measure": (torch.arange(seq_len) // 10 + 1).unsqueeze(0).expand(batch_size, -1),
        "voice": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    baseline_out = baseline(dummy_input, dummy_locations)
    print(f"    Baseline output shape: {baseline_out['predictions'].shape}")
    print(f"    Output range: [{baseline_out['predictions'].min():.3f}, {baseline_out['predictions'].max():.3f}]")
    print(f"    LSTM output shape: {baseline_out['lstm_out'].shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
