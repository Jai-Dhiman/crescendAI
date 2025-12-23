"""
Score-Aligned Lightning Module for piano performance evaluation.

Extends the MIDI-only approach by incorporating score alignment features
that compare performance MIDI to reference score MIDI/MusicXML.

Following PercePiano research findings:
- Score alignment provides 21% absolute R-squared improvement
- Key for tempo dimension (currently R-squared = -0.15)
- Enables evaluation against composer's intent
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.shared.models.aggregation import (
    HierarchicalAggregator,
    PercePianoSelfAttention,
)
from src.shared.models.mtl_head import MultiTaskHead, PercePianoHead

from .calibration import IsotonicCalibrator, TemperatureScaling
from .midi_encoder import MIDIBertEncoder
from .score_encoder import (
    HierarchicalScoreEncoder,
    ScoreAlignmentEncoder,
    ScoreMIDIFusion,
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


class ScoreAlignedModule(pl.LightningModule):
    """
    PyTorch Lightning module for score-aligned piano performance evaluation.

    Architecture:
        MIDI tokens -> MIDIBert Encoder -> Self-Attention Aggregation
                                                    |
                                                    v
        Score Features -> Score Encoder ----------> Fusion -> MTL Head -> 19 Scores

    The score alignment features capture:
    - Timing deviations from score
    - Tempo variations relative to marked tempo
    - Dynamic deviations from marked dynamics
    - Articulation adherence

    This should significantly improve tempo dimension (currently R-squared = -0.15)
    and overall model performance (target R-squared = 0.30-0.40).
    """

    def __init__(
        self,
        # MIDI Encoder params
        midi_hidden_dim: int = 768,
        midi_num_layers: int = 12,
        midi_num_heads: int = 12,
        max_seq_length: int = 512,
        # Score Encoder params
        score_note_features: int = 20,  # Expanded to 20 features
        score_global_features: int = 12,
        score_hidden_dim: int = 256,
        score_num_layers: int = 2,
        # Aggregation params
        attention_da: int = 128,
        attention_r: int = 4,
        # Fusion params
        fusion_type: str = "gated",  # 'concat', 'crossattn', or 'gated'
        fused_dim: int = 768,
        # MTL head params
        head_hidden_dim: int = 256,
        # Training params
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        # Task params
        dimensions: Optional[List[str]] = None,
        # Optional: freeze MIDI encoder initially
        freeze_midi_encoder: bool = False,
        # Hierarchical encoder option
        use_hierarchical_encoder: bool = False,
        # Pre-trained MIDI encoder checkpoint
        midi_pretrained_checkpoint: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)
        self.freeze_midi_encoder = freeze_midi_encoder
        self.use_hierarchical_encoder = use_hierarchical_encoder

        # MIDI Encoder (12 layers, 768 hidden to match MidiBERT)
        self.midi_encoder = MIDIBertEncoder(
            hidden_size=midi_hidden_dim,
            num_layers=midi_num_layers,
            num_heads=midi_num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        # Load pre-trained MIDI encoder weights if provided
        if midi_pretrained_checkpoint is not None:
            pretrained_path = Path(midi_pretrained_checkpoint)
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Pre-trained MIDI encoder checkpoint not found: {pretrained_path}\n"
                    "\n"
                    "ACTION REQUIRED:\n"
                    "1. Download from GDrive:\n"
                    "   rclone copy gdrive:crescendai_checkpoints/midi_pretrain/encoder_pretrained.pt /tmp/checkpoints/\n"
                    "2. Or run pre-training first:\n"
                    "   python scripts/pretrain_midi_encoder.py --midi_dir <path>"
                )
            self.midi_encoder.load_pretrained(pretrained_path)
            print(f"[OK] Loaded pre-trained MIDI encoder from: {pretrained_path}")

        if freeze_midi_encoder:
            self.midi_encoder.freeze()

        # MIDI Aggregation (PercePiano Self-Attention)
        self.midi_aggregator = PercePianoSelfAttention(
            input_dim=midi_hidden_dim,
            da=attention_da,
            r=attention_r,
        )
        midi_aggregated_dim = attention_r * midi_hidden_dim  # r * D

        # Score Alignment Encoder (flat or hierarchical)
        if use_hierarchical_encoder:
            self.score_encoder = HierarchicalScoreEncoder(
                note_features=score_note_features,
                global_features=score_global_features,
                hidden_dim=score_hidden_dim,
                note_size=score_hidden_dim // 2,
                beat_size=score_hidden_dim // 4,
                measure_size=score_hidden_dim // 4,
                num_note_layers=score_num_layers,
                num_attention_heads=4,
                dropout=dropout,
                use_voice_processing=False,  # Start without voice for simplicity
                output_mode="global",
            )
        else:
            self.score_encoder = ScoreAlignmentEncoder(
                note_features=score_note_features,
                global_features=score_global_features,
                hidden_dim=score_hidden_dim,
                num_note_layers=score_num_layers,
                num_heads=4,
                dropout=dropout,
                output_mode="global",  # Use global representation for fusion
            )

        # Fusion module
        self.fusion = ScoreMIDIFusion(
            midi_dim=midi_aggregated_dim,
            score_dim=score_hidden_dim,
            output_dim=fused_dim,
            fusion_type=fusion_type,
            dropout=dropout,
        )

        # MTL Head (simple 2-layer for PercePiano style)
        self.mtl_head = PercePianoHead(
            input_dim=fused_dim,
            num_dims=self.num_dimensions,
            hidden_dim=head_hidden_dim,
        )

        # Learnable log-variance for uncertainty-weighted loss
        self.log_vars = nn.Parameter(torch.zeros(self.num_dimensions))

        # Calibration support
        self.isotonic_calibrator: Optional[IsotonicCalibrator] = None
        self.temperature_scaling: Optional[TemperatureScaling] = None
        self.calibration_method: Optional[str] = None

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        midi_tokens: torch.Tensor,
        score_note_features: torch.Tensor,
        score_global_features: torch.Tensor,
        score_tempo_curve: torch.Tensor,
        midi_attention_mask: Optional[torch.Tensor] = None,
        score_attention_mask: Optional[torch.Tensor] = None,
        note_locations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with both MIDI and score alignment features.

        Args:
            midi_tokens: [batch, seq_len, 8] OctupleMIDI tokens
            score_note_features: [batch, num_notes, 20] per-note deviations (expanded)
            score_global_features: [batch, 12] aggregated statistics
            score_tempo_curve: [batch, num_segments] tempo ratios
            midi_attention_mask: [batch, seq_len] MIDI attention mask
            score_attention_mask: [batch, num_notes] score attention mask
            note_locations: Dict with 'beat', 'measure', 'voice' tensors for hierarchical encoder

        Returns:
            Dict with 'predictions' [batch, num_dimensions] and 'log_vars'
        """
        # Encode MIDI
        midi_features = self.midi_encoder(midi_tokens, midi_attention_mask)  # [B, T, H]

        # Aggregate MIDI over time
        midi_aggregated, _ = self.midi_aggregator(
            midi_features, midi_attention_mask
        )  # [B, r*H]

        # Encode score alignment features
        if self.use_hierarchical_encoder:
            score_outputs = self.score_encoder(
                score_note_features,
                score_global_features,
                score_tempo_curve,
                note_locations=note_locations,
                attention_mask=score_attention_mask,
            )
        else:
            score_outputs = self.score_encoder(
                score_note_features,
                score_global_features,
                score_tempo_curve,
                score_attention_mask,
            )
        score_features = score_outputs["global"]  # [B, score_hidden_dim]

        # Fuse MIDI and score features
        fused = self.fusion(midi_aggregated, score_features)  # [B, fused_dim]

        # Predict scores
        predictions, _ = self.mtl_head(fused)  # [B, num_dims]

        # Apply sigmoid to bound predictions to 0-1
        predictions = torch.sigmoid(predictions)

        return {
            "predictions": predictions,
            "log_vars": self.log_vars,
            "midi_features": midi_aggregated,
            "score_features": score_features,
        }

    def forward_midi_only(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MIDI only (for inference without score).

        Creates zero score features as placeholder.

        Args:
            midi_tokens: [batch, seq_len, 8] OctupleMIDI tokens
            attention_mask: [batch, seq_len] attention mask

        Returns:
            Dict with 'predictions' [batch, num_dimensions]
        """
        batch_size = midi_tokens.size(0)
        device = midi_tokens.device

        # Create zero score features (use configured number of features)
        num_note_features = self.hparams.score_note_features
        score_note_features = torch.zeros(
            batch_size, 1, num_note_features, device=device
        )
        score_global_features = torch.zeros(batch_size, 12, device=device)
        score_tempo_curve = torch.ones(batch_size, 1, device=device)
        score_attention_mask = torch.ones(batch_size, 1, device=device)

        return self.forward(
            midi_tokens,
            score_note_features,
            score_global_features,
            score_tempo_curve,
            attention_mask,
            score_attention_mask,
        )

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_vars: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MSE loss (matching PercePiano reference).

        Args:
            predictions: [batch, num_dims] predicted scores (0-1 from sigmoid)
            targets: [batch, num_dims] target scores (0-1 scale)
            log_vars: [num_dims] (unused - kept for API compatibility)

        Returns:
            Tuple of (total_loss, per_dimension_losses)
        """
        per_dim_losses = {}

        # Simple MSE loss
        total_loss = torch.nn.functional.mse_loss(
            predictions, targets, reduction="mean"
        )

        # Per-dimension losses for logging
        for i, dim in enumerate(self.dimensions):
            mse = torch.nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            per_dim_losses[dim] = mse

        return total_loss, per_dim_losses

    def _get_note_locations(self, batch: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Extract note_locations from batch if available."""
        if "note_locations_beat" in batch:
            return {
                "beat": batch["note_locations_beat"],
                "measure": batch["note_locations_measure"],
                "voice": batch["note_locations_voice"],
            }
        return None

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["midi_tokens"],
            batch["score_note_features"],
            batch["score_global_features"],
            batch["score_tempo_curve"],
            batch.get("midi_attention_mask"),
            batch.get("score_attention_mask"),
            note_locations=note_locations,
        )
        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        self.log("train/loss", loss, prog_bar=True)
        for dim, dim_loss in per_dim_losses.items():
            self.log(f"train/loss_{dim}", dim_loss)

        self.training_step_outputs.append({"loss": loss.detach()})
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["midi_tokens"],
            batch["score_note_features"],
            batch["score_global_features"],
            batch["score_tempo_curve"],
            batch.get("midi_attention_mask"),
            batch.get("score_attention_mask"),
            note_locations=note_locations,
        )
        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log("val/loss", loss, prog_bar=True)
        return result

    def _safe_pearson_r(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute Pearson correlation with handling for constant arrays."""
        if len(preds) < 2:
            return 0.0
        if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
            return 0.0
        r = np.corrcoef(preds, targets)[0, 1]
        return 0.0 if np.isnan(r) else r

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        r_values = []
        r2_values = []

        for i, dim in enumerate(self.dimensions):
            preds = all_preds[:, i].cpu().numpy()
            targets = all_targets[:, i].cpu().numpy()

            # Pearson correlation
            r = self._safe_pearson_r(preds, targets)
            r_values.append(r)

            # R-squared
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2) + 1e-8
            r2 = 1 - ss_res / ss_tot
            r2_values.append(r2)

            # MAE
            mae = np.mean(np.abs(preds - targets))

            self.log(f"val/r_{dim}", r)
            self.log(f"val/r2_{dim}", r2)
            self.log(f"val/mae_{dim}", mae)

        # Mean metrics
        mean_r = np.mean(r_values)
        mean_r2 = np.mean(r2_values)
        self.log("val/mean_r", mean_r, prog_bar=True)
        self.log("val/mean_r2", mean_r2, prog_bar=True)

        # Specifically track tempo dimension improvement (key for score-aligned training)
        if "tempo" in self.dimensions:
            tempo_idx = self.dimensions.index("tempo")
            self.log("val/tempo_r2", r2_values[tempo_idx], prog_bar=True)

        # Also track timing for custom dimensions
        if "timing" in self.dimensions:
            timing_idx = self.dimensions.index("timing")
            self.log("val/timing_r2", r2_values[timing_idx], prog_bar=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step - same as validation step."""
        note_locations = self._get_note_locations(batch)
        outputs = self(
            batch["midi_tokens"],
            batch["score_note_features"],
            batch["score_global_features"],
            batch["score_tempo_curve"],
            batch.get("midi_attention_mask"),
            batch.get("score_attention_mask"),
            note_locations=note_locations,
        )
        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        self.log("test/loss", loss)
        for dim, dim_loss in per_dim_losses.items():
            self.log(f"test/loss_{dim}", dim_loss)

        return {
            "test_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }

    def on_test_epoch_end(self):
        """Compute test metrics at end of epoch."""
        pass  # Metrics logged per-batch in test_step

    def configure_optimizers(self):
        """Configure optimizer with separate LR for MIDI encoder and rest."""
        # Group parameters
        midi_params = list(self.midi_encoder.parameters())
        other_params = (
            list(self.score_encoder.parameters())
            + list(self.midi_aggregator.parameters())
            + list(self.fusion.parameters())
            + list(self.mtl_head.parameters())
            + [self.log_vars]
        )

        # Different learning rates: lower for pretrained MIDI encoder
        param_groups = [
            {"params": midi_params, "lr": self.hparams.learning_rate * 0.1},
            {"params": other_params, "lr": self.hparams.learning_rate},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def unfreeze_midi_encoder(self):
        """Unfreeze the MIDI encoder for fine-tuning."""
        self.midi_encoder.unfreeze()
        self.freeze_midi_encoder = False

    @classmethod
    def load_for_inference(cls, checkpoint_path: str, device: str = "cuda"):
        """Load model for inference."""
        model = cls.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model.freeze()
        return model


class ScoreAlignedModuleWithFallback(ScoreAlignedModule):
    """
    Score-aligned module with graceful fallback when score is unavailable.

    This variant learns to handle both:
    1. Full mode: Performance MIDI + Score alignment features
    2. Fallback mode: Performance MIDI only (zero score features)

    Training uses both modes to ensure the model works without score.
    """

    def __init__(self, *args, fallback_probability: float = 0.2, **kwargs):
        """
        Args:
            fallback_probability: Probability of training without score features
        """
        super().__init__(*args, **kwargs)
        self.fallback_probability = fallback_probability

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Randomly drop score features during training
        if torch.rand(1).item() < self.fallback_probability:
            # Use fallback mode (MIDI only)
            outputs = self.forward_midi_only(
                batch["midi_tokens"],
                batch.get("midi_attention_mask"),
            )
        else:
            # Use full mode with score
            note_locations = self._get_note_locations(batch)
            outputs = self(
                batch["midi_tokens"],
                batch["score_note_features"],
                batch["score_global_features"],
                batch["score_tempo_curve"],
                batch.get("midi_attention_mask"),
                batch.get("score_attention_mask"),
                note_locations=note_locations,
            )

        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        self.log("train/loss", loss, prog_bar=True)
        for dim, dim_loss in per_dim_losses.items():
            self.log(f"train/loss_{dim}", dim_loss)

        self.training_step_outputs.append({"loss": loss.detach()})
        return loss


if __name__ == "__main__":
    print("Score-aligned module loaded successfully")
    print("Features:")
    print("- MIDI encoder (MIDIBert-style, 12 layers, 768 dim)")
    print("- Score alignment encoder (flat or hierarchical HAN)")
    print("- Expanded 20 note-level features")
    print("- Note locations for hierarchical processing (beat/measure/voice)")
    print("- Gated/concat/crossattn fusion options")
    print("- Fallback mode for inference without score")
    print("- Separate LR for pretrained MIDI encoder")
