"""
MIDI-only Lightning Module for piano performance evaluation.

Architecture aligned with PercePiano reference:
- MIDIBert Encoder (12 layers, 768 hidden to match MidiBERT)
- PercePiano Self-Attention Aggregation
- Simplified 2-layer classification head
- Post-hoc calibration support
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader

from .midi_encoder import MIDIBertEncoder
from .aggregation import HierarchicalAggregator, PercePianoSelfAttention
from .mtl_head import MultiTaskHead, PercePianoHead
from .calibration import CalibrationWrapper, IsotonicCalibrator, TemperatureScaling


# All 19 PercePiano dimensions (matching reference implementation)
PERCEPIANO_DIMENSIONS = [
    "timing",              # 0: Stable <-> Unstable
    "articulation_length", # 1: Short <-> Long
    "articulation_touch",  # 2: Soft/Cushioned <-> Hard/Solid
    "pedal_amount",        # 3: Sparse/Dry <-> Saturated/Wet
    "pedal_clarity",       # 4: Clean <-> Blurred
    "timbre_variety",      # 5: Even <-> Colorful
    "timbre_depth",        # 6: Shallow <-> Rich
    "timbre_brightness",   # 7: Bright <-> Dark
    "timbre_loudness",     # 8: Soft <-> Loud
    "dynamic_range",       # 9: Little Range <-> Large Range
    "tempo",               # 10: Fast-paced <-> Slow-paced
    "space",               # 11: Flat <-> Spacious
    "balance",             # 12: Disproportioned <-> Balanced
    "drama",               # 13: Pure <-> Dramatic
    "mood_valence",        # 14: Optimistic <-> Dark
    "mood_energy",         # 15: Low Energy <-> High Energy
    "mood_imagination",    # 16: Honest <-> Imaginative
    "sophistication",      # 17: Sophisticated/Mellow <-> Raw/Crude
    "interpretation",      # 18: Unsatisfactory <-> Convincing
]


class MIDIOnlyModule(pl.LightningModule):
    """
    PyTorch Lightning module for MIDI-only piano performance evaluation.

    Architecture aligned with PercePiano reference:
        MIDI tokens -> MIDIBert Encoder (768d, 12L) -> Self-Attention Aggregation -> 2-layer Head -> 19 Scores

    Supports post-hoc calibration for fixing systematic bias (positive Pearson r, negative R^2).
    """

    def __init__(
        self,
        # Encoder params (defaults match MidiBERT)
        midi_hidden_dim: int = 768,
        midi_num_layers: int = 12,
        midi_num_heads: int = 12,
        max_seq_length: int = 512,
        # Aggregation params (PercePiano Self-Attention)
        attention_da: int = 128,
        attention_r: int = 4,
        # MTL head params (simplified PercePiano-style)
        head_hidden_dim: int = 256,
        # Training params (matching PercePiano)
        learning_rate: float = 1e-5,  # PercePiano uses 1e-5
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        # Architecture selection
        use_percepiano_architecture: bool = True,  # Use PercePiano-style components
        # Legacy params for backward compatibility
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        shared_hidden: int = 256,
        task_hidden: int = 128,
        warmup_steps: int = 500,
        # Task params
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)
        self.use_percepiano_architecture = use_percepiano_architecture

        # MIDI Encoder (12 layers, 768 hidden to match MidiBERT)
        self.midi_encoder = MIDIBertEncoder(
            hidden_size=midi_hidden_dim,
            num_layers=midi_num_layers,
            num_heads=midi_num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        if use_percepiano_architecture:
            # PercePiano-style: Self-Attention Aggregation
            self.aggregator = PercePianoSelfAttention(
                input_dim=midi_hidden_dim,
                da=attention_da,
                r=attention_r,
            )
            aggregator_output_dim = attention_r * midi_hidden_dim  # r * D

            # PercePiano-style: Simple 2-layer head
            self.mtl_head = PercePianoHead(
                input_dim=aggregator_output_dim,
                num_dims=self.num_dimensions,
                hidden_dim=head_hidden_dim,
            )
        else:
            # Legacy: BiLSTM + Attention Aggregation
            self.aggregator = HierarchicalAggregator(
                input_dim=midi_hidden_dim,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                attention_heads=attention_heads,
                dropout=dropout,
                output_dim=shared_hidden,
            )

            # Legacy: Multi-task head with uncertainty
            self.mtl_head = MultiTaskHead(
                input_dim=shared_hidden,
                shared_hidden=shared_hidden,
                task_hidden=task_hidden,
                dimensions=self.dimensions,
                dropout=dropout,
            )

        # Learnable log-variance for uncertainty-weighted loss (kept for API compatibility)
        self.log_vars = nn.Parameter(torch.zeros(self.num_dimensions))

        # Calibration support (initialized after training)
        self.isotonic_calibrator: Optional[IsotonicCalibrator] = None
        self.temperature_scaling: Optional[TemperatureScaling] = None
        self.calibration_method: Optional[str] = None

        # Metrics storage for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            midi_tokens: [batch, seq_len, 8] OctupleMIDI tokens
            attention_mask: [batch, seq_len] attention mask

        Returns:
            Dict with 'predictions' [batch, num_dimensions] and 'log_vars' [num_dimensions]
        """
        # Encode MIDI
        midi_features = self.midi_encoder(midi_tokens, attention_mask)  # [B, T, H]

        # Aggregate over time (returns tuple: aggregated, attention_weights)
        aggregated, _ = self.aggregator(midi_features, attention_mask)  # [B, H]

        # Predict scores (returns tuple: scores, uncertainties)
        predictions, _ = self.mtl_head(aggregated)  # [B, num_dims]

        # Apply sigmoid to bound predictions to 0-1 (matching PercePiano reference)
        predictions = torch.sigmoid(predictions)

        return {
            "predictions": predictions,
            "log_vars": self.log_vars,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_vars: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute simple MSE loss (matching PercePiano reference).

        Args:
            predictions: [batch, num_dims] predicted scores (0-1 from sigmoid)
            targets: [batch, num_dims] target scores (0-1 scale)
            log_vars: [num_dims] (unused - kept for API compatibility)

        Returns:
            Tuple of (total_loss, per_dimension_losses)
        """
        per_dim_losses = {}

        # Simple MSE loss matching PercePiano reference
        total_loss = torch.nn.functional.mse_loss(predictions, targets, reduction="mean")

        # Also compute per-dimension losses for logging
        for i, dim in enumerate(self.dimensions):
            mse = torch.nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            per_dim_losses[dim] = mse

        return total_loss, per_dim_losses

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch["midi_tokens"], batch.get("attention_mask"))
        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        for dim, dim_loss in per_dim_losses.items():
            self.log(f"train/loss_{dim}", dim_loss)

        self.training_step_outputs.append({"loss": loss.detach()})
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        outputs = self(batch["midi_tokens"], batch.get("attention_mask"))
        loss, per_dim_losses = self.compute_loss(
            outputs["predictions"], batch["scores"], outputs["log_vars"]
        )

        # Store for epoch-end metrics
        result = {
            "val_loss": loss.detach(),
            "predictions": outputs["predictions"].detach(),
            "targets": batch["scores"].detach(),
        }
        self.validation_step_outputs.append(result)

        self.log("val/loss", loss, prog_bar=True)
        return result

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step - same as validation but logs to test/ namespace."""
        outputs = self(batch["midi_tokens"], batch.get("attention_mask"))
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

    def _safe_pearson_r(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute Pearson correlation with handling for constant arrays."""
        if len(preds) < 2:
            return 0.0
        # Check for zero variance (constant values)
        if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
            return 0.0
        r = np.corrcoef(preds, targets)[0, 1]
        # Handle NaN that can still occur in edge cases
        return 0.0 if np.isnan(r) else r

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # Collect all predictions and targets
        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Compute per-dimension metrics
        r_values = []
        for i, dim in enumerate(self.dimensions):
            preds = all_preds[:, i].cpu().numpy()
            targets = all_targets[:, i].cpu().numpy()

            # Pearson correlation (safe version)
            r = self._safe_pearson_r(preds, targets)
            r_values.append(r)
            # R-squared
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2) + 1e-8
            r2 = 1 - ss_res / ss_tot
            # MAE
            mae = np.mean(np.abs(preds - targets))

            self.log(f"val/r_{dim}", r)
            self.log(f"val/r2_{dim}", r2)
            self.log(f"val/mae_{dim}", mae)

        # Mean metrics (use already computed r_values)
        mean_r = np.mean(r_values)
        self.log("val/mean_r", mean_r, prog_bar=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer (constant LR matching PercePiano reference).

        PercePiano uses constant LR of 1e-5 without any scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # No scheduler - constant LR matching PercePiano reference
        return optimizer

    def predict_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Inference step for prediction."""
        outputs = self(batch["midi_tokens"], batch.get("attention_mask"))

        # Predictions are already 0-1 from sigmoid, scale to 0-100 for display
        predictions = outputs["predictions"] * 100

        # Get uncertainty estimates (sigma)
        sigmas = torch.exp(0.5 * outputs["log_vars"])

        return {
            "predictions": predictions,
            "uncertainties": sigmas.expand(predictions.shape[0], -1),
            "dimension_names": self.dimensions,
        }

    @classmethod
    def load_for_inference(cls, checkpoint_path: str, device: str = "cuda"):
        """Load model for inference."""
        model = cls.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model.freeze()
        return model

    def score_midi(self, midi_tokens: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Score a single MIDI performance.

        Args:
            midi_tokens: [1, seq_len, 8] OctupleMIDI tokens

        Returns:
            Dict mapping dimension names to {score, confidence}
        """
        self.eval()
        with torch.no_grad():
            outputs = self(midi_tokens)
            # Predictions are already 0-1 from sigmoid, scale to 0-100 for display
            predictions = outputs["predictions"] * 100
            sigmas = torch.exp(0.5 * outputs["log_vars"])

            # Convert sigma to confidence (inverse relationship)
            # Higher sigma = lower confidence
            confidences = 1.0 / (1.0 + sigmas)

            results = {}
            for i, dim in enumerate(self.dimensions):
                results[dim] = {
                    "score": float(predictions[0, i].cpu()),
                    "confidence": float(confidences[i].cpu()),
                }

            return results

    def calibrate(self, val_loader: DataLoader, method: str = 'both') -> Dict[str, float]:
        """
        Fit post-hoc calibration on validation set.

        This should be called after training to fix systematic bias
        (positive Pearson r but negative R^2).

        Args:
            val_loader: Validation data loader
            method: 'temperature', 'isotonic', or 'both'

        Returns:
            Dictionary with calibration results (R^2 before/after)
        """
        self.eval()
        device = next(self.parameters()).device

        # Collect all predictions and targets
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                midi_tokens = batch['midi_tokens'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['scores'].to(device)

                outputs = self(midi_tokens, attention_mask)
                all_preds.append(outputs['predictions'])
                all_targets.append(targets)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Convert to numpy for metrics
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        results = {}

        # Baseline metrics
        results['baseline_r2'] = self._compute_mean_r2(preds_np, targets_np)
        results['baseline_mse'] = np.mean((preds_np - targets_np) ** 2)

        # Fit isotonic regression
        if method in ['isotonic', 'both']:
            self.isotonic_calibrator = IsotonicCalibrator(num_dims=self.num_dimensions)
            mse_before, mse_after = self.isotonic_calibrator.fit(preds_np, targets_np)
            iso_preds = self.isotonic_calibrator.calibrate(preds_np)
            results['isotonic_r2'] = self._compute_mean_r2(iso_preds, targets_np)
            results['isotonic_mse'] = mse_after

        # Fit temperature scaling
        if method in ['temperature', 'both']:
            self.temperature_scaling = TemperatureScaling().to(device)
            # Approximate logits from sigmoid outputs
            eps = 1e-7
            approx_logits = torch.log(preds.clamp(eps, 1-eps) / (1 - preds.clamp(eps, 1-eps)))
            ts_mse = self.temperature_scaling.fit(approx_logits, targets)
            ts_preds = self.temperature_scaling(approx_logits).detach().cpu().numpy()
            results['temperature_r2'] = self._compute_mean_r2(ts_preds, targets_np)
            results['temperature_mse'] = ts_mse
            results['temperature_value'] = self.temperature_scaling.get_temperature()

        # Select best method
        if method == 'both':
            if results.get('isotonic_r2', -float('inf')) > results.get('temperature_r2', -float('inf')):
                self.calibration_method = 'isotonic'
            else:
                self.calibration_method = 'temperature'
            results['selected_method'] = self.calibration_method
        elif method == 'isotonic':
            self.calibration_method = 'isotonic'
        else:
            self.calibration_method = 'temperature'

        return results

    def predict_calibrated(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get calibrated predictions.

        Args:
            midi_tokens: [batch, seq_len, 8] OctupleMIDI tokens
            attention_mask: Optional attention mask

        Returns:
            Calibrated predictions [batch, num_dims] in [0, 1]

        Raises:
            RuntimeError: If calibrate() has not been called
        """
        if self.calibration_method is None:
            raise RuntimeError("Model has not been calibrated. Call calibrate() first.")

        self.eval()
        with torch.no_grad():
            outputs = self(midi_tokens, attention_mask)
            preds = outputs['predictions']

            if self.calibration_method == 'isotonic':
                return self.isotonic_calibrator.calibrate_torch(preds)
            else:
                # Temperature scaling
                eps = 1e-7
                logits = torch.log(preds.clamp(eps, 1-eps) / (1 - preds.clamp(eps, 1-eps)))
                return self.temperature_scaling(logits)

    def _compute_mean_r2(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean R^2 across all dimensions."""
        r2_scores = []
        for dim in range(self.num_dimensions):
            ss_res = np.sum((targets[:, dim] - preds[:, dim]) ** 2)
            ss_tot = np.sum((targets[:, dim] - np.mean(targets[:, dim])) ** 2) + 1e-8
            r2 = 1 - ss_res / ss_tot
            r2_scores.append(r2)
        return np.mean(r2_scores)
