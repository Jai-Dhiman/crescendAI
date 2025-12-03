"""
MIDI-only Lightning Module for piano performance evaluation.

Simplified architecture for hackathon:
- MIDI Encoder (MIDIBert-style)
- BiLSTM Aggregation
- Multi-task Head with uncertainty weighting
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import numpy as np

from .midi_encoder import MIDIBertEncoder
from .aggregation import HierarchicalAggregator
from .mtl_head import MultiTaskHead


# Default dimensions for PercePiano
PERCEPIANO_DIMENSIONS = [
    "timing_stability",
    "note_accuracy",
    "dynamic_range",
    "articulation",
    "pedal_technique",
    "expression",
    "tone_quality",
    "overall",
]


class MIDIOnlyModule(pl.LightningModule):
    """
    PyTorch Lightning module for MIDI-only piano performance evaluation.

    Architecture:
        MIDI tokens -> MIDIBert Encoder -> BiLSTM Aggregation -> MTL Head -> 8 Scores

    Uses uncertainty-weighted loss for automatic task balancing.
    """

    def __init__(
        self,
        # Encoder params
        midi_hidden_dim: int = 256,
        midi_num_layers: int = 6,
        midi_num_heads: int = 8,
        max_seq_length: int = 1024,
        # Aggregation params
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        # MTL head params
        shared_hidden: int = 256,
        task_hidden: int = 128,
        # Training params
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        dropout: float = 0.1,
        # Task params
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions or PERCEPIANO_DIMENSIONS
        self.num_dimensions = len(self.dimensions)

        # MIDI Encoder
        self.midi_encoder = MIDIBertEncoder(
            hidden_size=midi_hidden_dim,
            num_layers=midi_num_layers,
            num_heads=midi_num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        # Aggregation: BiLSTM + Attention
        self.aggregator = HierarchicalAggregator(
            input_dim=midi_hidden_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            output_dim=shared_hidden,
        )

        # Multi-task head with uncertainty
        self.mtl_head = MultiTaskHead(
            input_dim=shared_hidden,
            shared_hidden=shared_hidden,
            task_hidden=task_hidden,
            dimensions=self.dimensions,
            dropout=dropout,
        )

        # Learnable log-variance for uncertainty-weighted loss
        # Initialize to 0 (sigma=1 for all tasks)
        self.log_vars = nn.Parameter(torch.zeros(self.num_dimensions))

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
        Compute uncertainty-weighted multi-task loss.

        L = sum_i [ (1/2) * exp(-log_var_i) * MSE_i + (1/2) * log_var_i ]

        Args:
            predictions: [batch, num_dims] predicted scores
            targets: [batch, num_dims] target scores
            log_vars: [num_dims] learnable log-variance per task

        Returns:
            Tuple of (total_loss, per_dimension_losses)
        """
        per_dim_losses = {}
        total_loss = 0.0

        for i, dim in enumerate(self.dimensions):
            # MSE for this dimension
            mse = torch.nn.functional.mse_loss(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            per_dim_losses[dim] = mse

            # Uncertainty-weighted loss
            precision = torch.exp(-log_vars[i])
            weighted_loss = 0.5 * precision * mse + 0.5 * log_vars[i]
            total_loss = total_loss + weighted_loss

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

        # Log uncertainty (sigma = exp(log_var/2))
        sigmas = torch.exp(0.5 * outputs["log_vars"]).detach()
        for i, dim in enumerate(self.dimensions):
            self.log(f"train/sigma_{dim}", sigmas[i])

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
            print("[DEBUG] validation_step_outputs is empty!")
            return

        # Collect all predictions and targets
        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        print(f"[DEBUG] Validation samples: {len(all_preds)}")
        print(f"[DEBUG] Preds has NaN: {torch.isnan(all_preds).any()}")
        print(f"[DEBUG] Targets has NaN: {torch.isnan(all_targets).any()}")

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
        print(f"[DEBUG] r_values: {r_values}")
        print(f"[DEBUG] mean_r: {mean_r}")
        self.log("val/mean_r", mean_r, prog_bar=True)

        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Get total steps
        total_steps = self.trainer.estimated_stepping_batches

        # OneCycleLR requires at least 2 steps for warmup, fall back to constant LR
        if total_steps < 10:
            return optimizer

        # Linear warmup + cosine decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def predict_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Inference step for prediction."""
        outputs = self(batch["midi_tokens"], batch.get("attention_mask"))

        # Convert predictions to 0-100 scale (clamped)
        predictions = torch.clamp(outputs["predictions"], 0, 100)

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
            predictions = torch.clamp(outputs["predictions"], 0, 100)
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
