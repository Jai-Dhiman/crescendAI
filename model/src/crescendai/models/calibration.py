"""
Post-hoc calibration methods for improving R^2 score.

This module provides calibration techniques to fix systematic bias in model predictions
when Pearson r is positive but R^2 is negative (predictions are systematically shifted/scaled).

Implements:
- Temperature Scaling: Single learned temperature parameter
- Isotonic Regression: Per-dimension non-parametric calibration
- Calibration Wrapper: Combines both methods for inference
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.

    Learns a single temperature parameter that scales logits before sigmoid.
    This is the simplest calibration method and works well when predictions
    are uniformly biased across all dimensions.

    Usage:
        ts = TemperatureScaling()
        ts.fit(val_logits, val_targets)
        calibrated = ts(test_logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling and sigmoid.

        Args:
            logits: Raw model logits [batch, num_dims]

        Returns:
            Calibrated predictions [batch, num_dims] in [0, 1]
        """
        return torch.sigmoid(logits / self.temperature)

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Validation logits [N, num_dims]
            targets: Validation targets [N, num_dims]
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            Final MSE loss after calibration
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            preds = self.forward(logits)
            loss = nn.functional.mse_loss(preds, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        # Return final loss
        with torch.no_grad():
            final_preds = self.forward(logits)
            return nn.functional.mse_loss(final_preds, targets).item()

    def get_temperature(self) -> float:
        """Get learned temperature value."""
        return self.temperature.item()


class IsotonicCalibrator:
    """
    Isotonic regression calibration per dimension.

    Fits a separate monotonic function for each output dimension,
    allowing non-linear calibration that preserves ranking.
    More flexible than temperature scaling but requires more data.

    Usage:
        ic = IsotonicCalibrator(num_dims=19)
        ic.fit(val_preds, val_targets)
        calibrated = ic.calibrate(test_preds)
    """

    def __init__(self, num_dims: int = 19):
        self.num_dims = num_dims
        self.models: List[IsotonicRegression] = [
            IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            for _ in range(num_dims)
        ]
        self.fitted = False

    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Fit isotonic regression on validation set.

        Args:
            predictions: Model predictions [N, num_dims] in [0, 1]
            targets: Ground truth targets [N, num_dims] in [0, 1]

        Returns:
            Tuple of (mse_before, mse_after) for comparison
        """
        mse_before = np.mean((predictions - targets) ** 2)

        for dim in range(self.num_dims):
            self.models[dim].fit(predictions[:, dim], targets[:, dim])

        self.fitted = True

        # Compute MSE after calibration
        calibrated = self.calibrate(predictions)
        mse_after = np.mean((calibrated - targets) ** 2)

        return mse_before, mse_after

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to predictions.

        Args:
            predictions: Model predictions [N, num_dims] in [0, 1]

        Returns:
            Calibrated predictions [N, num_dims]

        Raises:
            RuntimeError: If calibrator has not been fitted
        """
        if not self.fitted:
            raise RuntimeError(
                "IsotonicCalibrator has not been fitted. Call fit() first."
            )

        calibrated = np.zeros_like(predictions)
        for dim in range(self.num_dims):
            calibrated[:, dim] = self.models[dim].predict(predictions[:, dim])

        return calibrated

    def calibrate_torch(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply isotonic calibration to torch tensor.

        Args:
            predictions: Model predictions [N, num_dims] in [0, 1]

        Returns:
            Calibrated predictions [N, num_dims] as torch tensor
        """
        device = predictions.device
        preds_np = predictions.detach().cpu().numpy()
        calibrated_np = self.calibrate(preds_np)
        return torch.from_numpy(calibrated_np).to(device)


class CalibrationWrapper:
    """
    Combines temperature scaling and isotonic regression for inference.

    Provides both calibration methods and tools to compare their effectiveness.
    Recommended workflow:
    1. Train model normally
    2. Collect validation predictions
    3. Fit both calibrators
    4. Compare R^2 scores to choose best method
    5. Apply chosen calibration at inference time

    Usage:
        wrapper = CalibrationWrapper(model)
        wrapper.fit(val_loader)
        calibrated_preds = wrapper.predict(test_batch, method='isotonic')
    """

    def __init__(
        self,
        model: nn.Module,
        num_dims: int = 19,
        device: str = "cpu",
    ):
        """
        Initialize calibration wrapper.

        Args:
            model: Trained model with forward() returning logits
            num_dims: Number of output dimensions
            device: Device for computation
        """
        self.model = model
        self.num_dims = num_dims
        self.device = device

        self.temperature_scaling = TemperatureScaling().to(device)
        self.isotonic_calibrator = IsotonicCalibrator(num_dims=num_dims)

        self.fitted = False

    def fit(self, val_loader: DataLoader) -> dict:
        """
        Fit calibration on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with calibration results and R^2 improvements
        """
        self.model.eval()

        # Collect all predictions and targets
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                midi_tokens = batch["midi_tokens"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["scores"].to(self.device)

                # Get model output (assumes model returns dict with 'predictions')
                output = self.model(midi_tokens, attention_mask)
                if isinstance(output, dict):
                    # Get raw logits before sigmoid
                    # This requires access to intermediate values
                    # For now, assume predictions are already sigmoided
                    preds = output["predictions"]
                else:
                    preds = output

                all_logits.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Convert to numpy for isotonic regression
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()

        results = {}

        # Compute baseline metrics
        results["baseline_mse"] = np.mean((preds_np - targets_np) ** 2)
        results["baseline_r2"] = self._compute_r2(preds_np, targets_np)

        # Fit isotonic calibrator
        mse_before, mse_after = self.isotonic_calibrator.fit(preds_np, targets_np)
        iso_preds = self.isotonic_calibrator.calibrate(preds_np)
        results["isotonic_mse"] = mse_after
        results["isotonic_r2"] = self._compute_r2(iso_preds, targets_np)

        # Note: Temperature scaling needs logits, not post-sigmoid predictions
        # If model only returns sigmoid outputs, we approximate by inverse sigmoid
        eps = 1e-7
        approx_logits = torch.log(
            all_preds.clamp(eps, 1 - eps) / (1 - all_preds.clamp(eps, 1 - eps))
        )
        ts_mse = self.temperature_scaling.fit(approx_logits, all_targets)
        ts_preds = self.temperature_scaling(approx_logits).detach().cpu().numpy()
        results["temperature_mse"] = ts_mse
        results["temperature_r2"] = self._compute_r2(ts_preds, targets_np)
        results["temperature_value"] = self.temperature_scaling.get_temperature()

        self.fitted = True

        return results

    def predict(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        method: str = "isotonic",
    ) -> torch.Tensor:
        """
        Get calibrated predictions.

        Args:
            midi_tokens: Input MIDI tokens [batch, seq_len, 8]
            attention_mask: Attention mask [batch, seq_len]
            method: Calibration method ('none', 'temperature', 'isotonic')

        Returns:
            Calibrated predictions [batch, num_dims]
        """
        if not self.fitted and method != "none":
            raise RuntimeError(
                "CalibrationWrapper has not been fitted. Call fit() first."
            )

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                midi_tokens.to(self.device), attention_mask.to(self.device)
            )
            if isinstance(output, dict):
                preds = output["predictions"]
            else:
                preds = output

            if method == "none":
                return preds
            elif method == "temperature":
                # Apply inverse sigmoid then temperature scaling
                eps = 1e-7
                logits = torch.log(
                    preds.clamp(eps, 1 - eps) / (1 - preds.clamp(eps, 1 - eps))
                )
                return self.temperature_scaling(logits)
            elif method == "isotonic":
                return self.isotonic_calibrator.calibrate_torch(preds)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

    def _compute_r2(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean R^2 across all dimensions."""
        r2_scores = []
        for dim in range(self.num_dims):
            ss_res = np.sum((targets[:, dim] - preds[:, dim]) ** 2)
            ss_tot = np.sum((targets[:, dim] - np.mean(targets[:, dim])) ** 2) + 1e-8
            r2 = 1 - ss_res / ss_tot
            r2_scores.append(r2)
        return np.mean(r2_scores)


if __name__ == "__main__":
    print("Calibration module loaded successfully")
    print("- TemperatureScaling: Single parameter scaling")
    print("- IsotonicCalibrator: Per-dimension monotonic calibration")
    print("- CalibrationWrapper: Combined calibration for inference")
