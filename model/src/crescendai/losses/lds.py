"""
Label Distribution Smoothing (LDS) for imbalanced regression.

Re-weights loss by inverse of smoothed label density to handle
narrow target distributions (e.g., all MAESTRO performances being virtuoso-level).

Reference: "Delving into Deep Imbalanced Regression" (Yang et al., ICML 2021)
"""

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


class LDSWeighting(nn.Module):
    """
    Label Distribution Smoothing (LDS) for sample re-weighting.

    Computes weights based on inverse of smoothed label density:
    1. Estimate label density via histogram
    2. Smooth with Gaussian/Laplacian kernel
    3. Compute weights as inverse of smoothed density
    4. Apply during loss computation to upweight rare labels

    This helps models learn from underrepresented regions of the
    label distribution (e.g., very high or very low quality scores).
    """

    def __init__(
        self,
        num_bins: int = 100,
        kernel_size: int = 5,
        kernel_type: Literal["gaussian", "laplacian", "triangular"] = "gaussian",
        sigma: float = 2.0,
        reweight_scale: float = 1.0,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        label_range: tuple = (0.0, 100.0),
    ):
        """
        Initialize LDS weighting.

        Args:
            num_bins: Number of bins for density estimation
            kernel_size: Size of smoothing kernel (must be odd)
            kernel_type: Type of kernel ('gaussian', 'laplacian', 'triangular')
            sigma: Standard deviation for Gaussian kernel
            reweight_scale: Scale factor for reweighting (1.0 = full reweight)
            min_weight: Minimum weight to prevent numerical issues
            max_weight: Maximum weight to prevent outlier domination
            label_range: (min, max) range of labels
        """
        super().__init__()

        self.num_bins = num_bins
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.reweight_scale = reweight_scale
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.label_min, self.label_max = label_range

        # Precompute kernel
        self.kernel = self._create_kernel()

        # Will be populated by fit()
        self.register_buffer("bin_weights", None)
        self._fitted = False

    def _create_kernel(self) -> np.ndarray:
        """Create smoothing kernel."""
        half = self.kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)

        if self.kernel_type == "gaussian":
            kernel = np.exp(-(x**2) / (2 * self.sigma**2))
        elif self.kernel_type == "laplacian":
            kernel = np.exp(-np.abs(x) / self.sigma)
        elif self.kernel_type == "triangular":
            kernel = np.maximum(0, 1 - np.abs(x) / (half + 1))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Normalize
        kernel = kernel / kernel.sum()
        return kernel

    def fit(self, labels: torch.Tensor) -> "LDSWeighting":
        """
        Fit LDS weights from training labels.

        Args:
            labels: Training labels [num_samples] or [num_samples, num_dims]
                    If multi-dimensional, uses mean across dimensions

        Returns:
            self (for method chaining)
        """
        # Flatten to 1D if needed
        if labels.dim() > 1:
            labels = labels.mean(dim=1)

        labels_np = labels.detach().cpu().numpy()

        # Compute histogram (density estimation)
        bin_edges = np.linspace(self.label_min, self.label_max, self.num_bins + 1)
        counts, _ = np.histogram(labels_np, bins=bin_edges)

        # Smooth counts with kernel
        smoothed_counts = gaussian_filter1d(counts.astype(np.float64), sigma=self.sigma)

        # Avoid division by zero
        smoothed_counts = np.maximum(smoothed_counts, 1e-8)

        # Compute weights (inverse of density)
        # Higher weight = rarer labels
        weights = 1.0 / smoothed_counts

        # Normalize weights to have mean 1
        weights = weights / weights.mean()

        # Apply reweight scale (interpolate between uniform and full reweight)
        weights = 1.0 + self.reweight_scale * (weights - 1.0)

        # Clip to prevent extreme values
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Convert to tensor
        self.bin_weights = torch.tensor(weights, dtype=torch.float32)
        self._fitted = True

        return self

    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get per-sample weights based on their label values.

        Args:
            labels: Labels to get weights for [batch_size] or [batch_size, num_dims]
                    If multi-dimensional, uses mean across dimensions

        Returns:
            Weights tensor [batch_size]
        """
        if not self._fitted:
            raise RuntimeError(
                "LDSWeighting must be fit() before calling get_weights()"
            )

        # Handle multi-dimensional labels
        if labels.dim() > 1:
            labels = labels.mean(dim=1)

        # Map labels to bin indices
        bin_indices = (
            (labels - self.label_min)
            / (self.label_max - self.label_min)
            * self.num_bins
        )
        bin_indices = bin_indices.long().clamp(0, self.num_bins - 1)

        # Look up weights
        weights = self.bin_weights.to(labels.device)[bin_indices]

        return weights

    def forward(
        self,
        loss: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LDS weighting to a per-sample loss.

        Args:
            loss: Per-sample loss [batch_size] or [batch_size, num_dims]
            labels: Labels for weighting [batch_size] or [batch_size, num_dims]

        Returns:
            Weighted loss (reduced to scalar if input was per-sample)
        """
        weights = self.get_weights(labels)

        # Expand weights if loss is multi-dimensional
        if loss.dim() > 1:
            weights = weights.unsqueeze(1).expand_as(loss)

        # Apply weights
        weighted_loss = loss * weights

        return weighted_loss.mean()

    def get_density_stats(self) -> dict:
        """Get statistics about the fitted density for debugging."""
        if not self._fitted:
            return {"fitted": False}

        weights = self.bin_weights.numpy()
        return {
            "fitted": True,
            "num_bins": self.num_bins,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "weight_std": float(weights.std()),
        }


class FDSFeatureSmoothing(nn.Module):
    """
    Feature Distribution Smoothing (FDS) for imbalanced regression.

    Calibrates feature statistics across nearby target bins to improve
    predictions in underrepresented regions. FDS works by:
    1. Tracking per-bin feature statistics (mean, variance) during training
    2. Smoothing statistics across nearby bins using a Gaussian kernel
    3. Calibrating features by standardizing with bin stats and restandardizing
       with smoothed global stats

    This helps the model generalize better to underrepresented target regions
    by ensuring feature distributions are consistent across all target values.

    Reference: "Delving into Deep Imbalanced Regression" (Yang et al., ICML 2021)
    """

    def __init__(
        self,
        feature_dim: int,
        num_bins: int = 100,
        start_update_epoch: int = 0,
        momentum: float = 0.9,
        kernel_sigma: float = 2.0,
        label_range: tuple = (0.0, 100.0),
    ):
        """
        Initialize FDS.

        Args:
            feature_dim: Dimension of features to smooth
            num_bins: Number of bins for target discretization
            start_update_epoch: Epoch to start updating statistics
            momentum: Momentum for running statistics (0.9 = 90% old, 10% new)
            kernel_sigma: Sigma for Gaussian smoothing kernel
            label_range: (min, max) range of target values
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.start_update_epoch = start_update_epoch
        self.momentum = momentum
        self.kernel_sigma = kernel_sigma
        self.label_min, self.label_max = label_range
        self.eps = 1e-6  # For numerical stability

        # Running statistics per bin
        self.register_buffer("running_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("running_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("bin_counts", torch.zeros(num_bins))

        # Smoothed statistics (computed from running stats)
        self.register_buffer("smoothed_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("smoothed_var", torch.ones(num_bins, feature_dim))

        # Precompute smoothing kernel
        self._precompute_kernel()

    def _precompute_kernel(self) -> None:
        """Precompute Gaussian smoothing kernel for bin statistics."""
        # Create kernel that spans all bins
        kernel_size = self.num_bins
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.float32) - center
        kernel = torch.exp(-(x**2) / (2 * self.kernel_sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        self.register_buffer("smoothing_kernel", kernel)

    def _get_bin_indices(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Map target values to bin indices.

        Args:
            targets: Target values [batch_size] or [batch_size, num_dims]

        Returns:
            Bin indices [batch_size]
        """
        # Handle multi-dimensional targets (use mean)
        if targets.dim() > 1:
            targets = targets.mean(dim=1)

        # Map to bin indices
        normalized = (targets - self.label_min) / (self.label_max - self.label_min)
        bin_indices = (normalized * self.num_bins).long()
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)

        return bin_indices

    def update_statistics(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        Update running statistics during training.

        Call this during forward pass in training mode to track
        per-bin feature statistics.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            targets: Target values [batch_size] or [batch_size, num_dims]
            epoch: Current training epoch
        """
        if epoch < self.start_update_epoch:
            return

        # Get bin indices for each sample
        bin_indices = self._get_bin_indices(targets)

        # Update statistics for each bin present in this batch
        for bin_idx in range(self.num_bins):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue

            # Get features for this bin
            bin_features = features[mask]  # [num_in_bin, feature_dim]

            # Compute batch statistics
            batch_mean = bin_features.mean(dim=0)
            batch_var = bin_features.var(dim=0, unbiased=False) + self.eps

            # Update running statistics with momentum
            if self.bin_counts[bin_idx] == 0:
                # First update for this bin - initialize directly
                self.running_mean[bin_idx] = batch_mean.detach()
                self.running_var[bin_idx] = batch_var.detach()
            else:
                # Exponential moving average
                self.running_mean[bin_idx] = (
                    self.momentum * self.running_mean[bin_idx]
                    + (1 - self.momentum) * batch_mean.detach()
                )
                self.running_var[bin_idx] = (
                    self.momentum * self.running_var[bin_idx]
                    + (1 - self.momentum) * batch_var.detach()
                )

            self.bin_counts[bin_idx] += mask.sum().float()

        # Update smoothed statistics
        self._update_smoothed_statistics()

    def _update_smoothed_statistics(self) -> None:
        """Smooth per-bin statistics using Gaussian kernel."""
        # Only smooth bins that have been observed
        observed_mask = self.bin_counts > 0

        if observed_mask.sum() < 2:
            # Not enough data to smooth
            self.smoothed_mean.copy_(self.running_mean)
            self.smoothed_var.copy_(self.running_var)
            return

        # Apply Gaussian smoothing to each feature dimension
        # Use 1D convolution with the precomputed kernel
        for dim_idx in range(self.feature_dim):
            means = self.running_mean[:, dim_idx]
            vars = self.running_var[:, dim_idx]

            # Pad for convolution
            pad_size = self.num_bins // 2
            means_padded = torch.nn.functional.pad(
                means.unsqueeze(0).unsqueeze(0), (pad_size, pad_size), mode="replicate"
            )
            vars_padded = torch.nn.functional.pad(
                vars.unsqueeze(0).unsqueeze(0), (pad_size, pad_size), mode="replicate"
            )

            # Apply kernel
            kernel = self.smoothing_kernel.view(1, 1, -1)
            smoothed_means = torch.nn.functional.conv1d(
                means_padded, kernel, padding=0
            ).squeeze()
            smoothed_vars = torch.nn.functional.conv1d(
                vars_padded, kernel, padding=0
            ).squeeze()

            # Handle edge cases where conv output size might differ
            if smoothed_means.shape[0] > self.num_bins:
                smoothed_means = smoothed_means[: self.num_bins]
                smoothed_vars = smoothed_vars[: self.num_bins]
            elif smoothed_means.shape[0] < self.num_bins:
                # Pad if needed
                diff = self.num_bins - smoothed_means.shape[0]
                smoothed_means = torch.nn.functional.pad(
                    smoothed_means, (0, diff), value=smoothed_means[-1]
                )
                smoothed_vars = torch.nn.functional.pad(
                    smoothed_vars, (0, diff), value=smoothed_vars[-1]
                )

            self.smoothed_mean[:, dim_idx] = smoothed_means
            self.smoothed_var[:, dim_idx] = smoothed_vars.clamp(min=self.eps)

    def smooth_features(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth features based on target values.

        Calibrates features by:
        1. Standardizing with the bin-specific mean/var
        2. Restandardizing with the smoothed mean/var

        This ensures features have consistent distributions across
        all target values, helping generalization to rare targets.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            targets: Target values [batch_size] or [batch_size, num_dims]

        Returns:
            Smoothed features [batch_size, feature_dim]
        """
        # Check if we have enough statistics
        if self.bin_counts.sum() < self.num_bins * 0.1:
            # Not enough data collected yet, return unchanged
            return features

        # Get bin indices
        bin_indices = self._get_bin_indices(targets)

        # Get per-sample statistics based on their target bin
        sample_mean = self.running_mean[bin_indices]  # [batch, feature_dim]
        sample_var = self.running_var[bin_indices]
        smoothed_sample_mean = self.smoothed_mean[bin_indices]
        smoothed_sample_var = self.smoothed_var[bin_indices]

        # Standardize with bin statistics
        standardized = (features - sample_mean) / (sample_var.sqrt() + self.eps)

        # Restandardize with smoothed statistics
        smoothed = (
            standardized * (smoothed_sample_var.sqrt() + self.eps)
            + smoothed_sample_mean
        )

        return smoothed

    def get_statistics_summary(self) -> dict:
        """Get summary of collected statistics for debugging."""
        observed_bins = (self.bin_counts > 0).sum().item()
        return {
            "observed_bins": observed_bins,
            "total_bins": self.num_bins,
            "coverage": observed_bins / self.num_bins,
            "total_samples": self.bin_counts.sum().item(),
            "mean_range": (
                self.running_mean.min().item(),
                self.running_mean.max().item(),
            ),
            "var_range": (self.running_var.min().item(), self.running_var.max().item()),
        }


if __name__ == "__main__":
    print("Label Distribution Smoothing (LDS)")
    print("- Reweights samples by inverse label density")
    print("- Helps learn from underrepresented label regions")
    print("- Supports Gaussian/Laplacian/Triangular kernels")

    # Example usage
    lds = LDSWeighting(num_bins=100, kernel_size=5, sigma=2.0)

    # Simulate narrow distribution (most samples around 85-95)
    labels = torch.randn(1000) * 5 + 90  # Mean 90, std 5
    labels = labels.clamp(0, 100)

    lds.fit(labels)
    print(f"\nFitted LDS: {lds.get_density_stats()}")

    # Get weights for some test labels
    test_labels = torch.tensor([50.0, 70.0, 90.0, 95.0])
    weights = lds.get_weights(test_labels)
    print(f"Weights for labels {test_labels.tolist()}: {weights.tolist()}")
