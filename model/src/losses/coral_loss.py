"""
CORAL (Consistent Rank Logits) loss for ordinal regression.

Converts continuous 0-100 scale regression to ordinal classification,
which provides rank-consistent predictions and better handling of
narrow label distributions.

Reference: "Rank Consistent Ordinal Regression for Neural Networks"
(Cao et al., Pattern Recognition 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) for ordinal regression.

    Converts continuous 0-100 scale to K ordinal thresholds:
    - Discretizes targets into K bins (e.g., 20 bins = 5-point resolution)
    - Uses K-1 binary classifiers with shared weights
    - Guarantees rank monotonicity (P(Y > k) <= P(Y > k-1))

    This is particularly useful when:
    1. Labels are concentrated in a narrow range (e.g., 80-95 for virtuoso performances)
    2. You need rank-consistent predictions (no predictions of 90 > 85 > 92)
    3. The ordinal nature of the target is important

    Architecture:
        logit_k = w^T * x + b_k  (shared w, per-threshold b)
        P(Y > k) = sigmoid(logit_k)
        Expected value = sum_k P(Y > k) * bin_width
    """

    def __init__(
        self,
        num_classes: int = 20,
        input_dim: int = 512,
        label_range: Tuple[float, float] = (0.0, 100.0),
        importance_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize CORAL loss.

        Args:
            num_classes: Number of ordinal classes (bins).
                         20 classes = 5-point resolution on 0-100 scale
            input_dim: Dimension of input features
            label_range: (min, max) range of continuous labels
            importance_weights: Optional per-threshold importance weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.label_min, self.label_max = label_range
        self.bin_width = (self.label_max - self.label_min) / num_classes

        # Shared weights across all thresholds (rank consistency)
        self.fc = nn.Linear(input_dim, 1, bias=False)

        # Per-threshold biases (K-1 thresholds for K classes)
        # Initialize biases to encourage uniform initial predictions
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        self._init_biases()

        # Optional importance weights
        if importance_weights is not None:
            self.register_buffer('importance_weights', importance_weights)
        else:
            self.register_buffer('importance_weights', None)

    def _init_biases(self) -> None:
        """Initialize biases to span the threshold range evenly."""
        # Set biases so that initial predictions span 0 to num_classes-1
        # This helps with faster convergence
        with torch.no_grad():
            # Spread biases from positive to negative
            # Higher threshold -> lower probability of exceeding
            self.bias.data = torch.linspace(
                2.0, -2.0, self.num_classes - 1
            )

    def _to_ordinal_labels(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert continuous targets to ordinal encoding.

        For target value y discretized to class c (0-indexed):
        ordinal_labels[k] = 1 if c > k else 0

        Example for 5 classes (target in class 2):
        [1, 1, 0, 0]  (exceeds thresholds 0 and 1, not 2, 3)

        Args:
            targets: Continuous targets [batch_size] or [batch_size, num_dims]

        Returns:
            Ordinal labels [batch_size, num_classes - 1]
        """
        # Handle multi-dimensional targets (use mean)
        if targets.dim() > 1:
            targets = targets.mean(dim=1)

        # Discretize to class indices
        normalized = (targets - self.label_min) / (self.label_max - self.label_min)
        class_indices = (normalized * self.num_classes).long()
        class_indices = class_indices.clamp(0, self.num_classes - 1)

        # Convert to ordinal encoding
        # ordinal[i, k] = 1 if class_indices[i] > k
        batch_size = targets.shape[0]
        ordinal_labels = torch.zeros(
            batch_size, self.num_classes - 1,
            device=targets.device, dtype=targets.dtype
        )

        for k in range(self.num_classes - 1):
            ordinal_labels[:, k] = (class_indices > k).float()

        return ordinal_labels

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CORAL loss.

        Args:
            features: Input features [batch_size, input_dim]
            targets: Continuous targets [batch_size] or [batch_size, num_dims]

        Returns:
            CORAL loss (scalar)
        """
        # Convert targets to ordinal encoding
        ordinal_labels = self._to_ordinal_labels(targets)

        # Compute shared logits + per-threshold biases
        # logits[i, k] = w^T * x_i + b_k
        shared_logits = self.fc(features)  # [batch_size, 1]
        logits = shared_logits + self.bias  # [batch_size, num_classes - 1]

        # Binary cross-entropy for each threshold
        # This is the CORAL loss: sum over all thresholds of BCE
        if self.importance_weights is not None:
            # Weighted BCE
            loss = F.binary_cross_entropy_with_logits(
                logits, ordinal_labels, reduction='none'
            )
            loss = loss * self.importance_weights.unsqueeze(0)
            loss = loss.mean()
        else:
            # Unweighted BCE
            loss = F.binary_cross_entropy_with_logits(
                logits, ordinal_labels
            )

        return loss

    def predict_probs(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        # Compute logits
        shared_logits = self.fc(features)
        logits = shared_logits + self.bias  # [batch_size, num_classes - 1]

        # P(Y > k) for each threshold
        cumulative_probs = torch.sigmoid(logits)  # [batch_size, num_classes - 1]

        # Convert cumulative to per-class probabilities
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        # With P(Y > -1) = 1 and P(Y > K-1) = 0
        batch_size = features.shape[0]
        class_probs = torch.zeros(
            batch_size, self.num_classes,
            device=features.device, dtype=features.dtype
        )

        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[:, 0] = 1 - cumulative_probs[:, 0]

        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]

        # P(Y = K-1) = P(Y > K-2)
        class_probs[:, -1] = cumulative_probs[:, -1]

        return class_probs

    def predict_continuous(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict continuous values from ordinal predictions.

        Uses expected value: E[Y] = sum_k P(Y = k) * bin_center_k

        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            Continuous predictions [batch_size]
        """
        class_probs = self.predict_probs(features)

        # Compute bin centers
        bin_centers = torch.linspace(
            self.label_min + self.bin_width / 2,
            self.label_max - self.bin_width / 2,
            self.num_classes,
            device=features.device, dtype=features.dtype
        )

        # Expected value
        predictions = (class_probs * bin_centers).sum(dim=1)

        return predictions

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict continuous values with uncertainty estimates.

        Returns both the expected value and variance of the prediction.

        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            Tuple of (predictions, uncertainty) both [batch_size]
        """
        class_probs = self.predict_probs(features)

        # Bin centers
        bin_centers = torch.linspace(
            self.label_min + self.bin_width / 2,
            self.label_max - self.bin_width / 2,
            self.num_classes,
            device=features.device, dtype=features.dtype
        )

        # Expected value
        predictions = (class_probs * bin_centers).sum(dim=1)

        # Variance (uncertainty)
        variance = (class_probs * (bin_centers - predictions.unsqueeze(1)) ** 2).sum(dim=1)
        uncertainty = variance.sqrt()

        return predictions, uncertainty


class CORALHead(nn.Module):
    """
    Multi-task CORAL head for multiple evaluation dimensions.

    Creates separate CORAL classifiers for each dimension while
    sharing the feature extraction backbone.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_dimensions: int = 8,
        num_classes: int = 20,
        label_range: Tuple[float, float] = (0.0, 100.0),
    ):
        """
        Initialize multi-task CORAL head.

        Args:
            input_dim: Dimension of input features
            num_dimensions: Number of evaluation dimensions
            num_classes: Number of ordinal classes per dimension
            label_range: (min, max) range of continuous labels
        """
        super().__init__()

        self.num_dimensions = num_dimensions
        self.num_classes = num_classes

        # Create CORAL classifier for each dimension
        self.coral_heads = nn.ModuleList([
            CORALLoss(
                num_classes=num_classes,
                input_dim=input_dim,
                label_range=label_range,
            )
            for _ in range(num_dimensions)
        ])

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total CORAL loss across all dimensions.

        Args:
            features: Input features [batch_size, input_dim]
            targets: Targets [batch_size, num_dimensions]

        Returns:
            Total CORAL loss (scalar)
        """
        total_loss = 0.0

        for dim_idx, coral_head in enumerate(self.coral_heads):
            dim_targets = targets[:, dim_idx]
            dim_loss = coral_head(features, dim_targets)
            total_loss = total_loss + dim_loss

        # Average across dimensions
        return total_loss / self.num_dimensions

    def predict(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict continuous values for all dimensions.

        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            Predictions [batch_size, num_dimensions]
        """
        predictions = []

        for coral_head in self.coral_heads:
            dim_pred = coral_head.predict_continuous(features)
            predictions.append(dim_pred)

        return torch.stack(predictions, dim=1)

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty for all dimensions.

        Args:
            features: Input features [batch_size, input_dim]

        Returns:
            Tuple of (predictions, uncertainties) both [batch_size, num_dimensions]
        """
        predictions = []
        uncertainties = []

        for coral_head in self.coral_heads:
            pred, unc = coral_head.predict_with_uncertainty(features)
            predictions.append(pred)
            uncertainties.append(unc)

        return torch.stack(predictions, dim=1), torch.stack(uncertainties, dim=1)


if __name__ == "__main__":
    print("CORAL (Consistent Rank Logits) for Ordinal Regression")
    print("- Converts regression to ordinal classification")
    print("- Guarantees rank monotonicity")
    print("- Better for narrow label distributions")
    print()

    # Example usage
    batch_size = 16
    input_dim = 512
    num_classes = 20  # 5-point resolution on 0-100 scale

    coral = CORALLoss(num_classes=num_classes, input_dim=input_dim)

    # Simulate features and targets
    features = torch.randn(batch_size, input_dim)
    targets = torch.rand(batch_size) * 30 + 70  # Concentrated in 70-100 range

    # Compute loss
    loss = coral(features, targets)
    print(f"CORAL Loss: {loss.item():.4f}")

    # Get predictions
    predictions = coral.predict_continuous(features)
    print(f"Predictions range: [{predictions.min().item():.1f}, {predictions.max().item():.1f}]")

    # Get predictions with uncertainty
    preds, uncertainty = coral.predict_with_uncertainty(features)
    print(f"Mean uncertainty: {uncertainty.mean().item():.2f}")

    # Multi-task head
    print("\nMulti-task CORAL Head:")
    coral_head = CORALHead(input_dim=input_dim, num_dimensions=8)
    multi_targets = torch.rand(batch_size, 8) * 30 + 70
    multi_loss = coral_head(features, multi_targets)
    print(f"Multi-task Loss: {multi_loss.item():.4f}")
