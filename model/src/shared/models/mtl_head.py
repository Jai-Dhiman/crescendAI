import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class MultiTaskHead(nn.Module):
    """
    Multi-task learning head for piano performance evaluation.

    Predicts 8-10 dimensions with uncertainty estimates:

    Technical (6):
    - note_accuracy
    - rhythmic_precision
    - dynamics_control
    - articulation_quality
    - pedaling_technique
    - tone_quality

    Interpretive (4):
    - phrasing
    - musicality
    - overall_quality
    - expressiveness

    Uses uncertainty-weighted loss (Kendall & Gal) with learnable σ_i per dimension.
    """

    def __init__(
        self,
        input_dim: int = 512,
        shared_hidden: int = 256,
        task_hidden: int = 128,
        dimensions: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-task head.

        Args:
            input_dim: Input feature dimension (from aggregator)
            shared_hidden: Shared feature extractor dimension
            task_hidden: Task-specific hidden dimension
            dimensions: List of dimension names (default: 10 dimensions)
            dropout: Dropout probability
        """
        super().__init__()

        if dimensions is None:
            # Default: 10 dimensions (6 technical + 4 interpretive)
            dimensions = [
                'note_accuracy',
                'rhythmic_precision',
                'dynamics_control',
                'articulation_quality',
                'pedaling_technique',
                'tone_quality',
                'phrasing',
                'musicality',
                'overall_quality',
                'expressiveness',
            ]

        self.dimensions = dimensions
        self.num_dimensions = len(dimensions)
        self.input_dim = input_dim

        # Shared feature extractor (all tasks benefit from this)
        # Expanded to add more capacity for learning complex patterns
        self.shared_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # Maintain dimension first
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, shared_hidden),  # Then compress
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        # Simplified to 2 layers for better generalization with noisy labels
        # Deeper heads (4 layers) were overparameterized for weak supervision
        # NOTE: No sigmoid here - sigmoid applied in MIDIOnlyModule.forward()
        self.task_heads = nn.ModuleDict()
        for dim_name in dimensions:
            self.task_heads[dim_name] = nn.Sequential(
                nn.Linear(shared_hidden, task_hidden),  # Layer 1: compress
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden, 1),              # Layer 2: raw logits
            )

        # Learnable uncertainty parameters (log variance) for each dimension
        # Initialize to small values (log(1.0) = 0)
        self.log_vars = nn.Parameter(
            torch.zeros(self.num_dimensions)
        )

    def forward(
        self,
        features: torch.Tensor,
        return_uncertainties: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-task head.

        Args:
            features: Aggregated features [batch, input_dim]
            return_uncertainties: Whether to return uncertainty estimates

        Returns:
            Tuple of:
                - scores: Predicted scores [batch, num_dimensions] (0-100 scale)
                - uncertainties: Uncertainty values [num_dimensions] if requested
        """
        batch_size = features.shape[0]

        # Shared feature extraction
        shared_features = self.shared_extractor(features)

        # Task-specific predictions
        predictions = []
        for dim_name in self.dimensions:
            pred = self.task_heads[dim_name](shared_features)  # [batch, 1]
            predictions.append(pred)

        # Stack predictions: [batch, num_dimensions]
        # Returns raw logits - sigmoid applied in MIDIOnlyModule.forward()
        scores = torch.cat(predictions, dim=1)

        if return_uncertainties:
            # Convert log variance to standard deviation
            uncertainties = torch.exp(0.5 * self.log_vars)
            return scores, uncertainties
        else:
            return scores, None

    def get_dimension_names(self) -> List[str]:
        """Get list of dimension names."""
        return self.dimensions

    def get_uncertainties(self) -> torch.Tensor:
        """
        Get current uncertainty estimates for all dimensions.

        Returns:
            Uncertainties (standard deviations) [num_dimensions]
        """
        return torch.exp(0.5 * self.log_vars)


class PercePianoHead(nn.Module):
    """
    Simple 2-layer classifier matching PercePiano exactly.

    This is a simplified classification head without uncertainty estimation,
    designed to match the PercePiano reference implementation.

    Args:
        input_dim: Input dimension from aggregator (typically r * encoder_dim)
        num_dims: Number of output dimensions (19 for PercePiano)
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        input_dim: int = 3072,  # 4 * 768 = r * encoder_dim
        num_dims: int = 19,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_dims = num_dims

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_dims),
        )

    def forward(
        self,
        features: torch.Tensor,
        return_uncertainties: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through classifier.

        Args:
            features: Aggregated features [batch, input_dim]
            return_uncertainties: Ignored (for API compatibility)

        Returns:
            Tuple of:
                - scores: Predicted scores [batch, num_dims] (raw logits)
                - uncertainties: None (not supported)
        """
        scores = self.classifier(features)
        return scores, None

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.num_dims


if __name__ == "__main__":
    print("Multi-task learning head module loaded successfully")
    print("- 19 dimensions (matching PercePiano)")
    print("- 2-layer shared feature extractor")
    print("- 2-layer task-specific heads (raw logits output)")
    print("- Sigmoid applied in MIDIOnlyModule, not here")
    print("- Learnable uncertainty parameters")
