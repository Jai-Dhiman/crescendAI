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
        self.shared_extractor = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for dim_name in dimensions:
            self.task_heads[dim_name] = nn.Sequential(
                nn.Linear(shared_hidden, task_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden, 1),
                nn.Sigmoid(),  # Output 0-1, will scale to 0-100
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
        scores = torch.cat(predictions, dim=1)

        # Scale from [0, 1] to [0, 100]
        scores = scores * 100.0

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


if __name__ == "__main__":
    print("Multi-task learning head module loaded successfully")
    print("- 10 dimensions (6 technical + 4 interpretive)")
    print("- Shared feature extractor + task-specific heads")
    print("- Learnable uncertainty parameters")
    print("- Output range: 0-100")
