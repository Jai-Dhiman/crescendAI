"""
Weak supervision label aggregation for piano performance evaluation.

Combines outputs from multiple imperfect labeling functions to create
consensus labels. Implements weighted aggregation and confidence estimation.

Reference: "Snorkel: rapid training data creation with weak supervision"
(Ratner et al., 2017)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from collections import defaultdict


class WeakSupervisionAggregator:
    """
    Aggregates labels from multiple labeling functions.

    Supports multiple aggregation strategies:
    - Weighted average (default): Combine labels weighted by function confidence
    - Majority vote: Use most common label (for discrete labels)
    - Snorkel-style: Learn latent accuracy parameters
    """

    def __init__(
        self,
        aggregation_method: str = "weighted_average",
        min_coverage: int = 1,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize weak supervision aggregator.

        Args:
            aggregation_method: Aggregation strategy
                - "weighted_average": Weighted mean of labels
                - "median": Robust median aggregation
                - "trimmed_mean": Remove outliers then average
            min_coverage: Minimum number of labeling functions that must vote
            confidence_threshold: Minimum confidence score to include a label
        """
        self.aggregation_method = aggregation_method
        self.min_coverage = min_coverage
        self.confidence_threshold = confidence_threshold

        # Track labeling function statistics for auto-weighting
        self.lf_stats = defaultdict(lambda: {
            'count': 0,
            'mean': 0.0,
            'variance': 0.0,
        })

    def aggregate_labels(
        self,
        label_matrix: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Aggregate labels from multiple labeling functions.

        Args:
            label_matrix: Dictionary mapping dimension names to lists of (label, weight) tuples
                         Each tuple is output from one labeling function

        Returns:
            Dictionary mapping dimension names to (aggregated_label, confidence) tuples
        """
        aggregated = {}

        for dimension, labels_weights in label_matrix.items():
            if not labels_weights:
                # No labels for this dimension
                aggregated[dimension] = (None, 0.0)
                continue

            # Filter out None labels and apply confidence threshold
            valid_labels = [
                (label, weight) for label, weight in labels_weights
                if label is not None and weight >= self.confidence_threshold
            ]

            if len(valid_labels) < self.min_coverage:
                # Insufficient coverage
                aggregated[dimension] = (None, 0.0)
                continue

            # Apply aggregation method
            if self.aggregation_method == "weighted_average":
                agg_label, confidence = self._weighted_average(valid_labels)
            elif self.aggregation_method == "median":
                agg_label, confidence = self._median_aggregation(valid_labels)
            elif self.aggregation_method == "trimmed_mean":
                agg_label, confidence = self._trimmed_mean(valid_labels)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

            aggregated[dimension] = (agg_label, confidence)

        return aggregated

    def _weighted_average(
        self,
        labels_weights: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Compute weighted average of labels.

        Args:
            labels_weights: List of (label, weight) tuples

        Returns:
            (aggregated_label, confidence)
        """
        labels = np.array([lw[0] for lw in labels_weights])
        weights = np.array([lw[1] for lw in labels_weights])

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted average
        agg_label = np.sum(labels * weights)

        # Confidence = sum of weights (higher when more functions agree)
        confidence = np.sum(weights)

        # Adjust confidence by agreement (variance)
        if len(labels) > 1:
            variance = np.average((labels - agg_label) ** 2, weights=weights)
            # Higher variance = lower confidence
            # Scale by typical variance (10 points on 0-100 scale)
            confidence *= np.exp(-variance / 100)

        return float(agg_label), float(confidence)

    def _median_aggregation(
        self,
        labels_weights: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Compute weighted median (robust to outliers).

        Args:
            labels_weights: List of (label, weight) tuples

        Returns:
            (aggregated_label, confidence)
        """
        labels = np.array([lw[0] for lw in labels_weights])
        weights = np.array([lw[1] for lw in labels_weights])

        # Weighted median
        sorted_idx = np.argsort(labels)
        sorted_labels = labels[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)

        agg_label = sorted_labels[median_idx]

        # Confidence = sum of weights within ±10 of median
        close_mask = np.abs(labels - agg_label) <= 10
        confidence = np.sum(weights[close_mask]) / np.sum(weights)

        return float(agg_label), float(confidence)

    def _trimmed_mean(
        self,
        labels_weights: List[Tuple[float, float]],
        trim_fraction: float = 0.2
    ) -> Tuple[float, float]:
        """
        Compute trimmed mean (remove outliers then average).

        Args:
            labels_weights: List of (label, weight) tuples
            trim_fraction: Fraction of extreme values to remove

        Returns:
            (aggregated_label, confidence)
        """
        labels = np.array([lw[0] for lw in labels_weights])
        weights = np.array([lw[1] for lw in labels_weights])

        if len(labels) < 4:
            # Too few labels to trim
            return self._weighted_average(labels_weights)

        # Remove extreme values
        sorted_idx = np.argsort(labels)
        n_trim = int(len(labels) * trim_fraction)

        if n_trim > 0:
            # Trim both ends
            keep_idx = sorted_idx[n_trim:-n_trim]
            labels = labels[keep_idx]
            weights = weights[keep_idx]

        # Weighted average of remaining
        weights = weights / weights.sum()
        agg_label = np.sum(labels * weights)

        # Confidence = fraction of labels retained
        confidence = len(labels) / len(labels_weights)

        return float(agg_label), float(confidence)


class AdaptiveWeightLearner:
    """
    Learn optimal weights for labeling functions based on agreement patterns.

    Implements simplified version of Snorkel's generative model:
    - Assumes labeling functions have unknown accuracies
    - Learns weights by maximizing agreement with consensus
    - Iteratively refines weights and consensus labels
    """

    def __init__(self, num_iterations: int = 10, learning_rate: float = 0.1):
        """
        Initialize adaptive weight learner.

        Args:
            num_iterations: Number of EM-style iterations
            learning_rate: Step size for weight updates
        """
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.learned_weights = {}

    def fit(
        self,
        dimension: str,
        label_matrix: np.ndarray,
        initial_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Learn optimal weights for labeling functions.

        Args:
            dimension: Dimension name (for tracking)
            label_matrix: Matrix of labels [num_samples, num_functions]
                         NaN for abstaining functions
            initial_weights: Initial weights [num_functions]

        Returns:
            Learned weights [num_functions]
        """
        num_samples, num_functions = label_matrix.shape

        # Initialize weights
        if initial_weights is None:
            weights = np.ones(num_functions) / num_functions
        else:
            weights = initial_weights.copy()

        # EM-style iterations
        for iteration in range(self.num_iterations):
            # E-step: Compute consensus labels using current weights
            consensus = self._compute_consensus(label_matrix, weights)

            # M-step: Update weights based on agreement with consensus
            weights = self._update_weights(label_matrix, consensus, weights)

        self.learned_weights[dimension] = weights
        return weights

    def _compute_consensus(
        self,
        label_matrix: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute consensus labels using weighted average.

        Args:
            label_matrix: Labels [num_samples, num_functions]
            weights: Weights [num_functions]

        Returns:
            Consensus labels [num_samples]
        """
        # Mask for valid (non-NaN) labels
        valid_mask = ~np.isnan(label_matrix)

        # Weighted average (ignoring NaN)
        weighted_sum = np.nansum(label_matrix * weights, axis=1)
        weight_sum = np.sum(valid_mask * weights, axis=1)

        # Avoid division by zero
        consensus = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

        return consensus

    def _update_weights(
        self,
        label_matrix: np.ndarray,
        consensus: np.ndarray,
        current_weights: np.ndarray
    ) -> np.ndarray:
        """
        Update weights based on agreement with consensus.

        Args:
            label_matrix: Labels [num_samples, num_functions]
            consensus: Consensus labels [num_samples]
            current_weights: Current weights [num_functions]

        Returns:
            Updated weights [num_functions]
        """
        num_samples, num_functions = label_matrix.shape

        # Compute error for each labeling function
        errors = np.abs(label_matrix - consensus[:, None])  # [num_samples, num_functions]

        # Mean error per function (ignoring NaN)
        mean_errors = np.nanmean(errors, axis=0)  # [num_functions]

        # Convert errors to accuracies (inverse relationship)
        # Use exponential to emphasize low-error functions
        accuracies = np.exp(-mean_errors / 10)  # Scale factor of 10 for 0-100 labels

        # Update weights with learning rate
        new_weights = current_weights * (1 - self.learning_rate) + accuracies * self.learning_rate

        # Normalize
        new_weights = new_weights / new_weights.sum()

        return new_weights


def apply_labeling_functions(
    sample_data: Dict,
    labeling_functions: Dict[str, List],
    use_adaptive_weights: bool = False,
    weight_learner: Optional[AdaptiveWeightLearner] = None
) -> Dict[str, float]:
    """
    Apply all labeling functions to a sample and aggregate results.

    Args:
        sample_data: Dictionary containing 'midi_data', 'audio_data', 'sr', etc.
        labeling_functions: Dictionary mapping dimensions to lists of labeling functions
        use_adaptive_weights: Whether to use learned weights (requires weight_learner)
        weight_learner: Adaptive weight learner with learned weights

    Returns:
        Dictionary mapping dimension names to aggregated scores (0-100)
    """
    aggregator = WeakSupervisionAggregator(
        aggregation_method="weighted_average",
        min_coverage=1,
    )

    # Collect labels from all functions
    label_matrix = {}

    for dimension, lf_list in labeling_functions.items():
        labels_weights = []

        for lf in lf_list:
            # Apply labeling function
            label = lf(**sample_data)

            if label is not None:
                # Use learned weight if available, otherwise use function's default weight
                if use_adaptive_weights and weight_learner and dimension in weight_learner.learned_weights:
                    # Find index of this function in the learned weights
                    # For now, use function's default weight
                    weight = lf.weight
                else:
                    weight = lf.weight

                labels_weights.append((label, weight))

        label_matrix[dimension] = labels_weights

    # Aggregate labels
    aggregated = aggregator.aggregate_labels(label_matrix)

    # Extract final scores (use 75.0 as default if no label available)
    final_scores = {}
    for dimension, (label, confidence) in aggregated.items():
        if label is None:
            # No valid labels for this dimension - use neutral score
            final_scores[dimension] = 75.0
        else:
            final_scores[dimension] = label

    return final_scores


if __name__ == "__main__":
    print("Weak supervision aggregation module loaded successfully")
    print("- Weighted average aggregation (default)")
    print("- Median aggregation (robust to outliers)")
    print("- Trimmed mean aggregation")
    print("- Adaptive weight learning (Snorkel-style EM)")
