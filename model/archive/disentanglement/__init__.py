"""Disentanglement module for separating piece characteristics from performer expression.

This module implements three approaches to improve pairwise ranking accuracy
for same-piece comparisons in piano performance evaluation.

Approaches:
- A: Contrastive Pairwise Ranking (InfoNCE + margin ranking)
- B: Siamese Dimension-Specific Ranking (per-dimension ranking heads)
- C: Disentangled Dual-Encoder (adversarial piece classification)

And their combinations: A+B, A+C, B+C, A+B+C
"""

from .models import (
    GradientReversalLayer,
    get_grl_lambda,
    ContrastivePairwiseRankingModel,
    SiameseDimensionRankingModel,
    DisentangledDualEncoderModel,
    ContrastiveDisentangledModel,
    SiameseDisentangledModel,
    FullCombinedModel,
    TripletRankingModel,
)

from .data import (
    build_multi_performer_pieces,
    create_piece_stratified_folds,
    PairwiseRankingDataset,
    HardPairRankingDataset,
    DisentanglementDataset,
    TripletRankingDataset,
    pairwise_collate_fn,
    disentanglement_collate_fn,
    triplet_collate_fn,
    sample_pairs_same_piece,
    sample_hard_pairs,
    compute_pairwise_statistics,
    get_fold_piece_mapping,
)

from .training import (
    run_pairwise_experiment,
    run_disentanglement_experiment,
    run_triplet_experiment,
    run_dimension_group_experiment,
    compute_pairwise_accuracy,
    compute_pairwise_metrics,
    compute_intra_piece_std,
    evaluate_disentanglement,
    bootstrap_pairwise_accuracy,
    compute_regression_pairwise_accuracy,
)

from .losses import (
    piece_based_infonce_loss,
    pairwise_margin_ranking_loss,
    DimensionWiseRankingLoss,
    ContrastiveRankingLoss,
    DisentanglementLoss,
    TripletPerformerLoss,
    TripletRankingLoss,
)

__all__ = [
    # Models
    "GradientReversalLayer",
    "get_grl_lambda",
    "ContrastivePairwiseRankingModel",
    "SiameseDimensionRankingModel",
    "DisentangledDualEncoderModel",
    "ContrastiveDisentangledModel",
    "SiameseDisentangledModel",
    "FullCombinedModel",
    "TripletRankingModel",
    # Data
    "build_multi_performer_pieces",
    "create_piece_stratified_folds",
    "PairwiseRankingDataset",
    "HardPairRankingDataset",
    "DisentanglementDataset",
    "TripletRankingDataset",
    "pairwise_collate_fn",
    "disentanglement_collate_fn",
    "triplet_collate_fn",
    "sample_pairs_same_piece",
    "sample_hard_pairs",
    "compute_pairwise_statistics",
    "get_fold_piece_mapping",
    # Training
    "run_pairwise_experiment",
    "run_disentanglement_experiment",
    "run_triplet_experiment",
    "run_dimension_group_experiment",
    "compute_pairwise_accuracy",
    "compute_pairwise_metrics",
    "compute_intra_piece_std",
    "evaluate_disentanglement",
    "bootstrap_pairwise_accuracy",
    "compute_regression_pairwise_accuracy",
    # Losses
    "piece_based_infonce_loss",
    "pairwise_margin_ranking_loss",
    "DimensionWiseRankingLoss",
    "ContrastiveRankingLoss",
    "DisentanglementLoss",
    "TripletPerformerLoss",
    "TripletRankingLoss",
]
