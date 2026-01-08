"""Model classes for audio experiments."""

from .base import BaseMERTModel, ccc_loss
from .probes import LinearProbeModel, StatsMLPModel
from .mel_cnn import MelCNNModel
from .cross_attention import (
    CrossAttentionFusion,
    FusionMLPModel,
    MultiHeadCrossAttention,
    train_fusion_mlp_cv,
)

__all__ = [
    # Base models
    "BaseMERTModel",
    "ccc_loss",
    "LinearProbeModel",
    "StatsMLPModel",
    "MelCNNModel",
    # Fusion models
    "CrossAttentionFusion",
    "FusionMLPModel",
    "MultiHeadCrossAttention",
    "train_fusion_mlp_cv",
]
