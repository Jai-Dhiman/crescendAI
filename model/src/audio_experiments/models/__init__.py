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
from .phase3 import (
    StatsPoolingModel,
    UncertaintyWeightedModel,
    DimensionSpecificModel,
    TransformerPoolingModel,
    MultiScalePoolingModel,
    MultiLayerMERTModel,
)
from .fusion_models import (
    ModalityDropoutFusion,
    OrthogonalityFusion,
    ResidualFusion,
    DimensionWeightedFusion,
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
    # Phase 3 models
    "StatsPoolingModel",
    "UncertaintyWeightedModel",
    "DimensionSpecificModel",
    "TransformerPoolingModel",
    "MultiScalePoolingModel",
    "MultiLayerMERTModel",
    # Learned fusion models
    "ModalityDropoutFusion",
    "OrthogonalityFusion",
    "ResidualFusion",
    "DimensionWeightedFusion",
]
