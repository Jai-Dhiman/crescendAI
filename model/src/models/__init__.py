"""Neural network models for piano performance evaluation."""

from .audio_encoder import MERTEncoder
from .midi_encoder import MIDIBertEncoder
from .fusion_crossattn import CrossAttentionFusion
from .fusion_concat import ConcatenationFusion
from .fusion_gated import GatedFusion, FiLMFusion
from .projection import ProjectionHead, DualProjection
from .aggregation import HierarchicalAggregator
from .mtl_head import MultiTaskHead
from .lightning_module import PerformanceEvaluationModel

__all__ = [
    "MERTEncoder",
    "MIDIBertEncoder",
    "CrossAttentionFusion",
    "ConcatenationFusion",
    "GatedFusion",
    "FiLMFusion",
    "ProjectionHead",
    "DualProjection",
    "HierarchicalAggregator",
    "MultiTaskHead",
    "PerformanceEvaluationModel",
]
