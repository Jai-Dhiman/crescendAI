"""Neural network models for piano performance evaluation."""

from .audio_encoder import MERTEncoder
from .midi_encoder import MIDIBertEncoder
from .fusion_crossattn import CrossAttentionFusion
from .fusion_concat import ConcatenationFusion
from .fusion_gated import GatedFusion, FiLMFusion
from .projection import ProjectionHead, DualProjection
from .aggregation import HierarchicalAggregator, PercePianoSelfAttention
from .mtl_head import MultiTaskHead, PercePianoHead
from .lightning_module import PerformanceEvaluationModel
from .midi_only_module import MIDIOnlyModule
from .score_encoder import (
    ScoreAlignmentEncoder,
    ScoreMIDIFusion,
    NoteFeatureEncoder,
    GlobalFeatureEncoder,
    TempoCurveEncoder,
)
from .score_aligned_module import (
    ScoreAlignedModule,
    ScoreAlignedModuleWithFallback,
)

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
    "PercePianoSelfAttention",
    "MultiTaskHead",
    "PercePianoHead",
    "PerformanceEvaluationModel",
    "MIDIOnlyModule",
    "ScoreAlignmentEncoder",
    "ScoreMIDIFusion",
    "NoteFeatureEncoder",
    "GlobalFeatureEncoder",
    "TempoCurveEncoder",
    "ScoreAlignedModule",
    "ScoreAlignedModuleWithFallback",
]
