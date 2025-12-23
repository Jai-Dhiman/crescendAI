"""
Backward-compatible imports for src.models.

All models have been reorganized into:
- src.percepiano.models: PercePiano SOTA replica
- src.crescendai.models: Custom multi-modal models
- src.shared.models: Shared model components

This file provides backward-compatible imports for existing code.
"""

# PercePiano models
# CrescendAI models
from src.crescendai.models.audio_encoder import MERTEncoder
from src.crescendai.models.calibration import IsotonicCalibrator, TemperatureScaling
from src.crescendai.models.fusion_concat import ConcatenationFusion
from src.crescendai.models.fusion_crossattn import CrossAttentionFusion
from src.crescendai.models.fusion_gated import FiLMFusion, GatedFusion
from src.crescendai.models.lightning_module import PerformanceEvaluationModel
from src.crescendai.models.midi_encoder import MIDIBertEncoder
from src.crescendai.models.midi_only_module import MIDIOnlyModule
from src.crescendai.models.projection import DualProjection, ProjectionHead
from src.crescendai.models.score_aligned_module import (
    ScoreAlignedModule,
    ScoreAlignedModuleWithFallback,
)
from src.crescendai.models.score_encoder import (
    GlobalFeatureEncoder,
    NoteFeatureEncoder,
    ScoreAlignmentEncoder,
    ScoreMIDIFusion,
    TempoCurveEncoder,
)
from src.percepiano.models.context_attention import ContextAttention
from src.percepiano.models.han_encoder import HanEncoder
from src.percepiano.models.hierarchy_utils import (
    compute_actual_lengths,
    make_higher_node,
    run_hierarchy_lstm_with_pack,
    span_beat_to_note_num,
)
from src.percepiano.models.percepiano_replica import (
    PERCEPIANO_DIMENSIONS,
    PercePianoHAN,
    PercePianoReplicaModule,
    PercePianoVNetModule,
)

# Shared models
from src.shared.models.aggregation import (
    HierarchicalAggregator,
    PercePianoSelfAttention,
)
from src.shared.models.mtl_head import MultiTaskHead, PercePianoHead

__all__ = [
    # PercePiano
    "PercePianoReplicaModule",
    "PercePianoVNetModule",
    "PercePianoHAN",
    "PERCEPIANO_DIMENSIONS",
    "HanEncoder",
    "ContextAttention",
    # CrescendAI
    "MERTEncoder",
    "MIDIBertEncoder",
    "PerformanceEvaluationModel",
    "MIDIOnlyModule",
    "ScoreAlignedModule",
    "ScoreAlignedModuleWithFallback",
    "ScoreAlignmentEncoder",
    "ScoreMIDIFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    "FiLMFusion",
    "ConcatenationFusion",
    "ProjectionHead",
    "DualProjection",
    "TemperatureScaling",
    "IsotonicCalibrator",
    # Shared
    "HierarchicalAggregator",
    "PercePianoSelfAttention",
    "MultiTaskHead",
    "PercePianoHead",
]
