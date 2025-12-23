"""CrescendAI multi-modal model implementations."""

from .audio_encoder import MERTEncoder
from .calibration import IsotonicCalibrator, TemperatureScaling
from .fusion_concat import ConcatenationFusion
from .fusion_crossattn import CrossAttentionFusion
from .fusion_gated import FiLMFusion, GatedFusion
from .lightning_module import PerformanceEvaluationModel
from .midi_encoder import MIDIBertEncoder
from .midi_only_module import MIDIOnlyModule
from .projection import DualProjection, ProjectionHead
from .score_aligned_module import ScoreAlignedModule, ScoreAlignedModuleWithFallback
from .score_encoder import (
    GlobalFeatureEncoder,
    NoteFeatureEncoder,
    ScoreAlignmentEncoder,
    ScoreMIDIFusion,
    TempoCurveEncoder,
)

__all__ = [
    "MERTEncoder",
    "MIDIBertEncoder",
    "PerformanceEvaluationModel",
    "MIDIOnlyModule",
    "ScoreAlignedModule",
    "ScoreAlignedModuleWithFallback",
    "ScoreAlignmentEncoder",
    "ScoreMIDIFusion",
    "NoteFeatureEncoder",
    "GlobalFeatureEncoder",
    "TempoCurveEncoder",
    "CrossAttentionFusion",
    "GatedFusion",
    "FiLMFusion",
    "ConcatenationFusion",
    "ProjectionHead",
    "DualProjection",
    "TemperatureScaling",
    "IsotonicCalibrator",
]
