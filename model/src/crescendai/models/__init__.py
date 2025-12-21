"""CrescendAI multi-modal model implementations."""

from .audio_encoder import MERTEncoder
from .midi_encoder import MIDIBertEncoder
from .lightning_module import PerformanceEvaluationModel
from .midi_only_module import MIDIOnlyModule
from .score_aligned_module import ScoreAlignedModule, ScoreAlignedModuleWithFallback
from .score_encoder import (
    ScoreAlignmentEncoder,
    ScoreMIDIFusion,
    NoteFeatureEncoder,
    GlobalFeatureEncoder,
    TempoCurveEncoder,
)
from .fusion_crossattn import CrossAttentionFusion
from .fusion_gated import GatedFusion, FiLMFusion
from .fusion_concat import ConcatenationFusion
from .projection import ProjectionHead, DualProjection
from .calibration import TemperatureScaling, IsotonicCalibrator

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
