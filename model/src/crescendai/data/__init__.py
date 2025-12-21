"""CrescendAI data loading and processing."""

from .dataset import PerformanceDataset, create_dataloaders
from .audio_processing import (
    load_audio,
    compute_cqt,
    normalize_audio,
)
from .midi_processing import (
    OctupleMIDITokenizer,
    load_midi,
    align_midi_to_audio,
    extract_midi_features,
    segment_midi,
    encode_octuple_midi,
)
from .augmentation import AudioAugmentation
from .score_alignment import (
    MusicXMLParser,
    ScorePerformanceAligner,
    ScoreAlignmentFeatureExtractor,
    ScoreNote,
    AlignedNote,
    load_score_midi,
)
from .mixed_dataset import MixedLabelDataset, create_mixed_dataloaders
from .gpu_augmentation import GPUAudioAugmentation
from .degradation import degrade_audio_quality, apply_quality_tier, QualityTier

__all__ = [
    "PerformanceDataset",
    "create_dataloaders",
    "load_audio",
    "compute_cqt",
    "normalize_audio",
    "OctupleMIDITokenizer",
    "load_midi",
    "align_midi_to_audio",
    "extract_midi_features",
    "segment_midi",
    "encode_octuple_midi",
    "AudioAugmentation",
    "MusicXMLParser",
    "ScorePerformanceAligner",
    "ScoreAlignmentFeatureExtractor",
    "ScoreNote",
    "AlignedNote",
    "load_score_midi",
    "MixedLabelDataset",
    "create_mixed_dataloaders",
    "GPUAudioAugmentation",
    "degrade_audio_quality",
    "apply_quality_tier",
    "QualityTier",
]
