"""CrescendAI data loading and processing."""

from .audio_processing import (
    compute_cqt,
    load_audio,
    normalize_audio,
)
from .augmentation import AudioAugmentation
from .dataset import PerformanceDataset, create_dataloaders
from .degradation import QualityTier, apply_quality_tier, degrade_audio_quality
from .gpu_augmentation import GPUAudioAugmentation
from .midi_processing import (
    OctupleMIDITokenizer,
    align_midi_to_audio,
    encode_octuple_midi,
    extract_midi_features,
    load_midi,
    segment_midi,
)
from .mixed_dataset import MixedLabelDataset, create_mixed_dataloaders
from .score_alignment import (
    AlignedNote,
    MusicXMLParser,
    ScoreAlignmentFeatureExtractor,
    ScoreNote,
    ScorePerformanceAligner,
    load_score_midi,
)

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
