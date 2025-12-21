"""CrescendAI data loading and processing."""

from .dataset import PerformanceDataset
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
from .augmentation import (
    AudioAugmentation,
    MIDIAugmentation,
)
from .score_alignment import (
    MusicXMLParser,
    ScorePerformanceAligner,
    ScoreAlignmentFeatureExtractor,
    ScoreNote,
    AlignedNote,
    load_score_midi,
)
from .mixed_dataset import MixedDataset, create_mixed_dataloaders
from .gpu_augmentation import GPUAugmentation
from .degradation import AudioDegradation

__all__ = [
    "PerformanceDataset",
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
    "MIDIAugmentation",
    "MusicXMLParser",
    "ScorePerformanceAligner",
    "ScoreAlignmentFeatureExtractor",
    "ScoreNote",
    "AlignedNote",
    "load_score_midi",
    "MixedDataset",
    "create_mixed_dataloaders",
    "GPUAugmentation",
    "AudioDegradation",
]
