"""
Backward-compatible imports for src.data.

All data modules have been reorganized into:
- src.percepiano.data: PercePiano datasets
- src.crescendai.data: Custom data processing

This file provides backward-compatible imports for existing code.
"""

# PercePiano data
from src.percepiano.data.percepiano_dataset import (
    PercePianoDataset,
    create_dataloaders,
    DIMENSIONS,
)
from src.percepiano.data.percepiano_vnet_dataset import (
    PercePianoVNetDataset,
    create_vnet_dataloaders,
)
from src.percepiano.data.percepiano_score_dataset import (
    PercePianoScoreDataset,
    create_score_dataloaders,
)

# CrescendAI data
from src.crescendai.data.dataset import PerformanceDataset
from src.crescendai.data.midi_processing import (
    OctupleMIDITokenizer,
    load_midi,
    align_midi_to_audio,
    extract_midi_features,
    segment_midi,
    encode_octuple_midi,
)
from src.crescendai.data.score_alignment import (
    MusicXMLParser,
    ScorePerformanceAligner,
    ScoreAlignmentFeatureExtractor,
    ScoreNote,
    AlignedNote,
    load_score_midi,
)
from src.crescendai.data.mixed_dataset import MixedLabelDataset, create_mixed_dataloaders

__all__ = [
    # PercePiano
    "PercePianoDataset",
    "create_dataloaders",
    "DIMENSIONS",
    "PercePianoVNetDataset",
    "create_vnet_dataloaders",
    "PercePianoScoreDataset",
    "create_score_dataloaders",
    # CrescendAI
    "PerformanceDataset",
    "OctupleMIDITokenizer",
    "load_midi",
    "align_midi_to_audio",
    "extract_midi_features",
    "segment_midi",
    "encode_octuple_midi",
    "MusicXMLParser",
    "ScorePerformanceAligner",
    "ScoreAlignmentFeatureExtractor",
    "ScoreNote",
    "AlignedNote",
    "load_score_midi",
    "MixedLabelDataset",
    "create_mixed_dataloaders",
]
