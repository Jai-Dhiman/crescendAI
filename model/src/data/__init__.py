"""Data loading and processing modules for piano performance evaluation."""

from .midi_processing import (
    OctupleMIDITokenizer,
    load_midi,
    align_midi_to_audio,
    extract_midi_features,
    segment_midi,
    encode_octuple_midi,
)
from .percepiano_dataset import (
    PercePianoDataset,
    create_dataloaders,
    DIMENSIONS,
)
from .score_alignment import (
    MusicXMLParser,
    ScorePerformanceAligner,
    ScoreAlignmentFeatureExtractor,
    ScoreNote,
    AlignedNote,
    load_score_midi,
)
from .percepiano_score_dataset import (
    PercePianoScoreDataset,
    create_score_dataloaders,
)

__all__ = [
    # MIDI Processing
    "OctupleMIDITokenizer",
    "load_midi",
    "align_midi_to_audio",
    "extract_midi_features",
    "segment_midi",
    "encode_octuple_midi",
    # PercePiano Dataset
    "PercePianoDataset",
    "create_dataloaders",
    "DIMENSIONS",
    # Score Alignment
    "MusicXMLParser",
    "ScorePerformanceAligner",
    "ScoreAlignmentFeatureExtractor",
    "ScoreNote",
    "AlignedNote",
    "load_score_midi",
    # Score-aligned Dataset
    "PercePianoScoreDataset",
    "create_score_dataloaders",
]
