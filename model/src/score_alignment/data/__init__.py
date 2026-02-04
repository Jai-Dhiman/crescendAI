"""Data loading and dataset classes for score alignment.

This submodule provides:
    - ASAP dataset parsing utilities
    - PyTorch datasets for frame-level and measure-level alignment
    - Collate functions for variable-length sequence batching
"""

from .asap import (
    ASAPPerformance,
    NoteAlignment,
    ASAPDatasetIndex,
    parse_asap_metadata,
    load_note_alignments,
    extract_onset_pairs,
    get_measure_boundaries,
)
from .alignment_dataset import (
    FrameAlignmentDataset,
    MeasureAlignmentDataset,
    frame_alignment_collate_fn,
)
