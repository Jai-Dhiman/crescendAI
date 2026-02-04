"""Score alignment training infrastructure.

This module provides tools for aligning student piano performances to rendered
MIDI scores in MuQ embedding space using DTW-based algorithms.

Submodules:
    config: Constants and configuration for MuQ, DTW, and training
    data: ASAP dataset parsing and PyTorch datasets for alignment pairs
    rendering: MIDI-to-audio rendering with Pianoteq
    alignment: DTW algorithms (standard and soft-DTW) and evaluation metrics
    models: Learned projection models for embedding space alignment
    training: Training orchestration and experiment runners
"""

from . import config
from .config import (
    MUQ_SAMPLE_RATE,
    MUQ_HOP_LENGTH,
    MUQ_FRAME_RATE,
    MUQ_HIDDEN_SIZE,
    DTW_DISTANCE_METRIC,
    SOFT_DTW_GAMMA,
)
