"""
Backward-compatible imports for src.utils.

Utils have been moved to src.shared.utils.
This file provides backward-compatible imports.
"""

from src.shared.utils.memory_profiler import (
    DetailedMemoryProfiler,
    MemoryProfilerCallback,
    get_memory_stats,
    log_memory,
    profile_object_counts,
)
from src.shared.utils.preflight_validation import (
    DataValidationError,
    EncoderValidationError,
    MIDIValidationError,
    PreflightValidationError,
    ScoreValidationError,
    run_preflight_validation,
    validate_data_files,
    validate_midi_files,
    validate_pretrained_encoder,
    validate_score_files,
)

__all__ = [
    "PreflightValidationError",
    "ScoreValidationError",
    "MIDIValidationError",
    "EncoderValidationError",
    "DataValidationError",
    "run_preflight_validation",
    "validate_data_files",
    "validate_midi_files",
    "validate_score_files",
    "validate_pretrained_encoder",
    "MemoryProfilerCallback",
    "DetailedMemoryProfiler",
    "get_memory_stats",
    "log_memory",
    "profile_object_counts",
]
