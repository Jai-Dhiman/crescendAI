"""Utility modules for training and debugging."""

from .memory_profiler import (
    MemoryProfilerCallback,
    DetailedMemoryProfiler,
    get_memory_stats,
    log_memory,
    profile_object_counts,
)

from .preflight_validation import (
    PreflightValidationError,
    ScoreValidationError,
    MIDIValidationError,
    EncoderValidationError,
    DataValidationError,
    run_preflight_validation,
    validate_data_files,
    validate_midi_files,
    validate_score_files,
    validate_pretrained_encoder,
)

__all__ = [
    # Memory profiler
    "MemoryProfilerCallback",
    "DetailedMemoryProfiler",
    "get_memory_stats",
    "log_memory",
    "profile_object_counts",
    # Preflight validation
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
]
