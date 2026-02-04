"""Configuration constants for score alignment.

This module defines constants for MuQ extraction, DTW algorithms,
projection networks, and training hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# MuQ audio model settings
MUQ_SAMPLE_RATE = 24000  # Hz
MUQ_HOP_LENGTH = 320  # samples
MUQ_FRAME_RATE = MUQ_SAMPLE_RATE // MUQ_HOP_LENGTH  # 75 fps
MUQ_HIDDEN_SIZE = 1024
MUQ_NUM_LAYERS = 24
MUQ_LAYER_RANGE = (9, 13)  # Layers 9-12 (exclusive end)


# DTW algorithm settings
DTW_DISTANCE_METRIC = "cosine"
SOFT_DTW_GAMMA = 1.0  # Smoothing parameter for soft-DTW
SAKOE_CHIBA_RADIUS = None  # None = no constraint, or int for band radius


# Evaluation thresholds
ONSET_ERROR_THRESHOLD_MS = 30.0  # Human perception threshold


# External resources
ASAP_REPO_URL = "https://github.com/CPJKU/asap-dataset.git"
PIANOTEQ_EXECUTABLE = "/usr/local/bin/pianoteq"  # Default path, override as needed


@dataclass
class ProjectionConfig:
    """Configuration for the projection MLP."""

    input_dim: int = MUQ_HIDDEN_SIZE
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    """Configuration for training the alignment model."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    max_epochs: int = 100
    patience: int = 10
    gradient_clip_val: float = 1.0
    num_workers: int = 4
    max_frames: int = 3000  # Max sequence length (40 seconds at 75 fps)

    # Soft-DTW specific
    soft_dtw_gamma: float = SOFT_DTW_GAMMA
    distance_metric: str = DTW_DISTANCE_METRIC

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class ExperimentConfig:
    """Full configuration for an alignment experiment."""

    # Experiment metadata
    exp_id: str = "alignment_exp"
    description: str = ""

    # Paths
    asap_root: Optional[Path] = None
    score_cache_dir: Optional[Path] = None
    perf_cache_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    log_dir: Optional[Path] = None

    # Model and training config
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Data splits
    train_keys: List[str] = field(default_factory=list)
    val_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.asap_root, str):
            self.asap_root = Path(self.asap_root)
        if isinstance(self.score_cache_dir, str):
            self.score_cache_dir = Path(self.score_cache_dir)
        if isinstance(self.perf_cache_dir, str):
            self.perf_cache_dir = Path(self.perf_cache_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(
        projection=ProjectionConfig(),
        training=TrainingConfig(),
    )
