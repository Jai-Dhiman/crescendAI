"""Shared path constants for all eval modules.

All eval scripts should import paths from here rather than computing
relative paths via Path(__file__).parents[N]. This eliminates fragile
parent-counting that breaks when files move.
"""

from pathlib import Path

# Project root: apps/evals/paths.py -> apps/evals -> apps -> crescendai
PROJECT_ROOT = Path(__file__).parents[2]

# Model data (checkpoints, eval cache, skill eval manifests, etc.)
MODEL_DATA = PROJECT_ROOT / "model" / "data"

# Model source code (for importing model_improvement.* when needed)
MODEL_SRC = PROJECT_ROOT / "model" / "src"

# Inference module (audio_chunker, models/*, constants, preprocessing/*)
INFERENCE_DIR = PROJECT_ROOT / "apps" / "inference"
