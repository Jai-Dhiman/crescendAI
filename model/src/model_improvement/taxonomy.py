"""Teacher-grounded taxonomy: 6-dim composite label system.

Single source of truth for dimension names, count, and label loading.
Replaces the hardcoded 19-dim PercePiano labels throughout the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIMENSIONS: list[str] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
]

NUM_DIMS: int = len(DIMENSIONS)


def load_composite_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Load composite labels JSON, returning segment_id -> [6] numpy array.

    The JSON has structure: {segment_id: {dim_name: score, ...}}.
    Returns arrays with dimensions ordered per DIMENSIONS constant.
    """
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for seg_id, dim_scores in raw.items():
        vec = np.array([dim_scores[d] for d in DIMENSIONS], dtype=np.float32)
        result[seg_id] = vec
    return result


def load_dimension_definitions(path: str | Path) -> dict:
    """Load dimension_definitions.json (full taxonomy metadata)."""
    with open(path) as f:
        return json.load(f)


def get_dimension_names() -> list[str]:
    """Return ordered dimension names."""
    return list(DIMENSIONS)
