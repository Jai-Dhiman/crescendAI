"""Tests for PercePiano bridge (composite label computation)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_compute_weights_normalized():
    from masterclass_experiments.bridge import compute_weights

    # Mapping: "sound_quality" -> ["dynamic_range", "timbre_depth"]
    # R2: dynamic_range=0.596, timbre_depth=0.582
    dim_mapping = {"sound_quality": ["dynamic_range", "timbre_depth"]}
    r2 = {"dynamic_range": 0.596, "timbre_depth": 0.582}

    weights = compute_weights(dim_mapping, r2)

    assert "sound_quality" in weights
    w = weights["sound_quality"]
    assert len(w) == 2
    # Weights should be normalized to sum to 1
    total = sum(w.values())
    assert total == pytest.approx(1.0, abs=0.001)


def test_compute_weights_zero_r2():
    from masterclass_experiments.bridge import compute_weights

    dim_mapping = {"technique": ["timing"]}
    r2 = {"timing": -0.05}  # Negative R2 -> clamped to 0

    weights = compute_weights(dim_mapping, r2)

    # With single dim clamped to 0, weight should still be 1.0 (uniform fallback)
    assert weights["technique"]["timing"] == pytest.approx(1.0)


def test_compute_weights_empty_mapping():
    from masterclass_experiments.bridge import compute_weights

    dim_mapping = {"technique": []}
    r2 = {}

    weights = compute_weights(dim_mapping, r2)
    assert weights["technique"] == {}


def test_compute_composite_labels():
    from masterclass_experiments.bridge import compute_composite_labels

    # 2 segments, 19 dims each (only first 2 matter for this test)
    labels = {
        "seg_a": np.array([0.5, 0.8] + [0.0] * 17),
        "seg_b": np.array([0.3, 0.6] + [0.0] * 17),
    }
    # "sound" maps to dims 0 (timing=0.5/0.3) and 1 (art_length=0.8/0.6)
    weights = {"sound": {"timing": 0.4, "articulation_length": 0.6}}

    from audio_experiments.constants import PERCEPIANO_DIMENSIONS

    dim_index = {d: i for i, d in enumerate(PERCEPIANO_DIMENSIONS)}

    composites = compute_composite_labels(labels, weights, dim_index)

    assert "seg_a" in composites
    # 0.4 * 0.5 + 0.6 * 0.8 = 0.20 + 0.48 = 0.68
    assert composites["seg_a"]["sound"] == pytest.approx(0.68, abs=0.01)
    # 0.4 * 0.3 + 0.6 * 0.6 = 0.12 + 0.36 = 0.48
    assert composites["seg_b"]["sound"] == pytest.approx(0.48, abs=0.01)


def test_save_and_load_composite_labels():
    from masterclass_experiments.bridge import save_composite_labels, load_composite_labels

    composites = {
        "seg_a": {"sound": 0.68, "shaping": 0.42},
        "seg_b": {"sound": 0.48, "shaping": 0.55},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "composite_labels.json"
        save_composite_labels(composites, path)
        loaded = load_composite_labels(path)

    assert loaded["seg_a"]["sound"] == pytest.approx(0.68)
    assert loaded["seg_b"]["shaping"] == pytest.approx(0.55)
