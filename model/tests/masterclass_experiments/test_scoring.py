"""Tests for multi-signal scoring and dimension selection."""

import numpy as np
import pytest


# -- Per-dimension R2 from the PercePiano audit (definitive_all_results.json) --
AUDIT_R2 = {
    "timing": 0.393,
    "articulation_length": 0.671,
    "articulation_touch": 0.542,
    "pedal_amount": 0.557,
    "pedal_clarity": 0.469,
    "timbre_variety": 0.443,
    "timbre_depth": 0.582,
    "timbre_brightness": 0.500,
    "timbre_loudness": 0.578,
    "dynamic_range": 0.596,
    "tempo": 0.270,
    "space": 0.653,
    "balance": 0.554,
    "drama": 0.448,
    "mood_valence": 0.424,
    "mood_energy": 0.529,
    "mood_imagination": 0.485,
    "sophistication": 0.663,
    "interpretation": 0.477,
}


def test_teacher_frequency():
    from masterclass_experiments.scoring import compute_teacher_frequency

    labels = np.array([0, 0, 0, 1, 1, -1, -1])
    freq = compute_teacher_frequency(labels)
    assert freq[0] == pytest.approx(3 / 5)
    assert freq[1] == pytest.approx(2 / 5)


def test_teacher_frequency_no_noise():
    from masterclass_experiments.scoring import compute_teacher_frequency

    labels = np.array([0, 0, 1, 1, 2, 2])
    freq = compute_teacher_frequency(labels)
    assert sum(freq.values()) == pytest.approx(1.0)


def test_muq_predictability():
    from masterclass_experiments.scoring import compute_muq_predictability

    # Mapping: cluster 0 -> ["dynamic_range", "timbre_loudness"]
    pp_mapping = {0: ["dynamic_range", "timbre_loudness"]}
    scores = compute_muq_predictability(pp_mapping, AUDIT_R2)
    # Mean of 0.596 and 0.578
    assert scores[0] == pytest.approx((0.596 + 0.578) / 2, abs=0.01)


def test_muq_predictability_unmapped():
    from masterclass_experiments.scoring import compute_muq_predictability

    pp_mapping = {0: []}  # No PercePiano mapping (e.g., "technique")
    scores = compute_muq_predictability(pp_mapping, AUDIT_R2)
    assert scores[0] == 0.0


def test_select_dimensions_keeps_frequent():
    from masterclass_experiments.scoring import select_dimensions

    candidates = {
        0: {"frequency": 0.15, "muq_r2": 0.5, "stop_delta_auc": 0.02},
        1: {"frequency": 0.08, "muq_r2": 0.0, "stop_delta_auc": 0.0},
        2: {"frequency": 0.02, "muq_r2": 0.0, "stop_delta_auc": 0.0},
    }
    kept, dropped = select_dimensions(candidates)
    # Cluster 0: freq > 5% AND muq_r2 > 0 -> KEEP
    assert 0 in kept
    # Cluster 1: freq > 5% but no soft signal -> DROP
    assert 1 in dropped
    # Cluster 2: freq < 3% and no soft signal -> DROP
    assert 2 in dropped


def test_select_dimensions_keeps_with_stop_signal():
    from masterclass_experiments.scoring import select_dimensions

    candidates = {
        0: {"frequency": 0.06, "muq_r2": 0.0, "stop_delta_auc": 0.03},
    }
    kept, _ = select_dimensions(candidates)
    # freq > 5% AND stop_delta_auc > 0 -> KEEP
    assert 0 in kept


def test_build_hierarchy():
    from masterclass_experiments.scoring import build_hierarchy

    # Flat list of named dimensions with their cluster embeddings
    dimensions = [
        {"name": "dynamics", "centroid": np.array([1.0, 0.0])},
        {"name": "tone_color", "centroid": np.array([1.1, 0.1])},
        {"name": "phrasing", "centroid": np.array([0.0, 1.0])},
        {"name": "timing", "centroid": np.array([0.1, 1.1])},
    ]
    hierarchy = build_hierarchy(dimensions, n_groups=2)
    assert len(hierarchy) == 2
    for group in hierarchy:
        assert "group_name" in group
        assert "dimensions" in group
        assert len(group["dimensions"]) >= 1
