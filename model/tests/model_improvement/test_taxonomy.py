import json
import numpy as np
import pytest
from model_improvement.taxonomy import (
    DIMENSIONS,
    NUM_DIMS,
    load_composite_labels,
    load_dimension_definitions,
    get_dimension_names,
)


def test_dimensions_constant():
    assert DIMENSIONS == [
        "dynamics", "timing", "pedaling",
        "articulation", "phrasing", "interpretation",
    ]
    assert NUM_DIMS == 6


def test_load_composite_labels(tmp_path):
    data = {
        "seg_a": {"dynamics": 0.5, "timing": 0.6, "pedaling": 0.7,
                  "articulation": 0.4, "phrasing": 0.3, "interpretation": 0.8},
        "seg_b": {"dynamics": 0.1, "timing": 0.2, "pedaling": 0.3,
                  "articulation": 0.4, "phrasing": 0.5, "interpretation": 0.6},
    }
    path = tmp_path / "composite_labels.json"
    path.write_text(json.dumps(data))

    result = load_composite_labels(path)
    assert set(result.keys()) == {"seg_a", "seg_b"}
    assert isinstance(result["seg_a"], np.ndarray)
    assert result["seg_a"].shape == (6,)
    np.testing.assert_allclose(result["seg_a"], [0.5, 0.6, 0.7, 0.4, 0.3, 0.8])


def test_load_dimension_definitions(tmp_path):
    data = {"dimensions": {"dynamics": {"description": "test"}}}
    path = tmp_path / "dimension_definitions.json"
    path.write_text(json.dumps(data))
    result = load_dimension_definitions(path)
    assert "dimensions" in result


def test_get_dimension_names():
    names = get_dimension_names()
    assert len(names) == 6
    assert names[0] == "dynamics"
