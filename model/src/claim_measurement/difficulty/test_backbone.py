"""Tests for the Backbone protocol's test double.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.backbone import FakeBackbone


def test_fake_backbone_returns_declared_poolings_and_dim():
    backbone = FakeBackbone(pooling_names=("mean_pool", "last_token"), dim=6)

    result = backbone.embed(Path("/fake/piece.mid"))

    assert set(result) == {"mean_pool", "last_token"}
    assert result["mean_pool"].shape == (6,)
    assert result["mean_pool"].dtype == np.float32


def test_fake_backbone_is_deterministic_per_path_but_differs_across_paths():
    backbone = FakeBackbone()

    a1 = backbone.embed(Path("/fake/a.mid"))["embedding"]
    a2 = backbone.embed(Path("/fake/a.mid"))["embedding"]
    b = backbone.embed(Path("/fake/b.mid"))["embedding"]

    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
