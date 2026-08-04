"""Tests for the MoonBeam pooling math, against an injected fake loader --
no transformers_minimal fork, no moonbeam_839M.pt, no isolated venv needed.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

import numpy as np
import pytest

from claim_measurement.difficulty.moonbeam_backbone import MoonBeamBackbone


def test_embed_computes_mean_pool_and_last_token_from_injected_loader():
    hidden_states = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 6.0]], dtype=np.float32)
    backbone = MoonBeamBackbone(loader=lambda midi_path: hidden_states)

    result = backbone.embed(Path("/fake/piece.mid"))

    assert set(result) == {"mean_pool", "last_token"}
    np.testing.assert_allclose(result["mean_pool"], [3.0, 2.0])
    np.testing.assert_allclose(result["last_token"], [5.0, 6.0])


def test_construction_without_loader_fails_loudly():
    with pytest.raises(ValueError, match="isolated MoonBeam venv"):
        MoonBeamBackbone(loader=None)
