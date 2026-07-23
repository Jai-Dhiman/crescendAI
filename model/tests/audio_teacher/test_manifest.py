"""Manifest loader behavior: full validation on load, loud failures."""
from __future__ import annotations

import pytest

from audio_teacher.manifest import load_manifest


def test_valid_manifest_loads_pairs_in_order_with_resolved_clips(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory(
        [
            {"id": "p1", "axis": "pedaling", "population": "real", "degraded": "a"},
            {"id": "p2", "axis": "dynamics", "population": "synthetic", "degraded": "b"},
        ]
    )
    manifest = load_manifest(manifest_path, repo_root=tmp_path)
    assert manifest.sample_rate == 16000
    assert [p.pair_id for p in manifest.pairs] == ["p1", "p2"]
    p1, p2 = manifest.pairs
    assert p1.axis == "pedaling" and p1.population == "real" and p1.degraded == "a"
    assert p2.axis == "dynamics" and p2.population == "synthetic" and p2.degraded == "b"
    assert p1.clip_a.is_absolute() and p1.clip_a.exists()
    assert p2.clip_b == tmp_path / "clips" / "p2_b.wav"
