"""Tests for bakeoff_paths.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
from pathlib import Path

from claim_measurement.difficulty.bakeoff_paths import resolve_paths


def test_resolve_paths_uses_override_root():
    paths = resolve_paths(data_root=Path("/tmp/fake_data_root"))
    assert paths.manifest == Path("/tmp/fake_data_root/results/amt_gap_curve/manifest.json")
    assert paths.labels == Path("/tmp/fake_data_root/raw/psyllabus/new_clean_data.json")
    assert paths.transkun_mid_dir == Path("/tmp/fake_data_root/results/amt_gap_curve/transkun_mid")
    assert paths.emb_root == Path("/tmp/fake_data_root/results/bakeoff")
