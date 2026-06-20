"""Smoke test: extract_bundle produces a bundle with the correct schema keys.

This test does NOT call the live AMT server. It verifies that a pre-existing
bundle file (if available) has the correct structure, or it skips if no bundle
has been generated yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BUNDLE_SCHEMA_KEYS = frozenset({
    "piece_id", "video_id", "audio_path",
    "notes", "pedal_events", "measure_table", "anchors", "substrate_versions",
})
ANCHOR_KEYS = frozenset({"perf_audio_sec", "score_audio_sec"})
SUBSTRATE_VERSION_KEYS = frozenset({"amt_checkpoint_hash", "parangonar_version", "bundle_schema"})

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_BUNDLE_ROOT = _MODULE_DIR.parents[3] / "data/evals/claim_bundles"


def _find_any_bundle() -> Path | None:
    if not DEFAULT_BUNDLE_ROOT.exists():
        return None
    for p in DEFAULT_BUNDLE_ROOT.rglob("*.json"):
        # Skip metadata/manifest files (e.g. _index.json written by the runner);
        # real bundles live at <piece>/<video>.json.
        if not p.name.endswith(".tmp") and not p.name.startswith("_"):
            return p
    return None


def test_bundle_schema_if_exists() -> None:
    """If any bundle has been extracted, verify its schema keys."""
    bundle_path = _find_any_bundle()
    if bundle_path is None:
        pytest.skip("No bundle files found; run extract_bundle first")
    bundle = json.loads(bundle_path.read_text())
    missing = BUNDLE_SCHEMA_KEYS - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {missing}"
    missing_anchors = ANCHOR_KEYS - set(bundle["anchors"].keys())
    assert not missing_anchors, f"anchors missing keys: {missing_anchors}"
    missing_versions = SUBSTRATE_VERSION_KEYS - set(bundle["substrate_versions"].keys())
    assert not missing_versions, f"substrate_versions missing keys: {missing_versions}"
    assert isinstance(bundle["notes"], list)
    assert isinstance(bundle["pedal_events"], list)
    assert isinstance(bundle["measure_table"], list)
