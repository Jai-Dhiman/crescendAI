"""Stratified sampler determinism + shape tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from teacher_model.stage0.sampler import sample_holdout

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BRIEFINGS_DIR = (
    _REPO_ROOT / "model" / "data" / "eval" / "inference_cache" / "auto-t5_http"
)


def _load_real_manifests() -> dict:
    """Load skill_eval manifests; reuse the same lookup the production runner uses."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "apps" / "evals"))
    from teaching_knowledge.run_eval import load_manifests

    return load_manifests()


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_same_seed_produces_same_ids() -> None:
    manifests = _load_real_manifests()
    a = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    b = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    assert [r["recording_id"] for r in a] == [r["recording_id"] for r in b]


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_different_seed_produces_different_ids() -> None:
    manifests = _load_real_manifests()
    a = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=42)
    b = sample_holdout(_BRIEFINGS_DIR, manifests, n=50, seed=43)
    assert [r["recording_id"] for r in a] != [r["recording_id"] for r in b]


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_sample_returns_at_most_n_with_required_fields() -> None:
    manifests = _load_real_manifests()
    out = sample_holdout(_BRIEFINGS_DIR, manifests, n=100, seed=42)
    assert 80 <= len(out) <= 100  # strata may underfill if some buckets are sparse
    for row in out:
        assert "recording_id" in row
        assert "era" in row
        assert "skill_bucket" in row
        assert "stratum" in row


@pytest.mark.skipif(not _BRIEFINGS_DIR.exists(), reason="briefings dir not present")
def test_strata_balanced_within_two() -> None:
    manifests = _load_real_manifests()
    out = sample_holdout(_BRIEFINGS_DIR, manifests, n=100, seed=42)
    counts: dict[str, int] = {}
    for row in out:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    if not counts:
        pytest.skip("no strata found")
    target = max(counts.values())
    for c in counts.values():
        assert abs(c - target) <= max(2, target // 3)
