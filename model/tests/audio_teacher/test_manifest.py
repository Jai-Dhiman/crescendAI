"""Manifest loader behavior: full validation on load, loud failures."""
from __future__ import annotations

import json

import pytest
import yaml

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


def test_missing_offloaded_clip_error_contains_rehydrate_command(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory([{"id": "p1"}])
    missing = tmp_path / "clips" / "p1_a.wav"
    missing.unlink()
    registry = tmp_path / "r2_offload.json"
    registry.write_text(
        json.dumps(
            {
                "bucket": "crescendai-bucket",
                "remote_name": "r2",
                "entries": {
                    "clips": {"r2_prefix": "mirex-probe/clips", "reason": "test offload"}
                },
            }
        )
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(manifest_path, repo_root=tmp_path, offload_registry=registry)
    message = str(excinfo.value)
    assert "p1_a.wav" in message
    assert "rclone copy r2:crescendai-bucket/mirex-probe/clips clips" in message


def test_missing_unregistered_clip_still_fails_naming_the_path(
    tmp_path, manifest_factory
):
    manifest_path = manifest_factory([{"id": "p1"}])
    (tmp_path / "clips" / "p1_b.wav").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(
            manifest_path,
            repo_root=tmp_path,
            offload_registry=tmp_path / "no_registry.json",
        )
    assert "p1_b.wav" in str(excinfo.value)


def _mutate_schema_version(doc):
    doc["schema_version"] = 2


def _mutate_axis(doc):
    doc["pairs"][0]["axis"] = "rubato"


def _mutate_population(doc):
    doc["pairs"][0]["population"] = "studio"


def _mutate_degraded(doc):
    doc["pairs"][0]["degraded"] = "c"


def _mutate_duplicate_id(doc):
    doc["pairs"].append(dict(doc["pairs"][0]))


def _mutate_missing_key(doc):
    del doc["pairs"][0]["description"]


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_schema_version,
        _mutate_axis,
        _mutate_population,
        _mutate_degraded,
        _mutate_duplicate_id,
        _mutate_missing_key,
    ],
    ids=[
        "schema_version", "axis", "population", "degraded",
        "duplicate_id", "missing_key",
    ],
)
def test_schema_violations_raise_manifest_error(tmp_path, manifest_factory, mutate):
    from audio_teacher.manifest import ManifestError

    manifest_path = manifest_factory([{"id": "p1"}])
    doc = yaml.safe_load(manifest_path.read_text())
    mutate(doc)
    manifest_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ManifestError):
        load_manifest(manifest_path, repo_root=tmp_path)
