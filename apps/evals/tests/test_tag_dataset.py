from __future__ import annotations

import json
from pathlib import Path

import pytest

from teaching_knowledge.scripts.tag_dataset import (
    RecordingTags,
    build_dataset_index,
    tag_recording,
)


def test_tag_recording_bach_short() -> None:
    manifest = {
        "piece_slug": "bach_wtc_1_prelude",
        "title": "WTC Book 1 Prelude No. 1",
        "composer": "Johann Sebastian Bach",
        "skill_bucket": 3,
    }
    cache = {"total_duration_seconds": 25.0, "chunks": []}
    tags = tag_recording("abc123", manifest, cache)
    assert tags.recording_id == "abc123"
    assert tags.composer_era == "Baroque"
    assert tags.skill_bucket == 3
    assert tags.duration_bucket == "<30s"


def test_tag_recording_chopin_medium() -> None:
    manifest = {
        "piece_slug": "chopin_ballade_1",
        "title": "Ballade No. 1",
        "composer": "Chopin",
        "skill_bucket": 5,
    }
    cache = {"total_duration_seconds": 45.0, "chunks": []}
    tags = tag_recording("xyz789", manifest, cache)
    assert tags.composer_era == "Romantic"
    assert tags.duration_bucket == "30-60s"


def test_tag_recording_long_recording() -> None:
    manifest = {
        "piece_slug": "debussy_clair",
        "title": "Clair de Lune",
        "composer": "Debussy",
        "skill_bucket": 4,
    }
    cache = {"total_duration_seconds": 120.0}
    tags = tag_recording("def456", manifest, cache)
    assert tags.composer_era == "Impressionist"
    assert tags.duration_bucket == "60s+"


def test_tag_recording_unknown_composer_still_tags() -> None:
    manifest = {
        "piece_slug": "unknown",
        "title": "Unknown",
        "composer": "Nobody Famous",
        "skill_bucket": 2,
    }
    cache = {"total_duration_seconds": 10.0}
    tags = tag_recording("unk001", manifest, cache)
    assert tags.composer_era == "Unknown"
    assert tags.skill_bucket == 2


def test_build_dataset_index_writes_jsonl(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    bach_path = cache_dir / "bach1.json"
    bach_path.write_text(json.dumps({
        "recording_id": "bach1",
        "total_duration_seconds": 25.0,
        "chunks": [],
    }))
    chopin_path = cache_dir / "chopin1.json"
    chopin_path.write_text(json.dumps({
        "recording_id": "chopin1",
        "total_duration_seconds": 90.0,
        "chunks": [],
    }))

    manifest_lookup = {
        "bach1": {
            "piece_slug": "bach", "title": "Bach", "composer": "Bach", "skill_bucket": 3,
        },
        "chopin1": {
            "piece_slug": "chopin", "title": "Chopin", "composer": "Chopin", "skill_bucket": 5,
        },
    }

    out = tmp_path / "dataset_index.jsonl"
    build_dataset_index(manifest_lookup, cache_dir, out)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    rows = sorted([json.loads(line) for line in lines], key=lambda r: r["recording_id"])
    assert rows[0]["recording_id"] == "bach1"
    assert rows[0]["composer_era"] == "Baroque"
    assert rows[0]["duration_bucket"] == "<30s"
    assert rows[1]["recording_id"] == "chopin1"
    assert rows[1]["composer_era"] == "Romantic"
    assert rows[1]["duration_bucket"] == "60s+"
