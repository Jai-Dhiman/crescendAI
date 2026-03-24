"""Tests for T5 scenario generation from skill_eval manifests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from pipeline.practice_eval.generate_t5_scenarios import (
    load_manifest,
    manifest_to_scenario,
    generate_scenario_file,
)


SAMPLE_MANIFEST = textwrap.dedent("""\
    piece: fur_elise
    title: Fur Elise
    composer: Beethoven
    recordings:
    - video_id: abc123
      title: 'Fur Elise - Beginner'
      channel: 'Test Channel'
      duration_seconds: 120
      skill_bucket: 1
      label_rationale: 'keyword match'
      downloaded: true
      download_error: null
    - video_id: def456
      title: 'Fur Elise - Intermediate'
      channel: 'Another Channel'
      duration_seconds: 180
      skill_bucket: 3
      label_rationale: 'human correction'
      downloaded: false
      download_error: null
    - video_id: ghi789
      title: 'Fur Elise - Advanced'
      channel: 'Pro Channel'
      duration_seconds: 200
      skill_bucket: 5
      label_rationale: 'known professional'
      downloaded: true
      download_error: null
""")


def test_load_manifest(tmp_path: Path) -> None:
    """load_manifest reads a manifest YAML and returns its dict."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(SAMPLE_MANIFEST)
    data = load_manifest(manifest_path)
    assert data["piece"] == "fur_elise"
    assert len(data["recordings"]) == 3


def test_manifest_to_scenario_basic() -> None:
    """manifest_to_scenario converts recordings to candidate format."""
    manifest = yaml.safe_load(SAMPLE_MANIFEST)
    scenario = manifest_to_scenario(manifest)

    assert "candidates" in scenario
    # piece_query deliberately omitted
    assert "piece_query" not in scenario

    candidates = scenario["candidates"]
    # Only downloaded=true recordings should appear
    assert len(candidates) == 2

    # Check first candidate
    c1 = candidates[0]
    assert c1["video_id"] == "abc123"
    assert c1["skill_level"] == 1
    assert c1["include"] is True
    assert c1["title"] == "Fur Elise - Beginner"
    assert "T5 skill corpus" in c1["general_notes"]
    assert "bucket 1" in c1["general_notes"]

    # Check second candidate
    c2 = candidates[1]
    assert c2["video_id"] == "ghi789"
    assert c2["skill_level"] == 5


def test_manifest_to_scenario_skips_not_downloaded() -> None:
    """Only downloaded: true recordings are included."""
    manifest = yaml.safe_load(SAMPLE_MANIFEST)
    scenario = manifest_to_scenario(manifest)
    video_ids = [c["video_id"] for c in scenario["candidates"]]
    assert "def456" not in video_ids
    assert "abc123" in video_ids
    assert "ghi789" in video_ids


def test_manifest_to_scenario_empty_when_none_downloaded() -> None:
    """If no recordings are downloaded, candidates list is empty."""
    manifest_text = textwrap.dedent("""\
        piece: test_piece
        title: Test Piece
        composer: Test
        recordings:
        - video_id: aaa
          title: 'Test'
          channel: 'Ch'
          duration_seconds: 60
          skill_bucket: 2
          label_rationale: 'test'
          downloaded: false
          download_error: null
    """)
    manifest = yaml.safe_load(manifest_text)
    scenario = manifest_to_scenario(manifest)
    assert scenario["candidates"] == []


def test_generate_scenario_file_writes_valid_yaml(tmp_path: Path) -> None:
    """generate_scenario_file writes a valid YAML file that load_scenarios can parse."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(SAMPLE_MANIFEST)

    output_path = tmp_path / "fur_elise.yaml"
    generate_scenario_file(manifest_path, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = yaml.safe_load(f)

    assert "candidates" in data
    assert len(data["candidates"]) == 2
    # Verify all candidates have include: true
    for c in data["candidates"]:
        assert c["include"] is True
        assert "skill_level" in c
        assert "video_id" in c
        assert "title" in c
        assert "general_notes" in c
