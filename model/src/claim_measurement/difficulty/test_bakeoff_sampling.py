"""Tests for bakeoff_sampling.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

from claim_measurement.difficulty.bakeoff_sampling import (
    ManifestEntry,
    load_bakeoff_manifest,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_load_bakeoff_manifest_joins_and_filters(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "new_clean_data.json"
    mid_dir = tmp_path / "transkun_mid"
    mid_dir.mkdir()

    _write_json(manifest_path, [
        {"seg_id": "has_composer_has_midi", "key": "A.mid", "grade": 3,
         "video_id": "x", "midi_name": "mid/A.mid"},
        {"seg_id": "no_composer", "key": "B.mid", "grade": 5,
         "video_id": "y", "midi_name": "mid/B.mid"},
        {"seg_id": "no_midi_on_disk", "key": "C.mid", "grade": 1,
         "video_id": "z", "midi_name": "mid/C.mid"},
    ])
    _write_json(labels_path, {
        "A.mid": {"composer": "Bach"},
        "B.mid": {"composer": ""},
        "C.mid": {"composer": "Czerny"},
    })
    (mid_dir / "has_composer_has_midi.mid").write_bytes(b"")
    # no_midi_on_disk.mid deliberately absent

    entries = load_bakeoff_manifest(manifest_path, labels_path, mid_dir)

    assert entries == [ManifestEntry(seg_id="has_composer_has_midi", key="A.mid",
                                      grade=3, composer="Bach")]
