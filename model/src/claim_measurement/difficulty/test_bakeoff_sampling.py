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


from claim_measurement.difficulty.bakeoff_sampling import composer_stratified_sample


def _make_entries(n_composers: int, pieces_per_composer: int) -> list[ManifestEntry]:
    entries = []
    for c in range(n_composers):
        for p in range(pieces_per_composer):
            entries.append(ManifestEntry(
                seg_id=f"c{c}_p{p}", key=f"c{c}_p{p}.mid",
                grade=p % 11, composer=f"composer_{c}",
            ))
    return entries


def test_composer_stratified_sample_covers_every_composer_and_hits_target_n():
    entries = _make_entries(n_composers=50, pieces_per_composer=20)  # 1000 entries

    sample = composer_stratified_sample(entries, target_n=200, seed=2026)

    assert len(sample) == 200
    assert len({e.seg_id for e in sample}) == 200  # no duplicates
    sampled_composers = {e.composer for e in sample}
    assert sampled_composers == {f"composer_{c}" for c in range(50)}  # every composer represented


def test_composer_stratified_sample_returns_everything_when_target_exceeds_pool():
    entries = _make_entries(n_composers=5, pieces_per_composer=3)  # 15 entries

    sample = composer_stratified_sample(entries, target_n=100, seed=2026)

    assert len(sample) == 15
