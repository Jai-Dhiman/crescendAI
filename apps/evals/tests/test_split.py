from __future__ import annotations

import json
from pathlib import Path

from teaching_knowledge.scripts.split import (
    Split,
    load_split,
    stratified_split,
    write_split,
)
from teaching_knowledge.scripts.tag_dataset import RecordingTags


def _make_tags(n: int) -> list[RecordingTags]:
    eras = ["Baroque", "Classical", "Romantic", "Impressionist"]
    tags = []
    for i in range(n):
        tags.append(
            RecordingTags(
                recording_id=f"rec_{i:04d}",
                composer_era=eras[i % len(eras)],
                skill_bucket=(i % 5) + 1,
                duration_bucket=["<30s", "30-60s", "60s+"][i % 3],
            )
        )
    return tags


def test_stratified_split_80_20_ratio() -> None:
    tags = _make_tags(100)
    split = stratified_split(tags, seed=42, holdout_ratio=0.2)
    assert len(split.holdout) == 20
    assert len(split.train) == 80
    assert set(split.train).isdisjoint(split.holdout)
    assert set(split.train) | set(split.holdout) == {t.recording_id for t in tags}


def test_stratified_split_is_deterministic() -> None:
    tags = _make_tags(100)
    a = stratified_split(tags, seed=42, holdout_ratio=0.2)
    b = stratified_split(tags, seed=42, holdout_ratio=0.2)
    assert a.train == b.train
    assert a.holdout == b.holdout


def test_stratified_split_different_seeds_differ() -> None:
    tags = _make_tags(100)
    a = stratified_split(tags, seed=1, holdout_ratio=0.2)
    b = stratified_split(tags, seed=2, holdout_ratio=0.2)
    assert set(a.holdout) != set(b.holdout)


def test_stratified_split_preserves_era_distribution() -> None:
    tags = _make_tags(200)
    split = stratified_split(tags, seed=42, holdout_ratio=0.2)
    holdout_ids = set(split.holdout)
    holdout_eras = [t.composer_era for t in tags if t.recording_id in holdout_ids]
    all_eras = [t.composer_era for t in tags]
    for era in set(all_eras):
        all_share = all_eras.count(era) / len(all_eras)
        holdout_share = holdout_eras.count(era) / len(holdout_eras) if holdout_eras else 0
        assert abs(all_share - holdout_share) < 0.1, f"era {era} drifted"


def test_write_and_load_split_roundtrip(tmp_path: Path) -> None:
    tags = _make_tags(50)
    split = stratified_split(tags, seed=7, holdout_ratio=0.2)
    path = tmp_path / "splits.json"
    write_split(split, path)

    assert path.exists()
    blob = json.loads(path.read_text())
    assert "train" in blob and "holdout" in blob

    train_set = load_split(path, which="train")
    holdout_set = load_split(path, which="holdout")
    all_set = load_split(path, which="all")
    assert train_set == set(split.train)
    assert holdout_set == set(split.holdout)
    assert all_set == train_set | holdout_set


def test_stratified_split_handles_small_strata() -> None:
    tags = [
        RecordingTags(f"r{i}", era, skill, "<30s")
        for i, (era, skill) in enumerate(
            [("Baroque", 1), ("Classical", 2), ("Romantic", 3), ("Impressionist", 4)]
        )
    ]
    split = stratified_split(tags, seed=0, holdout_ratio=0.25)
    assert len(split.train) + len(split.holdout) == 4
