"""Tests for push_train_dataset (#149 / #138 Phase 1) -- hermetic HF Jobs
training-bundle staging + upload, uploader injected so no test hits the network.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import pytest

from claim_measurement.difficulty.fold_plan import FoldPlan
from claim_measurement.difficulty.push_train_dataset import (
    BundleSources,
    stage_training_bundle,
)


def _write_fake_repo(repo_dir):
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "model_config.json").write_text("{}")
    (repo_dir / "README.md").write_text("moonbeam fork snapshot")


def test_stage_training_bundle_copies_every_referenced_piece_and_reports_counts(
    tmp_path,
):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    seg_ids = ["a", "b", "c", "d"]
    for seg_id in seg_ids:
        (midi_dir / f"{seg_id}.mid").write_bytes(b"midi-bytes")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)

    plans = [
        FoldPlan(
            fold=0, test_seg_ids=("a",), train_seg_ids=("b", "c"), val_seg_ids=("d",)
        ),
        FoldPlan(
            fold=1, test_seg_ids=("b",), train_seg_ids=("a", "c"), val_seg_ids=("d",)
        ),
    ]
    paths = BundleSources(
        midi_dir=midi_dir,
        grades={s: i for i, s in enumerate(seg_ids)},
        repo_snapshot_dir=repo_snapshot_dir,
    )
    staging_dir = tmp_path / "staging"

    report = stage_training_bundle(paths, plans, staging_dir)

    assert report.n_midis == 4  # {a,b,c,d}, deduplicated across both plans
    assert report.n_fold_plans == 2
    assert report.repo_snapshot_files == 2
    assert len(report.checksum) == 64  # sha256 hex digest
    for seg_id in seg_ids:
        assert (staging_dir / "midi" / f"{seg_id}.mid").exists()
    assert json.loads((staging_dir / "grades.json").read_text()) == {
        s: i for i, s in enumerate(seg_ids)
    }
    staged_plans = json.loads((staging_dir / "fold_plans.json").read_text())
    assert len(staged_plans) == 2


def test_stage_training_bundle_raises_when_a_referenced_piece_has_no_grade(tmp_path):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    (midi_dir / "a.mid").write_bytes(b"x")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    plans = [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())]
    paths = BundleSources(
        midi_dir=midi_dir, grades={}, repo_snapshot_dir=repo_snapshot_dir
    )

    with pytest.raises(ValueError, match="no grade"):
        stage_training_bundle(paths, plans, tmp_path / "staging")


def test_stage_training_bundle_raises_when_a_referenced_piece_has_no_midi_on_disk(
    tmp_path,
):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()  # empty -- "a.mid" is never written
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    plans = [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())]
    paths = BundleSources(
        midi_dir=midi_dir, grades={"a": 3}, repo_snapshot_dir=repo_snapshot_dir
    )

    with pytest.raises(FileNotFoundError, match="a.mid"):
        stage_training_bundle(paths, plans, tmp_path / "staging")
