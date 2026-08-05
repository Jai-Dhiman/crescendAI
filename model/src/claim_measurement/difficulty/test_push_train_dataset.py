"""Tests for push_train_dataset (#149 / #138 Phase 1) -- hermetic HF Jobs
training-bundle staging + upload, uploader injected so no test hits the network.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.fold_plan import FoldPlan
from claim_measurement.difficulty.push_train_dataset import (
    BundleSources,
    main,
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
    assert report.code_files == 2
    assert len(report.checksum) == 64  # sha256 hex digest
    for seg_id in seg_ids:
        assert (staging_dir / "midi" / f"{seg_id}.mid").exists()
    assert json.loads((staging_dir / "grades.json").read_text()) == {
        s: i for i, s in enumerate(seg_ids)
    }
    staged_plans = json.loads((staging_dir / "fold_plans.json").read_text())
    assert len(staged_plans) == 2
    # code/ carries train_fold.py's runtime deps into the HF Jobs container --
    # `hf jobs uv run` only uploads the one file passed on the command line.
    assert (staging_dir / "code" / "ranking_loss.py").read_text() == (
        Path(__file__).resolve().parent / "ranking_loss.py").read_text()
    assert (staging_dir / "code" / "bakeoff_cv.py").read_text() == (
        Path(__file__).resolve().parent / "bakeoff_cv.py").read_text()


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

    with pytest.raises(FileNotFoundError, match="fold plan references"):
        stage_training_bundle(paths, plans, tmp_path / "staging")


def test_main_builds_fold_plans_stages_and_calls_the_injected_uploader(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "labels.json"
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()

    seg_ids = [f"p{i:02d}" for i in range(10)]
    manifest = [
        {
            "seg_id": s,
            "key": f"{s}.mid",
            "grade": i % 11,
            "video_id": "x",
            "midi_name": f"mid/{s}.mid",
        }
        for i, s in enumerate(seg_ids)
    ]
    labels = {f"{s}.mid": {"composer": f"composer_{i}"} for i, s in enumerate(seg_ids)}
    manifest_path.write_text(json.dumps(manifest))
    labels_path.write_text(json.dumps(labels))
    for s in seg_ids:
        (midi_dir / f"{s}.mid").write_bytes(b"x")

    sample_manifest_path = tmp_path / "sample_manifest.json"
    sample_manifest_path.write_text(json.dumps([{"seg_id": s} for s in seg_ids[:6]]))

    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    staging_dir = tmp_path / "staging"

    calls = []

    def fake_uploader(staged_dir, repo_id):
        calls.append((staged_dir, repo_id))

    exit_code = main(
        [
            "--manifest", str(manifest_path),
            "--labels", str(labels_path),
            "--sample-manifest", str(sample_manifest_path),
            "--midi-dir", str(midi_dir),
            "--repo-snapshot-dir", str(repo_snapshot_dir),
            "--staging-dir", str(staging_dir),
            "--repo-id", "jaidhiman/phase1-lora-bundle",
            "--n-folds", "2",
        ],
        uploader=fake_uploader,
    )

    assert exit_code == 0
    assert calls == [(staging_dir, "jaidhiman/phase1-lora-bundle")]
    staged_plans = json.loads((staging_dir / "fold_plans.json").read_text())
    assert len(staged_plans) == 2


def test_main_refuses_to_stage_when_check_fold_plans_reports_a_violation(
    tmp_path, monkeypatch
):
    """check_fold_plans is the independent re-derivation that catches a leak
    in the artifact actually uploaded and trained on -- main() must call it
    and refuse to stage/upload on any violation, rather than leaving it dead."""
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "labels.json"
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()

    seg_ids = [f"p{i:02d}" for i in range(10)]
    manifest = [
        {
            "seg_id": s,
            "key": f"{s}.mid",
            "grade": i % 11,
            "video_id": "x",
            "midi_name": f"mid/{s}.mid",
        }
        for i, s in enumerate(seg_ids)
    ]
    labels = {f"{s}.mid": {"composer": f"composer_{i}"} for i, s in enumerate(seg_ids)}
    manifest_path.write_text(json.dumps(manifest))
    labels_path.write_text(json.dumps(labels))
    for s in seg_ids:
        (midi_dir / f"{s}.mid").write_bytes(b"x")

    sample_manifest_path = tmp_path / "sample_manifest.json"
    sample_manifest_path.write_text(json.dumps([{"seg_id": s} for s in seg_ids[:6]]))

    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    staging_dir = tmp_path / "staging"

    def corrupt_build_fold_plans(eval_entries, pool_entries, n_folds, seed, val_frac):
        # A deliberately corrupted plan: fold 0's train set overlaps its own
        # test set, which check_fold_plans must catch.
        seg_id = eval_entries[0].seg_id
        return [
            FoldPlan(fold=0, test_seg_ids=(seg_id,), train_seg_ids=(seg_id,),
                     val_seg_ids=())
        ]

    monkeypatch.setattr(
        "claim_measurement.difficulty.fold_plan.build_fold_plans",
        corrupt_build_fold_plans,
    )

    calls = []

    def fake_uploader(staged_dir, repo_id):
        calls.append((staged_dir, repo_id))

    with pytest.raises(ValueError, match="failed independent re-derivation"):
        main(
            [
                "--manifest", str(manifest_path),
                "--labels", str(labels_path),
                "--sample-manifest", str(sample_manifest_path),
                "--midi-dir", str(midi_dir),
                "--repo-snapshot-dir", str(repo_snapshot_dir),
                "--staging-dir", str(staging_dir),
                "--repo-id", "jaidhiman/phase1-lora-bundle",
                "--n-folds", "1",
            ],
            uploader=fake_uploader,
        )

    assert not calls
    assert not staging_dir.exists()
