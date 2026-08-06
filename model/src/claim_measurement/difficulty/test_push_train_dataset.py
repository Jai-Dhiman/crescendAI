"""Tests for push_train_dataset (#149 / #138 Phase 1) -- hermetic HF Jobs
training-bundle staging + upload, uploader injected so no test hits the network.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json
from pathlib import Path

import pytest

from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
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


def _write_fake_features37(features37_dir, seg_ids):
    """The per-piece features37 .npz files ft_eval.py reads, and which the
    staged eval_manifest.json must take its row order, grades and composer
    ids from. Written out of order on purpose: the manifest's order comes
    from the sorted filenames, not from insertion order."""
    import numpy as np

    from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz

    for i, seg_id in enumerate(reversed(list(seg_ids))):
        write_embedding_npz(features37_dir / f"{seg_id}.npz",
                            {"raw37": np.arange(37, dtype=np.float32) + i},
                            grade=i % 11, composer_id=i)


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
        eval_manifest=[{"seg_id": "a", "grade": 0, "composer_id": 0}],
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
        midi_dir=midi_dir, grades={}, repo_snapshot_dir=repo_snapshot_dir,
        eval_manifest=[]
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
        midi_dir=midi_dir, grades={"a": 3}, repo_snapshot_dir=repo_snapshot_dir,
        eval_manifest=[]
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
    features37_dir = tmp_path / "features37"
    features37_dir.mkdir()
    _write_fake_features37(features37_dir, seg_ids[:6])

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
            "--features37-dir", str(features37_dir),
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
    features37_dir = tmp_path / "features37"
    features37_dir.mkdir()
    _write_fake_features37(features37_dir, seg_ids[:6])

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
            "--features37-dir", str(features37_dir),
                "--n-folds", "1",
            ],
            uploader=fake_uploader,
        )

    assert not calls
    assert not staging_dir.exists()


def test_build_eval_manifest_row_order_matches_ft_evals_features37_seg_ids(tmp_path):
    """ft_eval.py refuses any emb_fold{F}.npz whose seg_ids do not match
    _load_features37's order, and train_fold.py emits its rows in the staged
    eval_manifest's order -- so if these two orders can ever disagree, the
    whole gate is unpaired. Proven against _load_features37 itself, not
    against a second sorted() that happens to agree."""
    from claim_measurement.difficulty.ft_eval import _load_features37
    from claim_measurement.difficulty.push_train_dataset import build_eval_manifest

    emb_root = tmp_path / "bakeoff"
    features37_dir = emb_root / "emb" / "features37"
    features37_dir.mkdir(parents=True)
    seg_ids = ["p003", "p001", "p010", "p002"]
    _write_fake_features37(features37_dir, seg_ids)

    manifest = build_eval_manifest(features37_dir)
    _, y, composers, loaded_seg_ids = _load_features37(emb_root)

    assert [m["seg_id"] for m in manifest] == loaded_seg_ids
    assert [m["grade"] for m in manifest] == list(y)
    assert [m["composer_id"] for m in manifest] == list(composers)


def test_stage_training_bundle_writes_the_eval_manifest_it_was_given(tmp_path):
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    (midi_dir / "a.mid").write_bytes(b"x")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    eval_manifest = [{"seg_id": "e0", "grade": 4, "composer_id": 2},
                     {"seg_id": "e1", "grade": 7, "composer_id": 3}]
    paths = BundleSources(
        midi_dir=midi_dir, grades={"a": 1}, repo_snapshot_dir=repo_snapshot_dir,
        eval_manifest=eval_manifest)
    staging_dir = tmp_path / "staging"

    report = stage_training_bundle(
        paths,
        [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())],
        staging_dir)

    assert report.n_eval_pieces == 2
    assert json.loads(
        (staging_dir / "eval_manifest.json").read_text()) == eval_manifest


def test_stage_training_bundle_excludes_git_history_and_pycache_from_the_snapshot(
    tmp_path,
):
    """The first real bundle was 119 MB, almost all of it the fork's .git
    history, and it shipped 16 __pycache__ dirs of stale bytecode. Neither is
    reachable from train_fold.py."""
    midi_dir = tmp_path / "transkun_mid"
    midi_dir.mkdir()
    (midi_dir / "a.mid").write_bytes(b"x")
    repo_snapshot_dir = tmp_path / "repo"
    _write_fake_repo(repo_snapshot_dir)
    (repo_snapshot_dir / ".git").mkdir()
    (repo_snapshot_dir / ".git" / "objects").mkdir()
    (repo_snapshot_dir / ".git" / "objects" / "pack").write_bytes(b"history")
    (repo_snapshot_dir / "src" / "__pycache__").mkdir()
    (repo_snapshot_dir / "src" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"x")
    (repo_snapshot_dir / "src" / "loose.pyc").write_bytes(b"x")
    paths = BundleSources(
        midi_dir=midi_dir, grades={"a": 1}, repo_snapshot_dir=repo_snapshot_dir,
        eval_manifest=[])
    staging_dir = tmp_path / "staging"

    report = stage_training_bundle(
        paths,
        [FoldPlan(fold=0, test_seg_ids=("a",), train_seg_ids=(), val_seg_ids=())],
        staging_dir)

    repo_out = staging_dir / "moonbeam_repo"
    assert not (repo_out / ".git").exists()
    assert not (repo_out / "src" / "__pycache__").exists()
    assert not (repo_out / "src" / "loose.pyc").exists()
    # only the two real files _write_fake_repo staged
    assert report.repo_snapshot_files == 2
    assert sorted(p.name for p in repo_out.rglob("*") if p.is_file()) == [
        "README.md", "model_config.json"]


# --------------------------------------------------------------------------
# #166: the submission plan and head_manifest.json ride in the SAME bundle as
# the 5 CV plans, so one upload serves both the fold jobs and the all-data job.
# --------------------------------------------------------------------------


def test_head_manifest_covers_every_pool_piece_in_seg_id_order():
    """The submission model's ridge head is fit on these rows. Missing pieces
    would silently shrink the head's training set, which is the one thing this
    manifest exists to prevent."""
    from claim_measurement.difficulty.push_train_dataset import build_head_manifest

    pool = [ManifestEntry(seg_id=f"p{i:03d}", key=f"k{i}", grade=i % 11,
                          composer=f"composer_{i % 7}") for i in range(40)]

    manifest = build_head_manifest(pool)

    assert len(manifest) == 40
    assert [m["seg_id"] for m in manifest] == sorted(e.seg_id for e in pool)
    assert {m["grade"] for m in manifest} == {e.grade for e in pool}


def test_head_manifest_composer_ids_are_stable_and_group_by_composer():
    """Bookkeeping only -- nothing fits a fold on these, because the submission
    model has no folds. Still asserted so two pieces by one composer cannot end
    up with different ids in the emitted npz."""
    from claim_measurement.difficulty.push_train_dataset import build_head_manifest

    pool = [ManifestEntry(seg_id=f"p{i:03d}", key=f"k{i}", grade=1,
                          composer=f"composer_{i % 3}") for i in range(9)]

    manifest = build_head_manifest(pool)

    by_seg = {m["seg_id"]: m["composer_id"] for m in manifest}
    composer_of = {e.seg_id: e.composer for e in pool}
    groups = {}
    for seg_id, cid in by_seg.items():
        groups.setdefault(composer_of[seg_id], set()).add(cid)
    assert all(len(ids) == 1 for ids in groups.values())
    assert build_head_manifest(pool) == manifest  # deterministic


def test_staging_refuses_a_head_manifest_piece_with_no_midi(tmp_path):
    """A head-manifest piece whose MIDI was never staged could never be
    embedded inside the job container, so the head would quietly be fit on
    fewer rows than the report claims."""
    from claim_measurement.difficulty.push_train_dataset import (
        BundleSources,
        stage_training_bundle,
    )

    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    for seg_id in ("a", "b"):
        (midi_dir / f"{seg_id}.mid").write_bytes(b"MThd")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("")
    plans = [FoldPlan(fold=0, test_seg_ids=(), train_seg_ids=("a", "b"),
                      val_seg_ids=())]
    sources = BundleSources(
        midi_dir=midi_dir, grades={"a": 1, "b": 2}, repo_snapshot_dir=repo,
        eval_manifest=[],
        head_manifest=[{"seg_id": "a", "grade": 1, "composer_id": 0},
                       {"seg_id": "ghost", "grade": 3, "composer_id": 1}])

    with pytest.raises(ValueError, match="no staged MIDI"):
        stage_training_bundle(sources, plans, tmp_path / "staging")


def test_a_bundle_without_a_head_manifest_does_not_write_the_file(tmp_path):
    """The #149 fold jobs read a bundle with no head_manifest.json; staging one
    unconditionally would change the bundle those jobs are pinned to."""
    from claim_measurement.difficulty.push_train_dataset import (
        BundleSources,
        stage_training_bundle,
    )

    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    (midi_dir / "a.mid").write_bytes(b"MThd")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("")
    sources = BundleSources(
        midi_dir=midi_dir, grades={"a": 1}, repo_snapshot_dir=repo,
        eval_manifest=[])

    staging = tmp_path / "staging"
    report = stage_training_bundle(
        sources, [FoldPlan(fold=0, test_seg_ids=(), train_seg_ids=("a",),
                           val_seg_ids=())], staging)

    assert not (staging / "head_manifest.json").exists()
    assert report.n_head_pieces == 0
