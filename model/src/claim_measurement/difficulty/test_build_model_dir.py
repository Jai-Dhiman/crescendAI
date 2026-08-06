"""Tests for build_model_dir (#166 / #104 S1). CPU-only, offline, no adapter
weights beyond a few stub files.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np

from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds
from claim_measurement.difficulty.build_model_dir import (
    build_model_dir,
    fold_test_seg_ids,
)
from claim_measurement.difficulty.score_wav import (
    ADAPTER_SUBDIR,
    HEAD_FILENAME,
    MANIFEST_FILENAME,
    fit_head_from_fold_embeddings,
    read_ridge_head,
)
from claim_measurement.difficulty.train_fold import write_fold_embeddings


def _fold_npz(path, n=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    seg_ids = [f"p{i:03d}" for i in range(n)]
    embeddings = rng.normal(size=(n, dim)).astype(np.float32)
    grades = np.array([i % 11 for i in range(n)])
    composer_ids = np.array([i % 17 for i in range(n)])
    write_fold_embeddings(path, seg_ids, embeddings, grades, composer_ids)
    return seg_ids, grades, composer_ids


def test_fold_test_seg_ids_matches_the_split_every_measurement_used():
    """A drifted split here would fit the head on rows this adapter never
    trained on -- the head would look better than the system it is deployed
    into, and nothing would flag it."""
    seg_ids = [f"p{i:03d}" for i in range(60)]
    composer_ids = np.array([i % 17 for i in range(60)])

    held_out = fold_test_seg_ids(seg_ids, composer_ids, fold=2)

    expected = [seg_ids[i] for i in composer_disjoint_folds(composer_ids, 5, 2026)[2]]
    assert held_out == expected
    assert held_out  # a fold that holds nothing out would silently pass above


def test_the_head_from_a_fold_artifact_excludes_that_folds_test_pieces(tmp_path):
    """The per-fold adapter never saw its own test pieces; a head fit on them
    would be a head for a model that does not exist."""
    path = tmp_path / "emb_fold0.npz"
    seg_ids, grades, composer_ids = _fold_npz(path)
    held_out = fold_test_seg_ids(seg_ids, composer_ids, fold=0)

    head = fit_head_from_fold_embeddings(path, exclude_seg_ids=held_out)

    kept = [g for s, g in zip(seg_ids, grades) if s not in set(held_out)]
    assert head.fallback_score == float(np.median(kept))
    assert head.n_features == 8


def test_no_exclusions_fits_on_every_row(tmp_path):
    """What the all-data submission model wants: the MIREX test set is
    disjoint from PSyllabus by construction, so nothing is held out."""
    path = tmp_path / "emb_fold0.npz"
    _seg_ids, grades, _composer_ids = _fold_npz(path)

    head = fit_head_from_fold_embeddings(path, exclude_seg_ids=None)

    assert head.fallback_score == float(np.median(grades))


def test_build_model_dir_copies_the_adapter_rather_than_linking_it(tmp_path):
    """The directory is what goes into a container image; a symlink into the
    research tree resolves to nothing there."""
    emb_path = tmp_path / "emb_fold0.npz"
    _fold_npz(emb_path)
    adapter_src = tmp_path / "src_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text("{}")
    (adapter_src / "adapter_model.safetensors").write_bytes(b"weights")
    head = fit_head_from_fold_embeddings(emb_path)

    out = build_model_dir(tmp_path / "model", adapter_src, head, kind="test")

    adapter_dst = out / ADAPTER_SUBDIR
    assert not adapter_dst.is_symlink()
    assert (adapter_dst / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (out / HEAD_FILENAME).exists()
    assert read_ridge_head(out / HEAD_FILENAME).n_features == head.n_features


def test_the_manifest_records_provenance(tmp_path):
    """MIREX 2026 adds training-data size, model size, and compute to the
    mandatory disclosure, so provenance travels with the artifact rather than
    being reconstructed from a shell history at report-writing time."""
    emb_path = tmp_path / "emb_fold0.npz"
    _fold_npz(emb_path)
    adapter_src = tmp_path / "src_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text("{}")

    out = build_model_dir(tmp_path / "model", adapter_src,
                          fit_head_from_fold_embeddings(emb_path),
                          kind="per-fold scaffold", fold=0, head_train_rows=48)

    manifest = json.loads((out / MANIFEST_FILENAME).read_text())
    assert manifest["kind"] == "per-fold scaffold"
    assert manifest["fold"] == 0
    assert manifest["head_train_rows"] == 48


def test_rebuilding_over_an_existing_dir_leaves_no_stale_adapter_files(tmp_path):
    """copytree into a populated directory would merge, leaving a previous
    adapter's shards beside the new ones -- peft would load a chimera."""
    emb_path = tmp_path / "emb_fold0.npz"
    _fold_npz(emb_path)
    head = fit_head_from_fold_embeddings(emb_path)
    first_src = tmp_path / "first"
    first_src.mkdir()
    (first_src / "adapter_config.json").write_text("{}")
    (first_src / "stale_shard.safetensors").write_bytes(b"old")
    second_src = tmp_path / "second"
    second_src.mkdir()
    (second_src / "adapter_config.json").write_text("{}")

    build_model_dir(tmp_path / "model", first_src, head, kind="test")
    out = build_model_dir(tmp_path / "model", second_src, head, kind="test")

    assert not (out / ADAPTER_SUBDIR / "stale_shard.safetensors").exists()
