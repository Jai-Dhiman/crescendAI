# apps/evals/tests/test_run_eval_split_flag.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teaching_knowledge.run_eval import _filter_cache_files_by_split


def _fake_cache_file(tmp: Path, name: str) -> Path:
    p = tmp / f"{name}.json"
    p.write_text("{}")
    return p


def test_filter_returns_only_train_set(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a", "b", "c"], "holdout": ["d", "e"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c", "d", "e"]]

    filtered = _filter_cache_files_by_split(files, splits, which="train")
    names = sorted(f.stem for f in filtered)
    assert names == ["a", "b", "c"]


def test_filter_returns_only_holdout_set(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a", "b"], "holdout": ["c", "d"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c", "d"]]
    filtered = _filter_cache_files_by_split(files, splits, which="holdout")
    assert sorted(f.stem for f in filtered) == ["c", "d"]


def test_filter_all_returns_everything(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["a"], "holdout": ["b"]}))
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b", "c"]]
    # "c" is not in the split; "all" returns only items present in the split
    filtered = _filter_cache_files_by_split(files, splits, which="all")
    assert sorted(f.stem for f in filtered) == ["a", "b"]


def test_filter_with_no_split_path_returns_all_files(tmp_path: Path) -> None:
    files = [_fake_cache_file(tmp_path, n) for n in ["a", "b"]]
    filtered = _filter_cache_files_by_split(files, None, which="all")
    assert sorted(f.stem for f in filtered) == ["a", "b"]


def test_filter_rejects_invalid_which(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "holdout": []}))
    with pytest.raises(ValueError):
        _filter_cache_files_by_split([], splits, which="nonsense")
