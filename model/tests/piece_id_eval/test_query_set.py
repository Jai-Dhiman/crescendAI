"""Verify QuerySet.load builds correct labeled windows from fixture data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from piece_id_eval.query_set import LabeledQueryWindow, LoadResult, QuerySet


def _write_fixture_wav(path: Path, duration_sec: float = 4.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.random.RandomState(42).randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    sf.write(path, y, sr)


def _fixture_candidates_yaml(tmp_path: Path, slug: str, video_id: str) -> Path:
    yaml_path = tmp_path / "practice_eval" / slug / "candidates.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"""\
piece: {slug}
title: Test Piece
composer: Test
recordings:
- video_id: {video_id}
  title: Test Recording
  channel: Test Channel
  duration_seconds: 4
  view_count: 100
  url: https://youtube.com/watch?v={video_id}
  query_source: test
  approved: true
  review_notes: ''
- video_id: unapproved123
  title: Unapproved Recording
  channel: Test Channel
  duration_seconds: 4
  view_count: 10
  url: https://youtube.com/watch?v=unapproved123
  query_source: test
  approved: false
  review_notes: ''
""")
    return yaml_path


def _fixture_piece_map(tmp_path: Path, slug: str, piece_id: str) -> Path:
    pm = tmp_path / "evals" / "piece_id" / "eval_piece_map.json"
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps({slug: piece_id}))
    return pm


def test_load_returns_windows_for_cached_approved_recordings(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert isinstance(result, LoadResult)
    assert len(result.windows) >= 1
    for w in result.windows:
        assert isinstance(w, LabeledQueryWindow)
        assert w.piece_id == piece_id
        assert w.slug == slug
        assert w.is_in_catalog is True
        assert w.chroma.shape[0] == 12


def test_unapproved_recordings_excluded(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")
    # Note: unapproved123.wav is NOT written; it should not be loaded

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    # Only windows from approved recording should appear
    for w in result.windows:
        assert w.video_id == video_id


def test_holdout_slug_tagged_not_in_catalog(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[slug],  # held out
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert all(w.is_in_catalog is False for w in result.windows)


def test_missing_audio_counts_as_excluded(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    # No audio file written

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert len(result.windows) == 0
    assert result.excluded_count >= 1
