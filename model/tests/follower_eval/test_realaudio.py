# model/tests/follower_eval/test_realaudio.py
"""Unit tests for the real-audio eval's OWN proxy logic (issue #133). The HMM
matcher itself is covered by follower_bench tests; here we pin the anchor-free
statistics, the loud loaders, and bundle discovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from follower_eval import realaudio as ra


def test_mono_stats_all_forward():
    steps, frac, max_back = ra._mono_stats([0.0, 1.0, 2.0, 3.0])
    assert steps == 0 and frac == 0.0 and max_back == 0.0


def test_mono_stats_counts_backward_beyond_tolerance():
    # deltas of [0,5,3,3.3,3.0] are +5, -2, +0.3, -0.3; only -2 exceeds the
    # -0.5s tolerance, so the 0.3s chord-noise wobble is correctly ignored.
    steps, frac, max_back = ra._mono_stats([0.0, 5.0, 3.0, 3.3, 3.0])
    assert steps == 1
    assert frac == pytest.approx(0.25)   # 1 of 4 deltas
    assert max_back == pytest.approx(2.0)


def test_mono_stats_short_input():
    assert ra._mono_stats([]) == (0, 0.0, 0.0)
    assert ra._mono_stats([1.0]) == (0, 0.0, 0.0)


def test_spearman_perfect_and_degenerate():
    assert ra._spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert ra._spearman([1.0, 2.0], [1.0, 2.0]) is None          # n<3
    assert ra._spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None  # zero variance


def test_load_bundle_notes_loud_on_empty(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"notes": []}))
    with pytest.raises(ra.RealAudioEvalError, match="no 'notes'"):
        ra.load_bundle_notes(p)


def test_load_bundle_notes_sorts_by_onset(tmp_path: Path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"notes": [
        {"onset": 2.0, "offset": 2.5, "pitch": 60, "velocity": 50},
        {"onset": 0.5, "offset": 1.0, "pitch": 62, "velocity": 40},
    ]}))
    notes = ra.load_bundle_notes(p)
    assert [n.onset for n in notes] == [0.5, 2.0]
    assert notes[0].pitch == 62


def test_discover_bundles_filters_meta_and_unknown(tmp_path: Path):
    # a known piece with a real bundle + a meta sidecar, and an unknown piece dir
    known = tmp_path / "bach_invention_1"
    known.mkdir()
    (known / "vid1.json").write_text("{}")
    (known / "vid1.meta.json").write_text("{}")
    (known / "_index.json").write_text("{}")
    unknown = tmp_path / "not_a_rep_piece"
    unknown.mkdir()
    (unknown / "x.json").write_text("{}")

    found = ra.discover_bundles(tmp_path)
    assert set(found) == {"bach_invention_1"}
    assert [p.name for p in found["bach_invention_1"]] == ["vid1.json"]


def test_score_filename_map_covers_rep():
    assert ra.SCORE_FILENAME_BY_PIECE["fur_elise"] == "beethoven.fur_elise.json"
    assert len(ra.SCORE_FILENAME_BY_PIECE) == 16
