"""Unit tests for FRONT 7b offline score alignment (align_notes.py).

Hermetic: no parangonar, no AMT, no score files. The parangonar seam (_match) and
score loader are exercised only through align_bundle_file with monkeypatched
module attributes.

Run: cd model && uv run python -m pytest src/claim_measurement/score_align/ -q --no-cov
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from claim_measurement.score_align.align_notes import (
    SCORE_ALIGN_SCHEMA,
    ScoreAlignError,
    align_bundle_file,
    annotate_bundle,
    bar_number_for_score_sec,
    fit_affine,
    matched_note_pairs,
)


def _score_na(onsets: list[float]) -> np.ndarray:
    dtype = [("onset_sec", float), ("duration_sec", float), ("pitch", int), ("id", "U32")]
    arr = np.empty(len(onsets), dtype=dtype)
    for i, o in enumerate(onsets):
        arr[i] = (o, 0.5, 60 + i % 12, f"s{i}")
    return arr


def _match_entries(pairs: list[tuple[int, int]]) -> list[dict]:
    return [
        {"label": "match", "score_id": f"s{s}", "performance_id": f"p{p}"}
        for p, s in pairs
    ]


# --- fit_affine ---------------------------------------------------------------


def test_fit_affine_recovers_known_transform():
    score = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    perf = 1.8 * score + 0.7
    a, b = fit_affine(perf, score)
    assert a == pytest.approx(1.8, abs=1e-9)
    assert b == pytest.approx(0.7, abs=1e-9)


def test_fit_affine_raises_on_degenerate_input():
    with pytest.raises(ScoreAlignError, match="distinct"):
        fit_affine(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ScoreAlignError, match="distinct"):
        fit_affine(np.array([1.0, 2.0]), np.array([3.0, 3.0]))


def test_fit_affine_rejects_nonpositive_tempo_ratio():
    score = np.array([0.0, 1.0, 2.0])
    perf = np.array([4.0, 3.0, 2.0])  # time running backwards
    with pytest.raises(ScoreAlignError, match="tempo ratio"):
        fit_affine(perf, score)


# --- matched_note_pairs --------------------------------------------------------


def test_matched_pairs_keeps_per_note_correspondence():
    score_na = _score_na([0.0, 1.0, 2.0])
    matches = _match_entries([(0, 0), (2, 1)]) + [
        {"label": "insertion", "performance_id": "p1"},
        {"label": "deletion", "score_id": "s2"},
    ]
    pairs = matched_note_pairs(score_na, matches, n_perf_notes=3)
    assert pairs == [(0, 0), (2, 1)]


def test_matched_pairs_skips_unknown_score_id_keeps_first_duplicate():
    score_na = _score_na([0.0, 1.0])
    matches = [
        {"label": "match", "score_id": "s_missing", "performance_id": "p0"},
        {"label": "match", "score_id": "s0", "performance_id": "p1"},
        {"label": "match", "score_id": "s1", "performance_id": "p1"},  # dup perf note
    ]
    pairs = matched_note_pairs(score_na, matches, n_perf_notes=2)
    assert pairs == [(1, 0)]


def test_matched_pairs_raises_on_zero_matches():
    score_na = _score_na([0.0])
    with pytest.raises(ScoreAlignError, match="zero note matches"):
        matched_note_pairs(score_na, [{"label": "insertion"}], n_perf_notes=1)


def test_matched_pairs_raises_on_out_of_range_perf_index():
    score_na = _score_na([0.0])
    matches = [{"label": "match", "score_id": "s0", "performance_id": "p99"}]
    with pytest.raises(ScoreAlignError, match="out of range"):
        matched_note_pairs(score_na, matches, n_perf_notes=3)


# --- bar_number_for_score_sec --------------------------------------------------

MEASURE_TABLE = [
    {"bar_number": 1, "start_sec": 0.0, "start_tick": 0},
    {"bar_number": 2, "start_sec": 2.0, "start_tick": 8},
    {"bar_number": 3, "start_sec": 4.0, "start_tick": 16},
]


def test_bar_number_lookup():
    assert bar_number_for_score_sec(MEASURE_TABLE, 0.0) == 1
    assert bar_number_for_score_sec(MEASURE_TABLE, 2.5) == 2
    assert bar_number_for_score_sec(MEASURE_TABLE, 4.1) == 3


def test_bar_number_clamps_before_first_bar():
    # AMT onset jitter can put a matched note's score onset epsilon-before 0.
    assert bar_number_for_score_sec(MEASURE_TABLE, -0.01) == 1


# --- annotate_bundle ------------------------------------------------------------


def _bundle_with_notes(perf_onsets: list[float]) -> dict:
    return {
        "piece_id": "test_piece",
        "video_id": "vid",
        "notes": [
            {"onset": o, "offset": o + 0.2, "pitch": 60, "velocity": 64}
            for o in perf_onsets
        ],
        "measure_table": MEASURE_TABLE,
    }


def test_annotate_score_onset_carries_rush_residual():
    # 22 clean notes on a known affine (a=2, b=1) + one note rushed by 50ms.
    # With 21 clean anchors the LSQ absorbs only a few ms of the injected rush.
    score_onsets = [0.25 * i for i in range(22)]
    perf_onsets = [2.0 * s + 1.0 for s in score_onsets]
    rushed_idx = 10
    perf_onsets[rushed_idx] -= 0.050
    bundle = _bundle_with_notes(perf_onsets)
    score_na = _score_na(score_onsets)
    matches = _match_entries([(i, i) for i in range(22)])

    annotate_bundle(bundle, score_na, matches)

    notes = bundle["notes"]
    rushed_residual = notes[rushed_idx]["onset"] - notes[rushed_idx]["score_onset"]
    assert rushed_residual < -0.040  # rush survives the detrend, sign preserved
    clean_residuals = [
        n["onset"] - n["score_onset"] for i, n in enumerate(notes) if i != rushed_idx
    ]
    assert max(abs(r) for r in clean_residuals) < 0.010


def test_annotate_sets_bar_numbers_from_score_seconds():
    score_onsets = [0.0, 1.0, 2.5, 4.5]
    perf_onsets = [1.5 * s + 0.3 for s in score_onsets]
    bundle = _bundle_with_notes(perf_onsets)
    annotate_bundle(bundle, _score_na(score_onsets), _match_entries([(i, i) for i in range(4)]))
    assert [n["bar_number"] for n in bundle["notes"]] == [1, 1, 2, 3]


def test_annotate_leaves_unmatched_notes_bare_and_strips_stale_fields():
    score_onsets = [0.0, 1.0, 2.0]
    perf_onsets = [2.0 * s for s in score_onsets]
    bundle = _bundle_with_notes(perf_onsets)
    bundle["notes"][2]["score_onset"] = 123.0  # stale from a previous run
    bundle["notes"][2]["bar_number"] = 99
    matches = _match_entries([(0, 0), (1, 1)])  # note 2 unmatched this time

    annotate_bundle(bundle, _score_na(score_onsets), matches)

    assert "score_onset" not in bundle["notes"][2]
    assert "bar_number" not in bundle["notes"][2]
    assert "score_onset" in bundle["notes"][0]


def test_annotate_writes_metadata_block():
    score_onsets = [0.0, 1.0, 2.0, 3.0]
    perf_onsets = [2.0 * s + 1.0 for s in score_onsets]
    bundle = _bundle_with_notes(perf_onsets)
    annotate_bundle(bundle, _score_na(score_onsets), _match_entries([(i, i) for i in range(4)]))

    meta = bundle["score_align"]
    assert meta["schema"] == SCORE_ALIGN_SCHEMA
    assert meta["affine_a"] == pytest.approx(2.0)
    assert meta["affine_b"] == pytest.approx(1.0)
    assert meta["n_matched"] == 4
    assert meta["n_notes"] == 4
    assert meta["residual_rms_ms"] == pytest.approx(0.0, abs=1e-6)


# --- align_cli._score_map --------------------------------------------------------


def test_score_map_rebase_keeps_filenames():
    from pathlib import Path

    from claim_measurement.score_align.align_cli import _score_map

    default = _score_map(None)
    rebased = _score_map(Path("/elsewhere/scores"))
    assert set(default) == set(rebased)
    for piece, p in rebased.items():
        assert p.parent == Path("/elsewhere/scores")
        assert p.name == default[piece].name


# --- align_bundle_file (I/O seam; loader + matcher monkeypatched) ---------------


def test_align_bundle_file_rewrites_bundle(tmp_path, monkeypatch):
    score_onsets = [0.0, 1.0, 2.0, 3.0]
    perf_onsets = [2.0 * s + 1.0 for s in score_onsets]
    bundle = _bundle_with_notes(perf_onsets)
    bundle_path = tmp_path / "vid.json"
    bundle_path.write_text(json.dumps(bundle))
    score_path = tmp_path / "score.json"
    score_path.write_text("{}")  # never parsed; loader is stubbed

    score_na = _score_na(score_onsets)

    import claim_measurement.score_align.align_notes as mod

    monkeypatch.setattr(
        mod, "_load_bach_json_score",
        lambda p: (score_na, MEASURE_TABLE, "sha", 0.5),
    )
    captured: dict = {}

    def fake_match(s_na, p_na):
        captured["perf_na"] = p_na
        return _match_entries([(i, i) for i in range(4)])

    monkeypatch.setattr(mod, "_match", fake_match)

    meta = align_bundle_file(bundle_path, score_path)

    assert meta["n_matched"] == 4
    on_disk = json.loads(bundle_path.read_text())
    assert on_disk["score_align"]["affine_a"] == pytest.approx(2.0)
    assert all("score_onset" in n for n in on_disk["notes"])
    # perf note array must mirror extractor.py: beat_sec from the score loader,
    # so onset_beat == onset_sec / 0.5
    p_na = captured["perf_na"]
    assert float(p_na[0]["onset_beat"]) == pytest.approx(float(p_na[0]["onset_sec"]) / 0.5)
