"""Tests for the catalog coverage / quality acceptance harness."""

from __future__ import annotations

import json
from pathlib import Path

from score_library.catalog_coverage import CANONICAL_MAP, check_coverage


def _write_score(path: Path, piece_id: str, bars: list[list[float]]) -> None:
    """Write a minimal schema-shaped score JSON.

    bars: list of bars, each a list of onset_seconds for that bar's notes.
    """
    bar_models = []
    for i, onsets in enumerate(bars, start=1):
        notes = [
            {
                "pitch": 60,
                "pitch_name": "C4",
                "velocity": 80,
                "onset_tick": int(o * 1000),
                "onset_seconds": o,
                "duration_ticks": 240,
                "duration_seconds": 0.25,
                "track": 0,
            }
            for o in onsets
        ]
        bar_models.append(
            {
                "bar_number": i,
                "start_tick": (i - 1) * 1920,
                "start_seconds": float(i - 1),
                "time_signature": "4/4",
                "notes": notes,
                "pedal_events": [],
                "note_count": len(notes),
                "pitch_range": [60, 60] if notes else [],
                "mean_velocity": 80 if notes else 0,
            }
        )
    data = {
        "piece_id": piece_id,
        "composer": "Test",
        "title": "Test",
        "key_signature": None,
        "time_signatures": [{"tick": 0, "numerator": 4, "denominator": 4}],
        "tempo_markings": [],
        "total_bars": len(bar_models),
        "bars": bar_models,
    }
    path.write_text(json.dumps(data))


def _good_onsets() -> list[list[float]]:
    """3 bars, 24 monotonic onsets total (>= 20)."""
    return [
        [i * 0.25 for i in range(8)],
        [2.0 + i * 0.25 for i in range(8)],
        [4.0 + i * 0.25 for i in range(8)],
    ]


class TestCheckCoverage:
    def test_canonical_map_has_16_entries(self) -> None:
        assert len(CANONICAL_MAP) == 16

    def test_all_present_and_good_returns_empty(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "good.piece"}
        _write_score(tmp_path / "good.piece.json", "good.piece", _good_onsets())
        assert check_coverage(tmp_path, mapping) == []

    def test_missing_file_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "absent.piece"}
        result = check_coverage(tmp_path, mapping)
        assert result == ["slug_a: MISSING absent.piece.json"]

    def test_too_few_notes_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "thin.piece"}
        _write_score(tmp_path / "thin.piece.json", "thin.piece", [[0.0, 0.25, 0.5]])
        result = check_coverage(tmp_path, mapping)
        assert len(result) == 1
        assert "thin.piece" in result[0]
        assert "20" in result[0]

    def test_zero_bars_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "empty.piece"}
        _write_score(tmp_path / "empty.piece.json", "empty.piece", [])
        result = check_coverage(tmp_path, mapping)
        assert any("total_bars" in r for r in result)

    def test_non_monotonic_onsets_reported(self, tmp_path: Path) -> None:
        mapping = {"slug_a": "jumbled.piece"}
        # 24 notes but bar 2 onsets dip below bar 1's last -> non-monotonic flat list
        bars = [
            [i * 0.25 for i in range(8)],   # 0.0 .. 1.75
            [0.5 + i * 0.25 for i in range(8)],  # restarts at 0.5 < 1.75
            [4.0 + i * 0.25 for i in range(8)],
        ]
        _write_score(tmp_path / "jumbled.piece.json", "jumbled.piece", bars)
        result = check_coverage(tmp_path, mapping)
        assert any("monotonic" in r for r in result)
