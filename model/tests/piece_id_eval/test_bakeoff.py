# model/tests/piece_id_eval/test_bakeoff.py
"""Integration test: bakeoff.run on a 2-piece synthetic catalog.

Verifies BakeoffReport is populated with recall tables, verdict, and per-matcher
results. CLI smoke test verifies --no-track exits 0.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from piece_id_eval.bakeoff import BakeoffReport, run
from piece_id_eval.notes import Note

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_piece(root: int, n: int = 40) -> list[Note]:
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=root + (i % 7), velocity=80) for i in range(n)]


def _synthetic_catalog() -> dict[str, list[Note]]:
    return {
        "piece_a": _make_piece(60),
        "piece_b": _make_piece(67),
    }


def _synthetic_recordings() -> dict[str, list[Note]]:
    """One recording per piece (same as catalog for self-query test)."""
    return {
        "piece_a": _make_piece(60),
        "piece_b": _make_piece(67),
    }


def test_bakeoff_run_returns_report() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None, 10.0],
        n_starts=2,
        corruption_grid=[
            {"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0},
            {"deletion_rate": 0.3, "insertion_rate": 0.0, "jitter_seconds": 0.0},
        ],
        seed=42,
        no_track=True,
    )
    assert isinstance(report, BakeoffReport)


def test_bakeoff_report_has_recall_table() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None],
        n_starts=1,
        corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
        seed=0,
        no_track=True,
    )
    # recall_table is a dict keyed by (matcher_name, window_label)
    assert len(report.recall_table) > 0
    for key, val in report.recall_table.items():
        matcher_name, window_label = key
        assert isinstance(matcher_name, str)
        assert "recall@1" in val or "recall@10" in val


def test_bakeoff_report_has_verdict() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None],
        n_starts=1,
        corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
        seed=0,
        no_track=True,
    )
    assert report.verdict in {"KILL", "TUNE", "PROCEED"}


def test_bakeoff_open_set_ok_is_true_on_well_separated_catalog() -> None:
    """open_set_ok must be True when catalog pieces have distinct chroma (cosine oracle).

    piece_a (root=60, pc=0 dominant) and piece_b (root=67, pc=7 dominant) are
    clearly separated in cosine chroma space. The NoteChromaMatcher oracle must
    produce in-catalog scores >> LOO scores so a qualifying FA<=0.05/TA>=0.60
    threshold exists. This test FAILS if the open-set oracle is DtwCeilingMatcher
    (scores always <= 0) because then TA=0 at every positive threshold.
    """
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None],
        n_starts=1,
        corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
        seed=0,
        no_track=True,
    )
    assert report.open_set_ok is True, (
        "open_set_ok must be True for a well-separated 2-piece catalog "
        "(fails if DtwCeilingMatcher is used as oracle instead of NoteChromaMatcher)"
    )


def test_bakeoff_cli_smoke(tmp_path: Path) -> None:
    """CLI help exits 0 without crashing."""
    result = subprocess.run(
        [sys.executable, "-m", "piece_id_eval.bakeoff", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "model"),
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_bakeoff_cli_no_track_synthetic(tmp_path: Path) -> None:
    """CLI runs with --no-track and --synthetic-only without needing real data."""
    result = subprocess.run(
        [sys.executable, "-m", "piece_id_eval.bakeoff", "--no-track", "--synthetic-only"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "model"),
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "VERDICT:" in result.stdout
