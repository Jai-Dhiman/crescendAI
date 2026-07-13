"""Verify the score-note loaders (golden-fixture JSON, score MIDI) through
their public interface only, against real committed/on-disk fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from follower_bench.score_notes import load_golden_fixture_notes

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_FIXTURE_PATH = (
    REPO_ROOT / "apps/api/src/wasm/score-analysis/tests/fixtures/bach_inv1_chunk0.json"
)


def test_load_golden_fixture_notes_matches_day0_spike_counts() -> None:
    perf_notes, score_notes = load_golden_fixture_notes(GOLDEN_FIXTURE_PATH)

    assert len(perf_notes) == 82
    assert len(score_notes) == 458

    assert perf_notes[0].onset == pytest.approx(0.70)
    assert perf_notes[-1].onset == pytest.approx(14.92)

    assert score_notes[0].pitch == 60
    assert score_notes[0].position == pytest.approx(0.1875)


def test_load_golden_fixture_notes_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_fixture_notes(Path("/nonexistent/path/does-not-exist.json"))
