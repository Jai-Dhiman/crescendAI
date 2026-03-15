"""Integration test: run full parse pipeline on actual ASAP cache.

Validates that all 242 pieces parse successfully and output conforms
to the Pydantic schema. Also spot-checks bar structure consistency.
"""
import json
from pathlib import Path

import pytest

from score_library.discover import discover_pieces
from score_library.parse import parse_score_midi
from score_library.schema import ScoreData

ASAP_DIR = Path("/Users/jdhiman/Documents/crescendai/model/data/asap_cache")


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
def test_full_pipeline_all_pieces():
    """All 242 ASAP pieces parse without errors."""
    pieces = discover_pieces(ASAP_DIR)
    assert len(pieces) >= 240, f"Expected ~242 pieces, got {len(pieces)}"

    failures = []
    for entry in pieces:
        try:
            result = parse_score_midi(
                entry.score_midi_path, entry.piece_id, entry.composer, entry.title,
            )
            data = result.model_dump()
            assert data["total_bars"] > 0, f"{entry.piece_id}: 0 bars"
            assert len(data["bars"]) == data["total_bars"], f"{entry.piece_id}: bar count mismatch"

            # JSON size check
            json_str = json.dumps(data)
            size_kb = len(json_str) / 1024
            assert size_kb < 3000, f"{entry.piece_id}: JSON too large ({size_kb:.0f}KB)"

        except Exception as e:
            failures.append((entry.piece_id, str(e)))

    if failures:
        msg = f"{len(failures)} pieces failed:\n"
        for pid, err in failures:
            msg += f"  {pid}: {err}\n"
        pytest.fail(msg)


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
def test_spot_check_bar_consistency():
    """Spot-check: bars have valid structure across 20 pieces."""
    pieces = discover_pieces(ASAP_DIR)

    for entry in pieces[:20]:
        result = parse_score_midi(
            entry.score_midi_path, entry.piece_id, entry.composer, entry.title,
        )
        prev_tick = -1
        for bar in result.bars:
            assert bar.start_tick >= prev_tick, (
                f"{entry.piece_id} bar {bar.bar_number}: "
                f"start_tick {bar.start_tick} < prev {prev_tick}"
            )
            assert bar.note_count == len(bar.notes), (
                f"{entry.piece_id} bar {bar.bar_number}: "
                f"note_count {bar.note_count} != len(notes) {len(bar.notes)}"
            )
            prev_tick = bar.start_tick


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
def test_all_composers_represented():
    """All 16 ASAP composers should be discovered."""
    pieces = discover_pieces(ASAP_DIR)
    composers = {p.composer for p in pieces}
    expected = {
        "Bach", "Balakirev", "Beethoven", "Brahms", "Chopin", "Debussy",
        "Glinka", "Haydn", "Liszt", "Mozart", "Prokofiev", "Rachmaninoff",
        "Ravel", "Schubert", "Schumann", "Scriabin",
    }
    assert composers == expected, f"Missing composers: {expected - composers}"
