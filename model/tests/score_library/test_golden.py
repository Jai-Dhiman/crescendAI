"""Golden file tests for bar grid accuracy.

These tests verify that the bar grid builder produces correct bar counts
for specific pieces with known time signature changes. Manually verified
against the MIDI files.

This is the highest-risk codepath -- wrong bar numbers silently propagate
to all downstream consumers (score alignment, teacher feedback, etc.).
"""

from pathlib import Path

import pytest

from score_library.parse import parse_score_midi

ASAP_DIR = Path("/Users/jdhiman/Documents/crescendai/model/data/asap_cache")

# (relative_path, piece_id, expected_bar_count, expected_time_sig_count)
# Each piece was chosen for diversity: different composers, different time
# signature patterns, different levels of complexity.
GOLDEN_PIECES = [
    # Bach BWV 856 fugue: 3 time sigs (1/8, 3/8, 2/8)
    ("Bach/Fugue/bwv_856", "bach-bwv856", 73, 3),
    # Beethoven Sonata 17 mvt 3: 4 time sigs (3/16, 3/8, 3/8, 2/8)
    ("Beethoven/Piano_Sonatas/17-3", "beethoven-17-3", 494, 4),
    # Chopin Ballade 2: 4 time sigs (4/8, 6/8, 8/8, 6/8)
    ("Chopin/Ballades/2", "chopin-ballade2", 205, 4),
    # Debussy Reflets dans l'eau: 3 time sigs (4/8, 3/8, 4/8)
    ("Debussy/Images_Book_1/1_Reflets_dans_lEau", "debussy-reflets", 97, 3),
    # Liszt Concert Etude S145/2: 9 time sigs (6/8, 9/8, 2/4, ...)
    ("Liszt/Concert_Etude_S145/2", "liszt-concert-etude-2", 168, 9),
]


def _parse_golden(rel_path: str, piece_id: str) -> "ScoreData":  # noqa: F821
    """Helper to parse a golden piece from ASAP cache."""
    piece_dir = ASAP_DIR / rel_path
    score_files = sorted(piece_dir.glob("score_*.mid"))
    assert score_files, f"No score files in {piece_dir}"
    return parse_score_midi(score_files[0], piece_id, "Test", "Test")


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
@pytest.mark.parametrize(
    "rel_path,piece_id,expected_bars,expected_ts_count", GOLDEN_PIECES
)
def test_golden_bar_count(rel_path, piece_id, expected_bars, expected_ts_count):
    """Bar count must match the golden value exactly."""
    result = _parse_golden(rel_path, piece_id)
    assert result.total_bars == expected_bars, (
        f"{piece_id}: expected {expected_bars} bars, got {result.total_bars}"
    )
    assert len(result.time_signatures) == expected_ts_count, (
        f"{piece_id}: expected {expected_ts_count} time sigs, "
        f"got {len(result.time_signatures)}"
    )


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
@pytest.mark.parametrize(
    "rel_path,piece_id,expected_bars,expected_ts_count", GOLDEN_PIECES
)
def test_golden_bar_monotonic(rel_path, piece_id, expected_bars, expected_ts_count):
    """Bar start ticks must be strictly monotonically increasing."""
    result = _parse_golden(rel_path, piece_id)
    for i in range(1, len(result.bars)):
        assert result.bars[i].start_tick > result.bars[i - 1].start_tick, (
            f"Bar {result.bars[i].bar_number} start_tick "
            f"({result.bars[i].start_tick}) not greater than "
            f"bar {result.bars[i - 1].bar_number} ({result.bars[i - 1].start_tick})"
        )


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
@pytest.mark.parametrize(
    "rel_path,piece_id,expected_bars,expected_ts_count", GOLDEN_PIECES
)
def test_golden_bar_numbers_contiguous(
    rel_path, piece_id, expected_bars, expected_ts_count
):
    """Bar numbers must be contiguous from 1 to total_bars."""
    result = _parse_golden(rel_path, piece_id)
    bar_numbers = [bar.bar_number for bar in result.bars]
    assert bar_numbers == list(range(1, expected_bars + 1)), (
        f"{piece_id}: bar numbers are not contiguous 1..{expected_bars}"
    )


@pytest.mark.skipif(not ASAP_DIR.exists(), reason="ASAP cache not available")
@pytest.mark.parametrize(
    "rel_path,piece_id,expected_bars,expected_ts_count", GOLDEN_PIECES
)
def test_golden_start_seconds_monotonic(
    rel_path, piece_id, expected_bars, expected_ts_count
):
    """Bar start_seconds must be monotonically non-decreasing."""
    result = _parse_golden(rel_path, piece_id)
    for i in range(1, len(result.bars)):
        assert result.bars[i].start_seconds >= result.bars[i - 1].start_seconds, (
            f"Bar {result.bars[i].bar_number} start_seconds "
            f"({result.bars[i].start_seconds}) less than "
            f"bar {result.bars[i - 1].bar_number} ({result.bars[i - 1].start_seconds})"
        )
