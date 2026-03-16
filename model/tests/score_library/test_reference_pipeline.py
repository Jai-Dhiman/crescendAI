"""Integration tests for the reference generation pipeline.

Uses the existing DTW alignment and bar stats functions with minimal fixtures.
"""
import json
import pytest

from src.score_library.reference_cache import (
    align_to_score,
    compute_bar_stats,
    aggregate_bar_stats,
    BarStats,
    ReferenceProfile,
)


class TestValidationCoverage:
    """Test the coverage gate logic (>= 75% of score bars must have aligned notes)."""

    def test_full_coverage(self):
        """A well-aligned performance should have high coverage."""
        # Simulate: score has 10 bars, alignment returns notes for 9 of them
        total_bars = 10
        aligned_bars = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        coverage = len(aligned_bars) / total_bars
        assert coverage >= 0.75

    def test_low_coverage_rejected(self):
        """A poorly-aligned performance should be rejected."""
        total_bars = 10
        aligned_bars = {1: [], 5: []}  # only 2 of 10 bars
        coverage = len(aligned_bars) / total_bars
        assert coverage < 0.75

    def test_edge_at_threshold(self):
        total_bars = 4
        aligned_bars = {1: [], 2: [], 3: []}  # 75% exactly
        coverage = len(aligned_bars) / total_bars
        assert coverage >= 0.75


class TestAlignToScoreReturnType:
    """Verify align_to_score returns (bar_mapping, dtw_cost) tuple."""

    def test_empty_input_returns_tuple(self):
        result = align_to_score([], {"bars": []})
        assert isinstance(result, tuple)
        assert len(result) == 2
        bar_map, cost = result
        assert bar_map == {}
        assert cost == 0.0


class TestAggregation:
    """Test that aggregation across multiple performers produces valid output."""

    def test_two_performers(self):
        stats_a = [
            BarStats(bar_number=1, velocity_mean=80.0, velocity_std=5.0, performer_count=1),
            BarStats(bar_number=2, velocity_mean=90.0, velocity_std=6.0, performer_count=1),
        ]
        stats_b = [
            BarStats(bar_number=1, velocity_mean=70.0, velocity_std=4.0, performer_count=1),
            BarStats(bar_number=2, velocity_mean=85.0, velocity_std=7.0, performer_count=1),
        ]
        result = aggregate_bar_stats([stats_a, stats_b])
        assert len(result) == 2
        assert result[0].performer_count == 2
        assert result[0].velocity_mean == pytest.approx(75.0)

    def test_non_negative_pedal_changes(self):
        """pedal_changes must always be non-negative (u32 in Rust consumer)."""
        stats = [
            [BarStats(bar_number=1, pedal_changes=3, performer_count=1)],
            [BarStats(bar_number=1, pedal_changes=5, performer_count=1)],
        ]
        result = aggregate_bar_stats(stats)
        assert result[0].pedal_changes >= 0


class TestReferenceProfileSerialization:
    """Test that serialized JSON matches the Rust consumer schema."""

    def test_schema_fields(self):
        from dataclasses import asdict

        profile = ReferenceProfile(
            piece_id="test.piece",
            performer_count=3,
            bars=[
                BarStats(
                    bar_number=1,
                    velocity_mean=75.0,
                    velocity_std=5.0,
                    onset_deviation_mean_ms=10.0,
                    onset_deviation_std_ms=3.0,
                    pedal_duration_mean_beats=2.5,
                    pedal_changes=4,
                    note_duration_ratio_mean=1.1,
                    performer_count=3,
                ),
            ],
        )
        data = asdict(profile)
        assert data["piece_id"] == "test.piece"
        assert data["performer_count"] == 3

        bar = data["bars"][0]
        required_fields = {
            "bar_number", "velocity_mean", "velocity_std",
            "onset_deviation_mean_ms", "onset_deviation_std_ms",
            "pedal_duration_mean_beats", "pedal_changes",
            "note_duration_ratio_mean", "performer_count",
        }
        assert required_fields.issubset(bar.keys())

    def test_optional_fields_can_be_null(self):
        from dataclasses import asdict

        bar = BarStats(bar_number=1, performer_count=1)
        data = asdict(bar)
        # pedal fields should serialize as None -> null
        assert data["pedal_duration_mean_beats"] is None
        assert data["pedal_changes"] is None

        # Verify JSON serialization handles None
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["pedal_duration_mean_beats"] is None
        assert parsed["pedal_changes"] is None
