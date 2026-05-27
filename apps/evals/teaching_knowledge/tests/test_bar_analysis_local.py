from __future__ import annotations

from teaching_knowledge.bar_analysis_local import (
    compute_tier2_dimensions,
    select_worst_chunk,
)


def test_returns_six_dimensions_with_analysis_strings() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        {"pitch": 64, "onset": 1.0, "offset": 1.5, "velocity": 90},
    ]
    pedal_events = [{"time": 0.2, "value": 127}, {"time": 0.8, "value": 0}]
    result = compute_tier2_dimensions(midi_notes, pedal_events)
    assert len(result) == 6
    dims = [r["dimension"] for r in result]
    assert dims == [
        "dynamics",
        "timing",
        "pedaling",
        "articulation",
        "phrasing",
        "interpretation",
    ]
    for r in result:
        assert isinstance(r["analysis"], str)
        assert len(r["analysis"]) > 0


def test_dynamics_string_mentions_velocity() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 100},
    ]
    result = compute_tier2_dimensions(midi_notes, [])
    dynamics = next(r for r in result if r["dimension"] == "dynamics")
    assert "velocity" in dynamics["analysis"].lower()


def test_selects_chunk_with_largest_absolute_deviation() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [
        {"chunk_index": 0, "predictions": {"dynamics": 0.55, "timing": 0.45}},
        {"chunk_index": 1, "predictions": {"dynamics": 0.50, "timing": 0.20}},  # 0.30
        {"chunk_index": 2, "predictions": {"dynamics": 0.60, "timing": 0.50}},
    ]
    result = select_worst_chunk(chunks, baselines)
    assert result is not None
    assert result["chunk_index"] == 1
    assert result["dimension"] == "timing"


def test_returns_none_on_empty_chunks() -> None:
    assert select_worst_chunk([], {"timing": 0.5}) is None
