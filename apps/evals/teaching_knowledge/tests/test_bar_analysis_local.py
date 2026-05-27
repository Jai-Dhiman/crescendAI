from __future__ import annotations

from teaching_knowledge.bar_analysis_local import compute_tier2_dimensions


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
