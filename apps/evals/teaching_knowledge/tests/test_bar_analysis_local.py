from __future__ import annotations

from teaching_knowledge.bar_analysis_local import (
    build_bar_analysis,
    compute_tier1_dimensions,
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


def test_tier1_articulation_mentions_duration_ratio() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 95},
    ]
    score_json = {
        "bars": [
            {"bar_number": 1, "notes": [
                {"pitch": 60, "velocity": 80, "duration_seconds": 0.5},
                {"pitch": 62, "velocity": 80, "duration_seconds": 0.5},
            ]}
        ]
    }
    result = compute_tier1_dimensions(midi_notes, [], score_json)
    articulation = next(r for r in result if r["dimension"] == "articulation")
    text = articulation["analysis"].lower()
    assert "ratio " in text or "performance/score" in text.lower()


def test_tier1_dynamics_does_not_mention_notated_score() -> None:
    # Score JSONs use default MIDI velocity 80 for every note; comparing
    # performance velocity to that constant is signal-free. The Tier-1 port
    # deliberately omits a dynamics-vs-notated line. See bar_analysis_local.py.
    midi_notes = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90}]
    score_json = {"bars": [{"bar_number": 1, "notes": [
        {"pitch": 60, "velocity": 80, "duration_seconds": 0.5}
    ]}]}
    result = compute_tier1_dimensions(midi_notes, [], score_json)
    dynamics = next(r for r in result if r["dimension"] == "dynamics")
    text = dynamics["analysis"].lower()
    assert "notated" not in text
    assert "(score)" not in text


def test_returns_none_on_no_chunks() -> None:
    assert build_bar_analysis([], {"timing": 0.5}, None) is None


def test_returns_tier2_facts_when_score_json_is_none() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [{
        "chunk_index": 0,
        "predictions": {"dynamics": 0.55, "timing": 0.20, "pedaling": 0.50,
                        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50},
        "midi_notes": [
            {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
            {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        ],
        "pedal_events": [],
    }]
    result = build_bar_analysis(chunks, baselines, None)
    assert result is not None
    assert result["tier"] == 2
    assert result["selected"]["dimension"] == "timing"
    # dynamics dev = 0.05 < 0.15 → correlated should be empty
    assert result["correlated"] == []


def test_correlated_includes_dimensions_above_threshold_cap_2() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [{
        "chunk_index": 0,
        # devs: dyn 0.30, tim selected(-0.31, strictly worst), ped 0.25, art 0.20, phr 0.10, int 0.05
        # NOTE: timing dev = 0.31 (not 0.30) to break the dyn/tim tie deterministically;
        # select_worst_chunk uses strict > comparison and dict iteration order is insertion order.
        "predictions": {"dynamics": 0.80, "timing": 0.19, "pedaling": 0.75,
                        "articulation": 0.70, "phrasing": 0.60, "interpretation": 0.55},
        "midi_notes": [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}],
        "pedal_events": [],
    }]
    result = build_bar_analysis(chunks, baselines, None)
    dims = [c["dimension"] for c in result["correlated"]]
    assert len(dims) == 2
    assert dims == ["dynamics", "pedaling"]  # top 2 by |dev|
