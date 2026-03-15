"""Tests for score library Pydantic schema models."""

from score_library.schema import (
    Bar,
    PedalEvent,
    PieceCatalogEntry,
    ScoreData,
    ScoreNote,
)


def test_score_note_valid():
    note = ScoreNote(
        pitch=60,
        pitch_name="C4",
        velocity=80,
        onset_tick=0,
        onset_seconds=0.0,
        duration_ticks=480,
        duration_seconds=0.5,
        track=0,
    )
    assert note.pitch == 60
    assert note.pitch_name == "C4"
    assert note.velocity == 80
    assert note.onset_tick == 0
    assert note.onset_seconds == 0.0
    assert note.duration_ticks == 480
    assert note.duration_seconds == 0.5
    assert note.track == 0


def test_bar_with_notes():
    notes = [
        ScoreNote(
            pitch=60,
            pitch_name="C4",
            velocity=80,
            onset_tick=0,
            onset_seconds=0.0,
            duration_ticks=480,
            duration_seconds=0.5,
            track=0,
        ),
        ScoreNote(
            pitch=64,
            pitch_name="E4",
            velocity=70,
            onset_tick=480,
            onset_seconds=0.5,
            duration_ticks=480,
            duration_seconds=0.5,
            track=0,
        ),
    ]
    bar = Bar(
        bar_number=1,
        start_tick=0,
        start_seconds=0.0,
        time_signature="4/4",
        notes=notes,
        pedal_events=[],
        note_count=2,
        pitch_range=[60, 64],
        mean_velocity=75,
    )
    assert bar.bar_number == 1
    assert len(bar.notes) == 2
    assert bar.note_count == 2
    assert bar.pitch_range == [60, 64]
    assert bar.mean_velocity == 75


def test_empty_bar():
    bar = Bar(
        bar_number=5,
        start_tick=1920,
        start_seconds=2.0,
        time_signature="4/4",
        notes=[],
        pedal_events=[],
        note_count=0,
        pitch_range=[],
        mean_velocity=0,
    )
    assert bar.note_count == 0
    assert bar.notes == []
    assert bar.pitch_range == []


def test_score_data_serialization():
    note = ScoreNote(
        pitch=60,
        pitch_name="C4",
        velocity=80,
        onset_tick=0,
        onset_seconds=0.0,
        duration_ticks=480,
        duration_seconds=0.5,
        track=0,
    )
    pedal = PedalEvent(type="on", tick=0, seconds=0.0)
    bar = Bar(
        bar_number=1,
        start_tick=0,
        start_seconds=0.0,
        time_signature="4/4",
        notes=[note],
        pedal_events=[pedal],
        note_count=1,
        pitch_range=[60],
        mean_velocity=80,
    )
    score = ScoreData(
        piece_id="Bach/Prelude_No1",
        composer="Bach",
        title="Prelude No. 1 in C Major",
        key_signature="C major",
        time_signatures=[{"time_sig": "4/4", "tick": 0}],
        tempo_markings=[{"bpm": 120, "tick": 0}],
        total_bars=1,
        bars=[bar],
    )

    # Round-trip: model -> dict -> model
    data = score.model_dump()
    restored = ScoreData.model_validate(data)

    assert restored.piece_id == "Bach/Prelude_No1"
    assert restored.composer == "Bach"
    assert restored.key_signature == "C major"
    assert len(restored.bars) == 1
    assert restored.bars[0].notes[0].pitch == 60
    assert restored.bars[0].pedal_events[0].type == "on"


def test_piece_catalog_entry():
    entry = PieceCatalogEntry(
        piece_id="Chopin/Ballade_No1",
        composer="Chopin",
        title="Ballade No. 1 in G minor, Op. 23",
        key_signature="G minor",
        time_signature="6/4",
        tempo_bpm=60,
        bar_count=264,
        duration_seconds=540.0,
        note_count=5000,
        pitch_range_low=28,
        pitch_range_high=96,
        has_time_sig_changes=True,
        has_tempo_changes=True,
    )
    assert entry.piece_id == "Chopin/Ballade_No1"
    assert entry.source == "asap"
    assert entry.has_time_sig_changes is True
    assert entry.tempo_bpm == 60
