"""Pydantic schema models for score library JSON output."""

from pydantic import BaseModel


class ScoreNote(BaseModel):
    """A single note event from a score MIDI file."""

    pitch: int
    pitch_name: str
    velocity: int
    onset_tick: int
    onset_seconds: float
    duration_ticks: int
    duration_seconds: float
    track: int


class PedalEvent(BaseModel):
    """A sustain pedal on/off event."""

    type: str  # "on" or "off"
    tick: int
    seconds: float


class Bar(BaseModel):
    """A single bar (measure) with its notes and metadata."""

    bar_number: int
    start_tick: int
    start_seconds: float
    time_signature: str
    notes: list[ScoreNote]
    pedal_events: list[PedalEvent]
    note_count: int
    pitch_range: list[int]
    mean_velocity: int


class ScoreData(BaseModel):
    """Full parsed score data for a single piece."""

    piece_id: str
    composer: str
    title: str
    key_signature: str | None
    time_signatures: list[dict]
    tempo_markings: list[dict]
    total_bars: int
    bars: list[Bar]


class PieceCatalogEntry(BaseModel):
    """Summary catalog entry for a piece (used in the catalog index)."""

    piece_id: str
    composer: str
    title: str
    key_signature: str | None
    time_signature: str | None
    tempo_bpm: int | None
    bar_count: int
    duration_seconds: float | None
    note_count: int
    pitch_range_low: int | None
    pitch_range_high: int | None
    has_time_sig_changes: bool
    has_tempo_changes: bool
    source: str = "asap"
