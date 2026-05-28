"""Exercise corpus construction and embedding validation pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Primitive:
    primitive_id: str
    source: str
    source_exercise_number: int
    title: str
    musicxml_path: Path
    midi_path: Path
    n_notes: int
