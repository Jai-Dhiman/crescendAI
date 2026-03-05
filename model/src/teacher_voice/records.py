"""Unified record format for teacher voice data pipeline.

Every data source (masterclass moments, pedagogy books, synthetic scenarios)
produces TeachingRecords. The same records serve three consumers:
  1. Eval benchmark -- test Claude's current capabilities
  2. Fine-tuning dataset -- SFT/DPO pairs for Llama training
  3. Retrieval corpus -- piece-specific knowledge for prompt enrichment
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# The 6-dimension taxonomy used by CrescendAI
DIMENSIONS = frozenset({
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
})

STUDENT_LEVELS = frozenset({"beginner", "intermediate", "advanced"})

FEEDBACK_TYPES = frozenset({
    "correction",
    "suggestion",
    "encouragement",
    "explanation",
    "question",
})

SOURCES = frozenset({
    "masterclass",
    "pedagogy_book",
    "youtube_tutorial",
    "synthetic",
    "golden_set",
})

# Mapping from raw masterclass pipeline dimensions to 6-dim taxonomy.
# The pipeline uses finer-grained categories that collapse into our 6.
RAW_DIMENSION_MAP: dict[str, str] = {
    "dynamics": "dynamics",
    "timing": "timing",
    "pedaling": "pedaling",
    "articulation": "articulation",
    "phrasing": "phrasing",
    "interpretation": "interpretation",
    "technique": "articulation",
    "voicing": "interpretation",
    "tone_color": "interpretation",
    "structure": "phrasing",
}


@dataclass
class TeachingRecord:
    """A single teaching scenario with context and teacher response."""

    id: str
    source: str  # one of SOURCES

    # Musical context
    piece: str | None = None
    composer: str | None = None
    bars: str | None = None
    passage_description: str | None = None

    # Teaching content
    dimension: str = ""  # one of DIMENSIONS
    student_level: str = ""  # one of STUDENT_LEVELS
    feedback_type: str = ""  # one of FEEDBACK_TYPES

    # The actual content
    scenario: str = ""  # what the student did (input context for the LLM)
    teacher_response: str = ""  # what the teacher said (target output)

    # Provenance
    teacher_name: str | None = None
    source_id: str = ""  # video_id, book chapter, etc.
    raw_transcript: str | None = None

    # Quality metadata
    quality_score: float | None = None
    has_embodied_language: bool = False
    has_piece_specificity: bool = False
    is_actionable: bool = False

    # Extra metadata (source-specific)
    metadata: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation errors, empty if valid."""
        errors = []
        if not self.id:
            errors.append("id is required")
        if self.source not in SOURCES:
            errors.append(f"source '{self.source}' not in {SOURCES}")
        if self.dimension not in DIMENSIONS:
            errors.append(f"dimension '{self.dimension}' not in {DIMENSIONS}")
        if self.student_level and self.student_level not in STUDENT_LEVELS:
            errors.append(f"student_level '{self.student_level}' not in {STUDENT_LEVELS}")
        if not self.scenario:
            errors.append("scenario is required")
        if not self.teacher_response:
            errors.append("teacher_response is required")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TeachingRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_records(records: list[TeachingRecord], path: Path) -> None:
    """Save records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[TeachingRecord]:
    """Load records from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(TeachingRecord.from_dict(json.loads(line)))
    return records
