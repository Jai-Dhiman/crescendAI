"""Convert raw masterclass moments into TeachingRecords.

Reads all_moments.jsonl from the masterclass pipeline and produces
structured records for eval, fine-tuning, and retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .records import (
    DIMENSIONS,
    RAW_DIMENSION_MAP,
    TeachingRecord,
    save_records,
)

# Patterns that indicate embodied/tactile teaching vocabulary
EMBODIED_PATTERNS = re.compile(
    r"(?i)\b("
    r"arm weight|wrist|finger(s|tip)?|hand position|touch|legato|staccato|"
    r"singing tone|breathe|breathing|relax|tension|tense|loose|"
    r"drop|fall|float|sink|lift|carry|throw|bounce|pull|push|"
    r"warm|bright|dark|round|sharp|soft|heavy|light|deep|"
    r"bow|string|voice|sing|ring|bloom|"
    r"pedal change|half.pedal|flutter pedal|una corda"
    r")\b"
)

# Patterns that suggest the feedback is actionable (suggests what to try)
ACTIONABLE_PATTERNS = re.compile(
    r"(?i)\b("
    r"try|practice|play it|play the|listen for|hear the|"
    r"think of|imagine|feel the|use|avoid|"
    r"slower|faster|softer|louder|more|less|"
    r"instead of|rather than|should be"
    r")\b"
)

# Minimum feedback summary length to be considered usable
MIN_SUMMARY_LENGTH = 20

# Maximum transcript length in chars -- beyond this, likely noisy
MAX_TRANSCRIPT_LENGTH = 5000


def _normalize_level(raw_level: str) -> str:
    """Map raw student_level to one of our three levels."""
    raw = raw_level.lower().strip()
    if raw in ("advanced", "professional"):
        return "advanced"
    if raw in ("intermediate",):
        return "intermediate"
    if raw in ("beginner",):
        return "beginner"
    # Anything else (young artist, etc.) defaults to advanced
    # since masterclass participants are typically advanced
    return "advanced"


def _normalize_feedback_type(raw_type: str) -> str:
    """Map raw feedback_type to our taxonomy."""
    raw = raw_type.lower().strip()
    if raw in ("correction",):
        return "correction"
    if raw in ("suggestion",):
        return "suggestion"
    if raw in ("explanation",):
        return "explanation"
    if raw in ("demonstration", "comparison"):
        return "suggestion"
    if raw in ("praise",):
        return "encouragement"
    return "correction"


def _build_scenario(moment: dict) -> str:
    """Construct the scenario description from raw moment fields."""
    parts = []

    piece = moment.get("piece")
    composer = moment.get("composer")
    if piece and composer:
        parts.append(f"The student is playing {piece} by {composer}.")
    elif piece:
        parts.append(f"The student is playing {piece}.")

    passage = moment.get("passage_description")
    if passage:
        parts.append(f"Passage: {passage}.")

    dim = moment.get("musical_dimension", "")
    mapped_dim = RAW_DIMENSION_MAP.get(dim, dim)
    severity = moment.get("severity", "")
    if mapped_dim and severity:
        parts.append(f"The teacher noticed a {severity} issue with {mapped_dim}.")
    elif mapped_dim:
        parts.append(f"The teacher focused on {mapped_dim}.")

    level = _normalize_level(moment.get("student_level", "advanced"))
    parts.append(f"Student level: {level}.")

    return " ".join(parts)


def _is_usable_moment(moment: dict) -> bool:
    """Filter out moments that won't produce quality records."""
    summary = moment.get("feedback_summary", "")
    if len(summary) < MIN_SUMMARY_LENGTH:
        return False

    transcript = moment.get("transcript_text", "")
    if len(transcript) > MAX_TRANSCRIPT_LENGTH:
        return False

    # Skip if dimension doesn't map to our taxonomy
    dim = moment.get("musical_dimension", "")
    if dim not in RAW_DIMENSION_MAP:
        return False

    # Skip very low confidence extractions
    confidence = moment.get("confidence", 0)
    if confidence < 0.5:
        return False

    return True


def convert_masterclass_moments(
    jsonl_path: Path,
    include_demonstrated: bool = True,
) -> list[TeachingRecord]:
    """Convert all_moments.jsonl to TeachingRecords.

    Args:
        jsonl_path: Path to all_moments.jsonl
        include_demonstrated: Whether to include moments where the teacher
            demonstrated at the piano. These may have less verbal content
            but the feedback_summary captures the teaching point.

    Returns:
        List of validated TeachingRecords.
    """
    records = []

    with open(jsonl_path) as f:
        for line in f:
            moment = json.loads(line)

            if not _is_usable_moment(moment):
                continue

            if not include_demonstrated and moment.get("demonstrated", False):
                continue

            raw_dim = moment.get("musical_dimension", "")
            mapped_dim = RAW_DIMENSION_MAP.get(raw_dim, "")
            if mapped_dim not in DIMENSIONS:
                continue

            summary = moment["feedback_summary"]
            scenario = _build_scenario(moment)

            record = TeachingRecord(
                id=moment["moment_id"],
                source="masterclass",
                piece=moment.get("piece"),
                composer=moment.get("composer"),
                bars=None,  # raw moments don't have bar numbers
                passage_description=moment.get("passage_description"),
                dimension=mapped_dim,
                student_level=_normalize_level(
                    moment.get("student_level", "advanced")
                ),
                feedback_type=_normalize_feedback_type(
                    moment.get("feedback_type", "correction")
                ),
                scenario=scenario,
                teacher_response=summary,
                teacher_name=moment.get("teacher"),
                source_id=moment.get("video_id", ""),
                raw_transcript=moment.get("transcript_text"),
                quality_score=moment.get("confidence"),
                has_embodied_language=bool(EMBODIED_PATTERNS.search(summary)),
                has_piece_specificity=bool(
                    moment.get("passage_description")
                    and len(moment["passage_description"]) > 5
                ),
                is_actionable=bool(ACTIONABLE_PATTERNS.search(summary)),
                metadata={
                    "severity": moment.get("severity"),
                    "demonstrated": moment.get("demonstrated", False),
                    "secondary_dimensions": moment.get("secondary_dimensions", []),
                    "video_title": moment.get("video_title"),
                    "stop_order": moment.get("stop_order"),
                    "total_stops": moment.get("total_stops"),
                    "time_spent_seconds": moment.get("time_spent_seconds"),
                },
            )

            errors = record.validate()
            if not errors:
                records.append(record)

    return records


def print_conversion_stats(records: list[TeachingRecord]) -> None:
    """Print summary statistics of converted records."""
    from collections import Counter

    print(f"Total records: {len(records)}")

    dims = Counter(r.dimension for r in records)
    print("\nBy dimension:")
    for dim, count in dims.most_common():
        print(f"  {dim}: {count}")

    levels = Counter(r.student_level for r in records)
    print("\nBy student level:")
    for level, count in levels.most_common():
        print(f"  {level}: {count}")

    ftypes = Counter(r.feedback_type for r in records)
    print("\nBy feedback type:")
    for ft, count in ftypes.most_common():
        print(f"  {ft}: {count}")

    embodied = sum(1 for r in records if r.has_embodied_language)
    specific = sum(1 for r in records if r.has_piece_specificity)
    actionable = sum(1 for r in records if r.is_actionable)
    print(f"\nQuality flags:")
    print(f"  Embodied language: {embodied} ({100*embodied/len(records):.0f}%)")
    print(f"  Piece-specific: {specific} ({100*specific/len(records):.0f}%)")
    print(f"  Actionable: {actionable} ({100*actionable/len(records):.0f}%)")


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2] / "data"
    jsonl_path = data_dir / "masterclass_pipeline" / "all_moments.jsonl"
    output_path = data_dir / "teacher_voice_eval" / "masterclass_records.jsonl"

    records = convert_masterclass_moments(jsonl_path)
    print_conversion_stats(records)
    save_records(records, output_path)
    print(f"\nSaved {len(records)} records to {output_path}")
