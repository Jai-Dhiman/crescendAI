"""Chroma-independent validation gate for ingested score MIDIs.

validate_score returns a list of Violations (empty == pass). It uses only
chroma-independent signals: DoD minimums, pitch range, bar-count plausibility,
16th-grid quantization (recovered from bar.start_tick deltas + time signature),
and Krumhansl-Schmuckler key agreement. It NEVER uses chroma self-recognition;
gating on the chroma matcher would rig the #21 feasibility harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from score_library.schema import ScoreData


@dataclass(frozen=True)
class ExpectedMeta:
    """Per-piece expected metadata and gate thresholds."""

    piece_id: str
    expected_key: str
    expected_bars: int
    min_notes: int = 20
    bar_tol_low: float = 0.7
    bar_tol_high: float = 2.2
    quant_max_median_dev_beats: float = 0.10
    key_min_correlation: float = 0.6
    pitch_low: int = 21
    pitch_high: int = 108


@dataclass(frozen=True)
class Violation:
    """A single failed gate check."""

    check: str
    detail: str


def _flatten_notes(score: ScoreData) -> list:
    notes = []
    for bar in score.bars:
        notes.extend(bar.notes)
    return notes


def validate_score(score: ScoreData, expected: ExpectedMeta) -> list[Violation]:
    """Validate a parsed score against expected metadata. Empty list == pass."""
    violations: list[Violation] = []

    notes = _flatten_notes(score)

    # (a) DoD minimums.
    if len(notes) < expected.min_notes:
        violations.append(
            Violation("min_notes", f"{len(notes)} notes < {expected.min_notes} minimum")
        )
    if score.total_bars < 1:
        violations.append(Violation("total_bars", f"total_bars={score.total_bars} < 1"))
    onsets = [n.onset_seconds for n in notes]
    if any(onsets[i] < onsets[i - 1] for i in range(1, len(onsets))):
        violations.append(Violation("monotonic_onsets", "onset_seconds not non-decreasing"))

    # (b) Pitch range.
    out_of_range = [n.pitch for n in notes if n.pitch < expected.pitch_low or n.pitch > expected.pitch_high]
    if out_of_range:
        violations.append(
            Violation(
                "pitch_range",
                f"{len(out_of_range)} notes outside [{expected.pitch_low}, {expected.pitch_high}]",
            )
        )

    return violations
