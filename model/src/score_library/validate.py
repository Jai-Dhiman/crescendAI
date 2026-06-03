"""Chroma-independent validation gate for ingested score MIDIs.

validate_score returns a list of Violations (empty == pass). It uses only
chroma-independent signals: DoD minimums, pitch range, bar-count plausibility,
16th-grid quantization (recovered from bar.start_tick deltas + time signature),
and Krumhansl-Schmuckler key agreement. It NEVER uses chroma self-recognition;
gating on the chroma matcher would rig the #21 feasibility harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from score_library.schema import ScoreData

KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_TONIC_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    denom = (var_a * var_b) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


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

    # (c) Bar-count plausibility.
    low = expected.bar_tol_low * expected.expected_bars
    high = expected.bar_tol_high * expected.expected_bars
    if not (low <= score.total_bars <= high):
        violations.append(
            Violation(
                "bar_count",
                f"total_bars={score.total_bars} outside plausible [{low:.1f}, {high:.1f}] for expected {expected.expected_bars}",
            )
        )

    # (d) Quantization: recover the 16th grid from bar.start_tick deltas + time sig.
    if len(score.bars) >= 2:
        bar_ticks_list: list[int] = []
        for i in range(len(score.bars) - 1):
            bar_ticks_list.append(score.bars[i + 1].start_tick - score.bars[i].start_tick)
        # Last bar reuses the previous bar's tick span.
        bar_ticks_list.append(bar_ticks_list[-1] if bar_ticks_list else 0)

        devs_beats: list[float] = []
        for bi, bar in enumerate(score.bars):
            bar_ticks = bar_ticks_list[bi]
            if bar_ticks <= 0:
                continue
            try:
                num_str, den_str = bar.time_signature.split("/")
                denominator = int(den_str)
            except (ValueError, AttributeError):
                continue
            numerator = int(num_str)
            subdivisions_per_bar = numerator * 16 / denominator
            sixteenths_per_beat = 16 / denominator
            for note in bar.notes:
                fraction = (note.onset_tick - bar.start_tick) / bar_ticks
                grid_pos = fraction * subdivisions_per_bar
                dev_sixteenths = abs(grid_pos - round(grid_pos))
                devs_beats.append(dev_sixteenths / sixteenths_per_beat)

        if devs_beats:
            med = median(devs_beats)
            if med > expected.quant_max_median_dev_beats:
                violations.append(
                    Violation(
                        "quantization",
                        f"median grid deviation {med:.3f} beats > {expected.quant_max_median_dev_beats} (grossly off-grid / fixed-offset?)",
                    )
                )

    # (e) Key agreement (Krumhansl-Schmuckler) -- chroma-independent.
    parts = expected.expected_key.split()
    if len(parts) == 2 and parts[0] in _TONIC_TO_PC and parts[1] in ("major", "minor"):
        tonic_pc = _TONIC_TO_PC[parts[0]]
        profile = KRUMHANSL_MAJOR if parts[1] == "major" else KRUMHANSL_MINOR
        rotated = [profile[(pc - tonic_pc) % 12] for pc in range(12)]

        pc_counts = [0] * 12
        for n in notes:
            pc_counts[n.pitch % 12] += 1
        total = sum(pc_counts)
        if total > 0:
            histogram = [c / total for c in pc_counts]
            corr = _pearson(histogram, rotated)
            if corr < expected.key_min_correlation:
                violations.append(
                    Violation(
                        "key_agreement",
                        f"key correlation {corr:.3f} < {expected.key_min_correlation} for expected {expected.expected_key}",
                    )
                )

    return violations
