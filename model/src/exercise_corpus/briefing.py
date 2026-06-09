"""Briefing + memory loop: turn a diagnosed weakness into an exercise prescription.

This is the model-side reference for what the api-side teacher briefing (issue
#29) will eventually do: take a Diagnosis (a dimension + severity + the bars
where the student struggled) plus a query embedding of that weak segment, match
it against the corpus (slice B), choose a dimension-appropriate exercise type +
deterministic transform (slice C), and emit an ExerciseBriefing whose fields map
onto the api-side ExerciseArtifact contract.

Memory: a 3-day cooldown keyed on (dimension, overlapping bar_range) mirrors the
production molecule's fetchSimilarPastObservation rule so the same weakness is
not re-prescribed back-to-back.
"""

from dataclasses import dataclass, field
from pathlib import Path

from exercise_corpus.keys import load_passage_key, parse_key_to_pc, transpose_interval
from exercise_corpus.match import CatalogIndex, Match, match_by_dimension
from exercise_corpus.tags import TagSet

# The 6 teacher dimensions (mirrors apps/api .../artifacts/diagnosis DIMENSIONS).
VALID_SEVERITIES = ("moderate", "significant")
COOLDOWN_DAYS = 3
_SECONDS_PER_DAY = 86_400


class CooldownError(Exception):
    """Raised when a diagnosis was already addressed within the cooldown window."""


@dataclass
class Diagnosis:
    dimension: str
    severity: str  # "moderate" | "significant" ("minor" is rejected)
    bar_range: tuple[int, int]  # bars in the STUDENT's piece
    piece_id: str


@dataclass
class PrescriptionRecord:
    primitive_id: str
    dimension: str
    bar_range: tuple[int, int]
    prescribed_at: float  # epoch seconds


@dataclass
class ExerciseBriefing:
    target_dimension: str
    severity: str
    exercise_type: str
    matched_primitive_id: str
    matched_source: str
    matched_title: str
    match_score: float
    transform: str | None  # "tempo" | "excerpt" | None
    transform_params: dict | None
    transpose_semitones: int | None  # key shift; None if passage key unknown
    target_key: str | None           # resolved passage key string e.g. "Eb"
    bar_range: tuple[int, int]  # student's piece bars (from the diagnosis)
    estimated_minutes: int
    instruction: str
    success_criterion: str
    action_binding: str | None
    candidates: list[Match] = field(default_factory=list)


# dimension -> (exercise_type, transform, action_binding tool | None).
# transform is the deterministic slice-C transform applied to the matched
# primitive; None where no honest symbolic transform exists yet (dynamics,
# articulation -- those are performance/hand-split concerns slice C does not cover).
_DIMENSION_PLAN: dict[str, tuple[str, str | None, str | None]] = {
    "timing": ("segment_loop", "excerpt", "segment_loop"),
    "pedaling": ("pedal_isolation", "excerpt", "mute_pedal"),
    "phrasing": ("slow_practice", "tempo", None),
    "interpretation": ("slow_practice", "tempo", None),
    "dynamics": ("dynamic_exaggeration", None, None),
    "articulation": ("isolated_hands", None, "isolated_hands"),
}

_MINUTES = {"moderate": 5, "significant": 8}

_INSTRUCTION = {
    "segment_loop": "Loop bars {start}-{end} with a metronome using the matched drill ({title}). Match the click exactly; do not accelerate or slow down.",
    "pedal_isolation": "Play bars {start}-{end} three times with no sustain pedal, using {title} as the model. Listen for whether the line sustains in your fingers.",
    "slow_practice": "Play bars {start}-{end} at half tempo, shaping the phrase as in {title}. Listen to the line as you play.",
    "dynamic_exaggeration": "Play bars {start}-{end} with exaggerated dynamics, modeled on {title}. Make loud louder and soft softer than feels right.",
    "isolated_hands": "Play bars {start}-{end} hands separately, modeled on {title}. Focus on even articulation between fingers.",
}

_SUCCESS = {
    "segment_loop": "Five consecutive repetitions matching the metronome within 20ms.",
    "pedal_isolation": "Three clean no-pedal repetitions where harmonies stay audibly distinct.",
    "slow_practice": "The phrase shape is clear and intentional at half tempo.",
    "dynamic_exaggeration": "The loudest and softest moments are clearly distinct.",
    "isolated_hands": "Each hand plays with consistent articulation for three repetitions.",
}


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def should_prescribe(
    dimension: str,
    bar_range: tuple[int, int],
    history: list[PrescriptionRecord],
    now: float,
    cooldown_days: int = COOLDOWN_DAYS,
) -> bool:
    """Return False if this diagnosis was addressed within the cooldown window.

    Keyed on (dimension, overlapping bar_range) -- which weakness was treated,
    independent of which exercise was prescribed.
    """
    cutoff = now - cooldown_days * _SECONDS_PER_DAY
    for r in history:
        if (
            r.dimension == dimension
            and r.prescribed_at >= cutoff
            and _overlaps(r.bar_range, bar_range)
        ):
            return False
    return True


def _transform_params(
    transform: str | None, severity: str, bar_range: tuple[int, int]
) -> dict | None:
    if transform == "tempo":
        # Slow harder for the worse problem.
        return {"factor": 0.5 if severity == "significant" else 0.66}
    if transform == "excerpt":
        length = bar_range[1] - bar_range[0] + 1
        return {"start_bar": 1, "end_bar": length}
    return None


def build_briefing(
    diagnosis: Diagnosis,
    tags: dict[str, TagSet],
    history: list[PrescriptionRecord],
    now: float,
    db_path=None,
    index: CatalogIndex | None = None,
    top_k: int = 5,
    scores_dir: Path | None = None,
) -> ExerciseBriefing:
    """Match a diagnosed weakness to an exercise and emit a prescription briefing.

    Retrieval is by diagnosed dimension via curated technique tags
    (match.match_by_dimension), not by a query embedding.

    Raises:
        ValueError: if severity is not in {moderate, significant} or the
            dimension is unknown.
        CooldownError: if the diagnosis was addressed within the cooldown window.
        NoPrimitiveForDimensionError: if no catalog primitive is tagged for the
            diagnosis dimension (e.g. pedaling or dynamics on the current corpus).
    """
    if diagnosis.severity not in VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {VALID_SEVERITIES}, got {diagnosis.severity!r}"
        )
    if diagnosis.dimension not in _DIMENSION_PLAN:
        raise ValueError(
            f"unknown dimension {diagnosis.dimension!r}; "
            f"supported: {sorted(_DIMENSION_PLAN)}"
        )
    if not should_prescribe(diagnosis.dimension, diagnosis.bar_range, history, now):
        raise CooldownError(
            f"{diagnosis.dimension} at bars {diagnosis.bar_range} was addressed "
            f"within the last {COOLDOWN_DAYS} days"
        )

    matches = match_by_dimension(
        diagnosis.dimension, tags, db_path=db_path, index=index, top_k=top_k
    )
    top = matches[0]

    # Key resolution: transpose C-major exercise into the student's passage key.
    exercise_pc = parse_key_to_pc(tags[top.primitive_id].key)
    passage_key_str = load_passage_key(diagnosis.piece_id, scores_dir)
    if passage_key_str is None:
        transpose_semitones = None
        target_key = None
    else:
        passage_pc = parse_key_to_pc(passage_key_str)
        transpose_semitones = transpose_interval(exercise_pc, passage_pc)
        target_key = passage_key_str

    exercise_type, transform, action_binding = _DIMENSION_PLAN[diagnosis.dimension]
    start, end = diagnosis.bar_range
    t_params = _transform_params(transform, diagnosis.severity, diagnosis.bar_range)

    base_instruction = _INSTRUCTION[exercise_type].format(
        start=start, end=end, title=top.title
    )
    if target_key is not None:
        instruction = base_instruction + f" Transpose into the key of {target_key}."
    else:
        instruction = base_instruction

    return ExerciseBriefing(
        target_dimension=diagnosis.dimension,
        severity=diagnosis.severity,
        exercise_type=exercise_type,
        matched_primitive_id=top.primitive_id,
        matched_source=top.source,
        matched_title=top.title,
        match_score=top.score,
        transform=transform,
        transform_params=t_params,
        transpose_semitones=transpose_semitones,
        target_key=target_key,
        bar_range=diagnosis.bar_range,
        estimated_minutes=_MINUTES[diagnosis.severity],
        instruction=instruction,
        success_criterion=_SUCCESS[exercise_type],
        action_binding=action_binding,
        candidates=matches,
    )


def record_prescription(briefing: ExerciseBriefing, now: float) -> PrescriptionRecord:
    """Build the memory record to append to history after prescribing a briefing."""
    return PrescriptionRecord(
        primitive_id=briefing.matched_primitive_id,
        dimension=briefing.target_dimension,
        bar_range=briefing.bar_range,
        prescribed_at=now,
    )
