"""Pure scoring module for ExerciseRoutingDecision correctness.

No I/O. All metric definitions live here. Callable with synthetic SessionCapture
fixtures -- the real test surface for the routing eval harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionCapture:
    """Structured output of driving one recording through the chunk_ready path."""
    session_id: str
    recording: Path
    piece_slug: str
    teaching_moments: list[dict]
    baselines: dict
    piece_identification: dict | None
    piece_resolved: bool
    dominant_dimension: str | None
    prescribed_exercise: dict | None
    synthesis_text: str


@dataclass
class SessionScore:
    """Per-session scoring result across all five axes."""
    session_id: str
    piece_slug: str
    invoked: bool
    kind_correct: bool | None
    dimension_match: bool | None
    bar_range_grounded: bool | None
    tempo_in_bounds: bool | None
    tempo_weak_prior_flag: bool | None
    error: str | None


@dataclass
class AxisScores:
    """Aggregate harness-level scores across all sessions."""
    invocation_rate: float
    kind_correctness_rate: float
    dimension_match_rate: float
    bar_range_grounding_rate: float
    bar_range_grounding_n: int
    tempo_sanity_rate: float
    tempo_weak_prior_flag_count: int
    n_sessions: int
    n_invoked: int
    n_errors: int


def score_session(capture: SessionCapture) -> SessionScore:
    """Score one session's capture across all five axes. Pure; no I/O."""
    ex = capture.prescribed_exercise
    invoked = ex is not None

    if not invoked:
        return SessionScore(
            session_id=capture.session_id,
            piece_slug=capture.piece_slug,
            invoked=False,
            kind_correct=None,
            dimension_match=None,
            bar_range_grounded=None,
            tempo_in_bounds=None,
            tempo_weak_prior_flag=None,
            error=None,
        )

    kind_correct = _score_kind(capture, ex)
    dimension_match = _score_dimension_match(capture, ex)
    bar_range_grounded = _score_bar_range_grounding(capture, ex)
    tempo_in_bounds, tempo_weak_prior_flag = _score_tempo(capture, ex)

    return SessionScore(
        session_id=capture.session_id,
        piece_slug=capture.piece_slug,
        invoked=True,
        kind_correct=kind_correct,
        dimension_match=dimension_match,
        bar_range_grounded=bar_range_grounded,
        tempo_in_bounds=tempo_in_bounds,
        tempo_weak_prior_flag=tempo_weak_prior_flag,
        error=None,
    )


def aggregate(scores: list[SessionScore]) -> AxisScores:
    """Aggregate per-session scores into harness-level AxisScores. Pure; no I/O."""
    n_sessions = len(scores)
    n_errors = sum(1 for s in scores if s.error is not None)
    invoked_scores = [s for s in scores if s.invoked and s.error is None]
    n_invoked = len(invoked_scores)

    invocation_rate = n_invoked / n_sessions if n_sessions > 0 else 0.0

    kind_judged = [s for s in invoked_scores if s.kind_correct is not None]
    kind_correctness_rate = (
        sum(1 for s in kind_judged if s.kind_correct) / len(kind_judged)
        if kind_judged else 0.0
    )

    dim_judged = [s for s in invoked_scores if s.dimension_match is not None]
    dimension_match_rate = (
        sum(1 for s in dim_judged if s.dimension_match) / len(dim_judged)
        if dim_judged else 0.0
    )

    bar_judged = [s for s in invoked_scores if s.bar_range_grounded is not None]
    bar_range_grounding_n = len(bar_judged)
    bar_range_grounding_rate = (
        sum(1 for s in bar_judged if s.bar_range_grounded) / bar_range_grounding_n
        if bar_range_grounding_n > 0 else 0.0
    )

    tempo_judged = [s for s in invoked_scores if s.tempo_in_bounds is not None]
    tempo_sanity_rate = (
        sum(1 for s in tempo_judged if s.tempo_in_bounds) / len(tempo_judged)
        if tempo_judged else 0.0
    )

    tempo_weak_prior_flag_count = sum(
        1 for s in invoked_scores if s.tempo_weak_prior_flag
    )

    return AxisScores(
        invocation_rate=invocation_rate,
        kind_correctness_rate=kind_correctness_rate,
        dimension_match_rate=dimension_match_rate,
        bar_range_grounding_rate=bar_range_grounding_rate,
        bar_range_grounding_n=bar_range_grounding_n,
        tempo_sanity_rate=tempo_sanity_rate,
        tempo_weak_prior_flag_count=tempo_weak_prior_flag_count,
        n_sessions=n_sessions,
        n_invoked=n_invoked,
        n_errors=n_errors,
    )


# ---------------------------------------------------------------------------
# Private axis scorers
# ---------------------------------------------------------------------------

def _score_kind(capture: SessionCapture, ex: dict) -> bool:
    """Correct kind = own_passage_loop iff piece_resolved; corpus_drill otherwise.

    Guard: own_passage_loop must carry a non-null bar_range (else it's a kind violation).
    """
    kind = ex.get("kind")
    if capture.piece_resolved:
        if kind != "own_passage_loop":
            return False
        bar_range = ex.get("bar_range")
        if not bar_range:
            return False  # own_passage_loop requires bar_range
        return True
    else:
        return kind == "corpus_drill"


def _score_dimension_match(capture: SessionCapture, ex: dict) -> bool | None:
    """prescription.target_dimension == capture.dominant_dimension."""
    if capture.dominant_dimension is None:
        return None
    return ex.get("target_dimension") == capture.dominant_dimension


def _score_bar_range_grounding(capture: SessionCapture, ex: dict) -> bool | None:
    """Prescription bar_range has non-empty intersection with top teaching moment's bar_range.

    Returns None if the top teaching moment has no bar_range (AMT did not provide bars).
    Only scored for own_passage_loop prescriptions (corpus_drill bar_range is a template hint).
    """
    if ex.get("kind") != "own_passage_loop":
        return None
    if not capture.teaching_moments:
        return None
    top_bar_range = capture.teaching_moments[0].get("bar_range")
    if not top_bar_range:
        return None
    prescription_bar = ex.get("bar_range")
    if not prescription_bar:
        return False
    # Non-empty intersection: max(starts) <= min(ends)
    p_start, p_end = prescription_bar[0], prescription_bar[1]
    m_start, m_end = top_bar_range[0], top_bar_range[1]
    return max(p_start, m_start) <= min(p_end, m_end)


def _score_tempo(capture: SessionCapture, ex: dict) -> tuple[bool, bool]:
    """Returns (tempo_in_bounds, tempo_weak_prior_flag).

    In-bounds: [0.25, 1.0] inclusive.
    Weak prior flag: tempo_factor==1.0 on a timing-dominant session (no tempo
    reduction prescribed for the dimension most likely to benefit from it).
    """
    tempo = ex.get("tempo_factor")
    if tempo is None:
        return (True, False)  # absent = schema default = 1.0, treated as in-bounds
    in_bounds = 0.25 <= tempo <= 1.0
    weak_prior = (tempo == 1.0) and (capture.dominant_dimension == "timing")
    return (in_bounds, weak_prior)
