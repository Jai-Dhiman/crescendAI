"""Unit tests for score.py through its public interface."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

from pipeline.exercise_routing.score import (
    SessionCapture,
    SessionScore,
    AxisScores,
    score_session,
    aggregate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_capture(
    *,
    piece_slug: str = "fur_elise",
    piece_resolved: bool = True,
    dominant_dimension: str = "dynamics",
    prescribed_exercise: dict | None = None,
    top_moment_bar_range: list[int] | None = None,
) -> SessionCapture:
    teaching_moments = []
    if top_moment_bar_range is not None:
        teaching_moments = [{"dimension": dominant_dimension, "bar_range": top_moment_bar_range}]
    elif dominant_dimension:
        teaching_moments = [{"dimension": dominant_dimension}]
    return SessionCapture(
        session_id="test-session",
        recording=Path("dummy.wav"),
        piece_slug=piece_slug,
        teaching_moments=teaching_moments,
        baselines={},
        piece_identification={"pieceId": "fur-elise", "confidence": 0.95} if piece_resolved else None,
        piece_resolved=piece_resolved,
        dominant_dimension=dominant_dimension,
        prescribed_exercise=prescribed_exercise,
        synthesis_text="some synthesis",
    )


# ---------------------------------------------------------------------------
# invocation_rate
# ---------------------------------------------------------------------------

def test_no_prescription_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.invoked is False


def test_with_prescription_is_invoked():
    capture = make_capture(
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        }
    )
    score = score_session(capture)
    assert score.invoked is True


# ---------------------------------------------------------------------------
# kind_correctness
# ---------------------------------------------------------------------------

def test_kind_correct_own_passage_loop_when_piece_resolved():
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is True


def test_kind_incorrect_corpus_drill_when_piece_resolved():
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is False


def test_kind_correct_corpus_drill_when_not_resolved():
    capture = make_capture(
        piece_resolved=False,
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is True


def test_kind_incorrect_own_passage_loop_without_bar_range():
    """own_passage_loop with null bar_range is a kind violation."""
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": None,
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is False


def test_kind_none_when_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.kind_correct is None


# ---------------------------------------------------------------------------
# dimension_match
# ---------------------------------------------------------------------------

def test_dimension_match_when_equal():
    capture = make_capture(
        dominant_dimension="dynamics",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.dimension_match is True


def test_dimension_mismatch():
    capture = make_capture(
        dominant_dimension="timing",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.dimension_match is False


def test_dimension_match_none_when_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.dimension_match is None


def test_dimension_match_none_when_invoked_but_no_dominant_dimension():
    """When a prescription is invoked but teaching_moments is empty (no dominant dimension),
    dimension_match must be None — not a crash and not False."""
    capture = make_capture(
        dominant_dimension=None,
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.dimension_match is None


# ---------------------------------------------------------------------------
# bar_range_grounding
# ---------------------------------------------------------------------------

def test_bar_range_grounded_when_overlapping():
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=[2, 6],
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [4, 8],   # overlaps [2,6] at bars 4-6
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is True


def test_bar_range_not_grounded_when_disjoint():
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=[1, 4],
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [10, 14],   # no overlap with [1,4]
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is False


def test_bar_range_grounded_none_when_no_moment_bars():
    """If the top moment has no bar_range, grounding cannot be scored."""
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=None,   # no bar data
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is None


# ---------------------------------------------------------------------------
# tempo_sanity
# ---------------------------------------------------------------------------

def test_tempo_in_bounds():
    capture = make_capture(
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.7,
        },
    )
    score = score_session(capture)
    assert score.tempo_in_bounds is True


def test_tempo_out_of_bounds():
    capture = make_capture(
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 1.5,   # above max 1.0
        },
    )
    score = score_session(capture)
    assert score.tempo_in_bounds is False


def test_tempo_weak_prior_flag_timing_at_1_0():
    """tempo_factor==1.0 on a timing-dominant session flags as weak prior."""
    capture = make_capture(
        dominant_dimension="timing",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "timing",
            "bar_range": [1, 4],
            "tempo_factor": 1.0,
        },
    )
    score = score_session(capture)
    assert score.tempo_weak_prior_flag is True


def test_tempo_no_weak_prior_flag_non_timing():
    """tempo_factor==1.0 on a non-timing session is not flagged."""
    capture = make_capture(
        dominant_dimension="dynamics",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 1.0,
        },
    )
    score = score_session(capture)
    assert score.tempo_weak_prior_flag is False


def test_tempo_absent_treated_as_in_bounds():
    """Absent tempo_factor is treated as 1.0 -- in-bounds."""
    capture = make_capture(
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            # tempo_factor absent
        },
    )
    score = score_session(capture)
    assert score.tempo_in_bounds is True


def test_tempo_absent_on_timing_session_flags_weak_prior():
    """Absent tempo_factor (treated as 1.0) on timing-dominant session flags weak prior."""
    capture = make_capture(
        dominant_dimension="timing",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "timing",
            "bar_range": [1, 4],
            # tempo_factor absent -- effectively 1.0
        },
    )
    score = score_session(capture)
    assert score.tempo_weak_prior_flag is True


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_invocation_rate():
    scores = [
        SessionScore(session_id="a", piece_slug="p", invoked=True, kind_correct=True,
                     dimension_match=True, bar_range_grounded=None, tempo_in_bounds=True,
                     tempo_weak_prior_flag=False, error=None),
        SessionScore(session_id="b", piece_slug="p", invoked=False, kind_correct=None,
                     dimension_match=None, bar_range_grounded=None, tempo_in_bounds=None,
                     tempo_weak_prior_flag=None, error=None),
        SessionScore(session_id="c", piece_slug="p", invoked=True, kind_correct=True,
                     dimension_match=False, bar_range_grounded=True, tempo_in_bounds=True,
                     tempo_weak_prior_flag=False, error=None),
    ]
    result = aggregate(scores)
    assert result.n_sessions == 3
    assert result.n_invoked == 2
    assert abs(result.invocation_rate - 2/3) < 0.001
    assert abs(result.kind_correctness_rate - 1.0) < 0.001   # 2/2 invoked
    assert abs(result.dimension_match_rate - 0.5) < 0.001    # 1/2 invoked
    assert abs(result.bar_range_grounding_rate - 1.0) < 0.001 # 1/1 with bar data
    assert result.bar_range_grounding_n == 1
    # Risk 3 fix: tempo axes must be asserted in the aggregate test too
    assert abs(result.tempo_sanity_rate - 1.0) < 0.001       # both invoked sessions have tempo_in_bounds=True
    assert result.tempo_weak_prior_flag_count == 0             # neither session flagged weak prior
