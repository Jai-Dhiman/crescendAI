# apps/evals/teaching_knowledge/tests/test_do_row.py
from __future__ import annotations

from dataclasses import dataclass, field

from teaching_knowledge.run_eval import build_do_row
from shared.provenance import make_run_provenance


@dataclass
class _FakeDim:
    criterion: str
    process: int | None
    outcome: int | None
    score: int | None
    evidence: str = ""
    reason: str = ""


@dataclass
class _FakeJudgeResult:
    dimensions: list
    model: str = "fake-judge"
    latency_ms: float = 12.0


@dataclass
class _FakeSynthesis:
    text: str
    is_fallback: bool = False
    eval_context: dict = field(default_factory=dict)


@dataclass
class _FakeSessionResult:
    recording_id: str
    synthesis: object | None
    errors: list
    synthesis_latency_ms: int = 700
    piece_identification: object | None = None


@dataclass
class _FakePieceId:
    piece_id: str


_META = {
    "piece_slug": "fur_elise",
    "title": "Fur Elise",
    "composer": "Beethoven",
    "skill_bucket": 3,
}


def _judge_ok(synthesis_text, context, **kwargs):
    assert synthesis_text  # judge only called with real text
    return _FakeJudgeResult(
        dimensions=[
            _FakeDim("Audible-Specific Corrective Feedback", 2, 1, 1),
            _FakeDim("Specific Positive Praise", 3, 3, 3),
        ]
    )


def test_successful_session_yields_judged_row() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec1",
        synthesis=_FakeSynthesis(text="Lovely phrasing, try softer pedaling."),
        errors=[],
        piece_identification=_FakePieceId("fur_elise"),
    )
    row = build_do_row(sr, _META, _judge_ok, prov)

    assert row["recording_id"] == "rec1"
    assert row["error"] == ""
    assert row["run_id"] == prov.run_id
    assert row["synthesis_text"] == "Lovely phrasing, try softer pedaling."
    assert row["piece_resolved"] is True
    crits = {d["criterion"]: d["outcome"] for d in row["judge_dimensions"]}
    assert crits["Audible-Specific Corrective Feedback"] == 1
    assert crits["Specific Positive Praise"] == 3


def test_do_failure_records_error_and_skips_judge() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec2",
        synthesis=None,
        errors=["WebSocket error: connection refused"],
    )

    def _judge_must_not_run(*a, **k):  # noqa: ANN001, ANN002
        raise AssertionError("judge must not run when the DO failed")

    row = build_do_row(sr, _META, _judge_must_not_run, prov)

    assert row["recording_id"] == "rec2"
    assert "connection refused" in row["error"]
    assert row["judge_dimensions"] == []
    assert row["synthesis_text"] == ""
    assert row["piece_resolved"] is False


def test_unresolved_piece_flags_false_but_still_judges() -> None:
    prov = make_run_provenance()
    sr = _FakeSessionResult(
        recording_id="rec3",
        synthesis=_FakeSynthesis(text="Keep the line singing through the rests."),
        errors=[],
        piece_identification=None,  # piece not resolved -> Tier 2/3, bar_range null
    )
    row = build_do_row(sr, _META, _judge_ok, prov)
    assert row["error"] == ""
    assert row["piece_resolved"] is False
    assert len(row["judge_dimensions"]) == 2
