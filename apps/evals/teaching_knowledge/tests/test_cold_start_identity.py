# apps/evals/teaching_knowledge/tests/test_cold_start_identity.py
"""Guard test: cold-start isolation.

Verifies that run_do_baseline assigns a UNIQUE studentId per recording AND
transmits it to the driver. Before the fix, every recording used the same
constant "eval-student-001", meaning session history from earlier recordings
polluted later ones. After the fix, each recording must get a distinct
"eval-<recording_id>" identity.

The test drives through the injectable `driver` boundary so no live DO is
needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from teaching_knowledge.run_eval import run_do_baseline


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
    latency_ms: float = 5.0


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
    synthesis_latency_ms: int = 100
    piece_identification: object | None = None


def _judge(synthesis_text, context, **kwargs):
    return _FakeJudgeResult(
        dimensions=[_FakeDim("Audible-Specific Corrective Feedback", 2, 2, 2)]
    )


def _write_briefing(path: Path, rid: str) -> None:
    path.write_text(
        json.dumps({
            "recording_id": rid,
            "total_duration_seconds": 60.0,
            "chunks": [{"chunk_index": 0, "predictions": {"dynamics": 0.5}}],
        })
    )


def _write_holdout(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cold_start_unique_student_id_per_recording(tmp_path: Path) -> None:
    """Each recording must receive a distinct eval studentId.

    Before fix: all recordings get student_id="eval-student-001" (constant).
    After fix:  each recording gets student_id="eval-<recording_id>".

    Fails against the pre-fix code because all student_ids are the same string.
    """
    recording_ids = ["rA", "rB", "rC"]
    briefing_paths = {}
    for rid in recording_ids:
        p = tmp_path / f"{rid}.json"
        _write_briefing(p, rid)
        briefing_paths[rid] = p

    holdout = tmp_path / "holdout.jsonl"
    _write_holdout(
        holdout,
        [
            {
                "recording_id": rid,
                "title": f"Title {rid}",
                "composer": "Bach",
                "skill_bucket": 3,
                "piece_slug": f"piece_{rid}",
                "briefing_path": str(briefing_paths[rid]),
            }
            for rid in recording_ids
        ],
    )

    seen_student_ids: list[str] = []

    def _driver(wrangler_url, recording_cache, student_id, piece_query):
        seen_student_ids.append(student_id)
        return _FakeSessionResult(
            recording_id=recording_cache["recording_id"],
            synthesis=_FakeSynthesis(text=f"feedback for {recording_cache['recording_id']}"),
            errors=[],
        )

    out = tmp_path / "out.jsonl"
    run_do_baseline(
        holdout_path=holdout,
        out_path=out,
        wrangler_url="http://localhost:8787",
        judge_fn=_judge,
        driver=_driver,
    )

    # Must have received one student_id per recording
    assert len(seen_student_ids) == len(recording_ids), (
        f"Expected {len(recording_ids)} driver calls, got {len(seen_student_ids)}"
    )

    # All student_ids must be DISTINCT (cold-start: each recording is a fresh student)
    assert len(set(seen_student_ids)) == len(recording_ids), (
        f"COLD-START VIOLATION: student_ids are not unique across recordings: {seen_student_ids}"
    )

    # Each student_id must embed the recording_id (format: "eval-<recording_id>")
    for rid, sid in zip(recording_ids, seen_student_ids):
        assert sid == f"eval-{rid}", (
            f"student_id for recording {rid!r} should be 'eval-{rid}', got {sid!r}"
        )
