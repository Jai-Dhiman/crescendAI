# apps/evals/teaching_knowledge/tests/test_run_do_baseline.py
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
    synthesis_latency_ms: int = 600
    piece_identification: object | None = None


def _judge(synthesis_text, context, **kwargs):
    return _FakeJudgeResult(dimensions=[_FakeDim("Audible-Specific Corrective Feedback", 2, 2, 2)])


def _write_briefing(path: Path, rid: str) -> None:
    path.write_text(json.dumps({
        "recording_id": rid,
        "total_duration_seconds": 120.0,
        "chunks": [{"chunk_index": 0, "predictions": {"dynamics": 0.5}}],
    }))


def _write_holdout(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_run_do_baseline_writes_one_row_per_recording(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.json"
    b2 = tmp_path / "b2.json"
    _write_briefing(b1, "rA")
    _write_briefing(b2, "rB")
    holdout = tmp_path / "holdout.jsonl"
    _write_holdout(holdout, [
        {"recording_id": "rA", "title": "P1", "composer": "Bach", "skill_bucket": 3,
         "piece_slug": "bach_invention_1", "briefing_path": str(b1)},
        {"recording_id": "rB", "title": "P2", "composer": "Chopin", "skill_bucket": 4,
         "piece_slug": "chopin_ballade_1", "briefing_path": str(b2)},
    ])

    seen_recordings = []

    def _driver(wrangler_url, recording_cache, student_id, piece_query):
        seen_recordings.append((recording_cache["recording_id"], piece_query))
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

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["recording_id"] for r in rows} == {"rA", "rB"}
    # piece_slug is passed as the piece_query to set_piece
    assert ("rA", "bach_invention_1") in seen_recordings
    assert all(r["judge_dimensions"] for r in rows)


def test_run_do_baseline_resumes_completed_recordings(tmp_path: Path) -> None:
    b1 = tmp_path / "b1.json"
    _write_briefing(b1, "rA")
    holdout = tmp_path / "holdout.jsonl"
    _write_holdout(holdout, [
        {"recording_id": "rA", "title": "P1", "composer": "Bach", "skill_bucket": 3,
         "piece_slug": "bach_invention_1", "briefing_path": str(b1)},
    ])
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({
        "recording_id": "rA", "synthesis_text": "already done",
        "judge_dimensions": [{"criterion": "x", "outcome": 2}], "error": "",
    }) + "\n")

    calls = []

    def _driver(wrangler_url, recording_cache, student_id, piece_query):
        calls.append(recording_cache["recording_id"])
        raise AssertionError("driver must not run for an already-completed recording")

    run_do_baseline(
        holdout_path=holdout,
        out_path=out,
        wrangler_url="http://localhost:8787",
        judge_fn=_judge,
        driver=_driver,
    )
    assert calls == []
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
