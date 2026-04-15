# apps/evals/tests/test_run_eval_provenance.py
from __future__ import annotations

from shared.provenance import RunProvenance
from teaching_knowledge.run_eval import _build_row


def test_build_row_includes_run_id_and_git_sha() -> None:
    prov = RunProvenance(run_id="2026-04-14T12-00-00Z_abc1234", git_sha="abc1234deadbeef", git_dirty=False)
    meta = {
        "piece_slug": "bach",
        "title": "WTC",
        "composer": "Bach",
        "skill_bucket": 3,
    }
    row = _build_row(
        recording_id="rec123",
        meta=meta,
        muq_means={"articulation": 0.5},
        synthesis_text="Nice articulation!",
        synthesis_latency_ms=100,
        judge_dimensions=[],
        judge_model="test-judge",
        judge_latency_ms=50,
        error="",
        provenance=prov,
    )
    assert row["run_id"] == "2026-04-14T12-00-00Z_abc1234"
    assert row["git_sha"] == "abc1234deadbeef"
    assert row["git_dirty"] is False
    assert row["recording_id"] == "rec123"
    assert row["synthesis_text"] == "Nice articulation!"


def test_build_row_marks_dirty_tree() -> None:
    prov = RunProvenance(run_id="x", git_sha="y", git_dirty=True)
    row = _build_row(
        recording_id="r1",
        meta={"piece_slug": "p", "title": "t", "composer": "Bach", "skill_bucket": 3},
        muq_means={},
        synthesis_text="",
        synthesis_latency_ms=0,
        judge_dimensions=[],
        judge_model="",
        judge_latency_ms=0,
        error="boom",
        provenance=prov,
    )
    assert row["git_dirty"] is True
    assert row["error"] == "boom"
