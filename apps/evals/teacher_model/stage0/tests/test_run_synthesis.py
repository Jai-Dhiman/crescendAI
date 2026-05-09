# apps/evals/teacher_model/stage0/tests/test_run_synthesis.py
"""Pipeline A runner: shape, resume behaviour, judge invocation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from teacher_model.stage0.run_synthesis import run as run_synthesis


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "fake/qwen-x"

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        self.calls += 1
        return "<analysis>scratchpad</analysis>Real teacher response, warm and specific."


@dataclass
class _FakeDim:
    criterion: str
    process: int = 2
    outcome: int = 2
    score: int = 2
    evidence: str = ""
    reason: str = ""


@dataclass
class _FakeJudge:
    dimensions: list
    model: str = "fake/judge"
    prompt_version: str = "judge_v2_extended"
    latency_ms: float = 1.0


_DIMS = [
    "Audible-Specific Corrective Feedback",
    "Concrete Artifact Provision",
    "Specific Positive Praise",
    "Autonomy-Supporting Motivation",
    "Scaffolded Guided Discovery",
    "Style-Consistent Musical Language",
    "Appropriate Tone & Language",
    "Taste Defensibility",
    "Adaptation Specificity",
]


def _fake_judge_fn(synthesis_text, context, **kwargs):
    return _FakeJudge(dimensions=[_FakeDim(criterion=d) for d in _DIMS])


def _write_holdout(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rid in ids:
            f.write(
                json.dumps(
                    {
                        "recording_id": rid,
                        "era": "Romantic",
                        "skill_bucket": 3,
                        "stratum": "Romantic|sk3",
                        "composer": "Chopin",
                        "piece_slug": "test_piece",
                        "title": "Test Piece",
                        "briefing_path": str(path.parent / f"{rid}.json"),
                    }
                )
                + "\n"
            )
            (path.parent / f"{rid}.json").write_text(
                json.dumps(
                    {
                        "recording_id": rid,
                        "chunks": [
                            {
                                "predictions": {
                                    "dynamics": 0.5, "timing": 0.5, "pedaling": 0.6,
                                    "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5,
                                }
                            }
                        ],
                        "total_duration_seconds": 600.0,
                    }
                )
            )


def test_run_writes_one_row_per_holdout_with_nine_dims(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.jsonl"
    out = tmp_path / "synth.jsonl"
    _write_holdout(holdout, ["r1", "r2", "r3"])
    client = _FakeClient()
    stats = run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client, judge_fn=_fake_judge_fn,
        system_prompt="You are a teacher.",
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert len(row["judge_dimensions"]) == 9
        assert row["error"] == ""
        assert row["model_id"] == "fake/qwen-x"
    assert stats.processed == 3 and stats.errors == 0


def test_run_resumes_skipping_completed_ids(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.jsonl"
    out = tmp_path / "synth.jsonl"
    _write_holdout(holdout, ["r1", "r2"])
    client = _FakeClient()
    run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client, judge_fn=_fake_judge_fn, system_prompt="x",
    )
    assert client.calls == 2

    # Add a third id; re-run should only process the new one.
    _write_holdout(holdout, ["r1", "r2", "r3"])
    client2 = _FakeClient()
    stats = run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=client2, judge_fn=_fake_judge_fn, system_prompt="x",
    )
    assert client2.calls == 1
    assert stats.processed == 1


def test_errored_rows_are_not_re_appended_on_retry(tmp_path: Path) -> None:
    """A row that previously errored stays present and is NOT duplicated on retry."""
    holdout = tmp_path / "holdout.jsonl"
    out = tmp_path / "synth.jsonl"
    _write_holdout(holdout, ["r1"])

    class _FailingClient:
        model = "fake/qwen-x"

        def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
            raise RuntimeError("transient upstream failure")

    run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=_FailingClient(), judge_fn=_fake_judge_fn, system_prompt="x",
    )
    rows_after_first = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows_after_first) == 1
    assert rows_after_first[0]["error"]

    # Re-run with a healthy client. The errored row must NOT be retried,
    # and the file must still contain exactly one row (no duplicate).
    run_synthesis(
        holdout_path=holdout, out_path=out,
        teacher_client=_FakeClient(), judge_fn=_fake_judge_fn, system_prompt="x",
    )
    rows_after_second = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows_after_second) == 1
    assert rows_after_second[0]["error"]  # still errored; user must clean to retry
