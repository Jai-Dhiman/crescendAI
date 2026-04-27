# apps/evals/teaching_knowledge/ablation/test_run_ablation.py
import json
from pathlib import Path

from teaching_knowledge.ablation.run_ablation import run_ablation


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.calls.append({"user": user, "system": system})
        marker = "POS" if "above_average" in user else "NEG"
        return f"Practice was good. ({marker})"


SESSION = {
    "recording_id": "rec_001",
    "muq_means": {"dynamics": 0.7, "timing": 0.4},
    "duration_seconds": 600,
    "meta": {"piece_slug": "test_piece", "title": "Test Prelude",
             "composer": "Bach", "skill_bucket": 3},
}


def test_run_ablation_emits_four_rows(tmp_path: Path):
    out = tmp_path / "ablation.jsonl"
    client = FakeClient()
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    conditions = {r["condition"] for r in rows}
    assert conditions == {"real", "shuffle", "marginal", "flip"}
    for r in rows:
        assert r["recording_id"] == "rec_001"
        assert "synthesis_text" in r
        assert "top_moments_used" in r


def test_run_ablation_resume_safe(tmp_path: Path):
    out = tmp_path / "ablation.jsonl"
    client = FakeClient()
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    n_first = len(client.calls)
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    assert len(client.calls) == n_first


def test_run_ablation_judge_called_per_condition(tmp_path: Path):
    out = tmp_path / "ablation.jsonl"

    class FakeJudge:
        def __init__(self) -> None:
            self.calls = 0
        def complete(self, *, user: str, system: str, max_tokens: int) -> str:
            self.calls += 1
            return '[{"criterion": "specificity", "process": 2, "outcome": 2, "score": 2, "evidence": "ok", "reason": "ok"}]'

    client = FakeClient()
    judge = FakeJudge()
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, judge_client=judge, seed=42)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    assert judge.calls == 4
    for r in rows:
        assert "judge_dimensions" in r
        assert isinstance(r["judge_dimensions"], list)
