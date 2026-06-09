# apps/evals/teaching_knowledge/ablation/test_run_ablation.py
import json
from dataclasses import dataclass, field
from pathlib import Path

from teaching_knowledge.ablation.run_ablation import run_ablation

DIMS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]


def _chunk(idx: int, score: float) -> dict:
    return {
        "chunk_index": idx,
        "predictions": {d: score for d in DIMS},
        "midi_notes": [],
        "pedal_events": [],
    }


def _recording(rid: str, score: float) -> dict:
    return {
        "recording_id": rid,
        "chunks": [_chunk(0, score), _chunk(1, score)],
        "meta": {"piece_slug": "p", "title": "T", "composer": "C", "skill_bucket": 3},
    }


@dataclass
class _StubSynthesis:
    text: str
    eval_context: dict = field(default_factory=dict)


@dataclass
class _StubSessionResult:
    recording_id: str
    synthesis: _StubSynthesis
    piece_identification: object | None = None
    errors: list = field(default_factory=list)
    synthesis_latency_ms: int = 0


def _artifact(headline: str) -> dict:
    return {
        "headline": headline,
        "focus_areas": [
            {"dimension": "pedaling", "one_liner": "blurred", "severity": "significant"}
        ],
        "strengths": [],
        "proposed_exercises": ["drill the cadence"],
    }


class _CapturingDriver:
    """Records every recording_cache it is handed and returns a stub SessionResult."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, wrangler_url, recording_cache, student_id, piece_query):
        self.calls.append(
            {"cache": recording_cache, "student_id": student_id, "piece": piece_query}
        )
        rid = recording_cache["recording_id"]
        return _StubSessionResult(
            recording_id=rid,
            synthesis=_StubSynthesis(
                text="Headline lead-in for the student.",
                eval_context={"artifact": _artifact("Headline lead-in for the student.")},
            ),
        )


def test_emits_four_conditions(tmp_path: Path) -> None:
    out = tmp_path / "ablation.jsonl"
    driver = _CapturingDriver()
    run_ablation(
        recordings=[_recording("rec_001", 0.7), _recording("rec_002", 0.3)],
        out_path=out,
        driver=driver,
        skip_judge=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 8  # 2 recordings x 4 conditions
    rec1 = {r["condition"] for r in rows if r["recording_id"] == "rec_001"}
    assert rec1 == {"real", "shuffle", "marginal", "flip"}


def test_corruption_is_applied_to_chunks_sent_to_the_do(tmp_path: Path) -> None:
    """The driver must receive corrupted predictions for non-real conditions."""
    out = tmp_path / "ablation.jsonl"
    driver = _CapturingDriver()
    run_ablation(
        recordings=[_recording("rec_001", 0.7), _recording("rec_002", 0.3)],
        out_path=out,
        driver=driver,
        skip_judge=True,
    )
    by_student = {c["student_id"]: c["cache"]["chunks"] for c in driver.calls}

    # real: predictions untouched (0.7).
    real_chunks = by_student["eval-ablation-rec_001-real"]
    assert real_chunks[0]["predictions"]["dynamics"] == 0.7

    # flip: every prediction inverted to 1 - 0.7 = 0.3.
    flip_chunks = by_student["eval-ablation-rec_001-flip"]
    assert all(
        v == 0.3 for ch in flip_chunks for v in ch["predictions"].values()
    )

    # shuffle: borrowed rec_002's signal (0.3), not rec_001's (0.7).
    shuffle_chunks = by_student["eval-ablation-rec_001-shuffle"]
    assert shuffle_chunks[0]["predictions"]["dynamics"] == 0.3


def test_request_shaping_passes_recording_id_and_piece(tmp_path: Path) -> None:
    out = tmp_path / "ablation.jsonl"
    driver = _CapturingDriver()
    run_ablation(
        recordings=[_recording("rec_001", 0.7), _recording("rec_002", 0.3)],
        out_path=out,
        driver=driver,
        skip_judge=True,
    )
    real_call = next(c for c in driver.calls if c["student_id"] == "eval-ablation-rec_001-real")
    assert real_call["cache"]["recording_id"] == "rec_001"
    assert real_call["piece"] == "p"
    assert len(real_call["cache"]["chunks"]) == 2


def test_judge_receives_full_artifact_per_condition(tmp_path: Path) -> None:
    out = tmp_path / "ablation.jsonl"
    driver = _CapturingDriver()
    judged: list[str] = []

    @dataclass
    class _Dim:
        criterion: str = "ascf"
        process: float = 1.0
        outcome: float = 1.0
        score: float = 1.0
        evidence: str = ""
        reason: str = ""

    @dataclass
    class _JR:
        dimensions: list = field(default_factory=lambda: [_Dim()])
        model: str = "stub"
        latency_ms: float = 0.0

    def fake_judge(text, ctx, *, provider, model):
        judged.append(text)
        return _JR()

    run_ablation(
        recordings=[_recording("rec_001", 0.7), _recording("rec_002", 0.3)],
        out_path=out,
        driver=driver,
        judge_fn=fake_judge,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 8
    assert len(judged) == 8  # judged once per condition
    # The judge saw the FULL artifact render (focus areas + exercises), not headline only.
    assert all("Focus areas:" in t for t in judged)
    assert all("Suggested exercises:" in t for t in judged)
    for r in rows:
        assert isinstance(r["judge_dimensions"], list)
        assert r["condition"] in {"real", "shuffle", "marginal", "flip"}


def test_resume_safe(tmp_path: Path) -> None:
    out = tmp_path / "ablation.jsonl"
    driver = _CapturingDriver()
    recordings = [_recording("rec_001", 0.7), _recording("rec_002", 0.3)]
    run_ablation(recordings=recordings, out_path=out, driver=driver, skip_judge=True)
    n_first = len(driver.calls)
    run_ablation(recordings=recordings, out_path=out, driver=driver, skip_judge=True)
    assert len(driver.calls) == n_first  # nothing re-driven
