# apps/evals/tests/test_render_artifact.py
from __future__ import annotations

from dataclasses import dataclass, field

from shared.provenance import RunProvenance
from teaching_knowledge.run_eval import build_do_row, render_artifact_text

FULL_ARTIFACT = {
    "headline": "Solid, focused session today. Your timing held steady.",
    "focus_areas": [
        {"dimension": "pedaling", "one_liner": "Harmonies blur at the cadence", "severity": "significant"},
        {"dimension": "dynamics", "one_liner": "Crescendos peak too early", "severity": "minor"},
    ],
    "strengths": [
        {"dimension": "timing", "one_liner": "Steady pulse throughout"},
    ],
    "proposed_exercises": [
        "Half-pedal the cadence at bar 12, listening for clean harmony changes",
    ],
}


def test_renders_all_teaching_sections() -> None:
    text = render_artifact_text(FULL_ARTIFACT)
    assert text is not None
    # Headline leads.
    assert text.startswith("Solid, focused session today.")
    # Focus areas the headline omits are present, with dimension + severity signal.
    assert "Focus areas:" in text
    assert "[significant] pedaling: Harmonies blur at the cadence" in text
    assert "[minor] dynamics: Crescendos peak too early" in text
    # Strengths present.
    assert "Strengths:" in text
    assert "timing: Steady pulse throughout" in text
    # Proposed exercises present.
    assert "Suggested exercises:" in text
    assert "Half-pedal the cadence at bar 12" in text


def test_none_artifact_returns_none() -> None:
    # No structured artifact -> caller falls back to the headline.
    assert render_artifact_text(None) is None
    assert render_artifact_text({}) is None


def test_headline_only_omits_empty_sections() -> None:
    text = render_artifact_text(
        {
            "headline": "Short session, good focus.",
            "focus_areas": [],
            "strengths": [],
            "proposed_exercises": [],
        }
    )
    assert text == "Short session, good focus."
    assert "Focus areas:" not in text
    assert "Strengths:" not in text
    assert "Suggested exercises:" not in text


def test_missing_headline_still_renders_other_sections() -> None:
    text = render_artifact_text(
        {
            "strengths": [{"dimension": "phrasing", "one_liner": "Shaped the line well"}],
        }
    )
    assert text is not None
    assert "Strengths:" in text
    assert "phrasing: Shaped the line well" in text


@dataclass
class _StubSynthesis:
    text: str  # the headline only (live WS field)
    eval_context: dict = field(default_factory=dict)


@dataclass
class _StubSessionResult:
    synthesis: _StubSynthesis
    recording_id: str = "rec-1"
    piece_identification: object | None = None
    errors: list = field(default_factory=list)
    synthesis_latency_ms: int = 0


def test_build_do_row_judges_full_artifact_not_headline() -> None:
    """The judge must receive the rendered full artifact, not the headline alone."""
    captured: dict = {}

    @dataclass
    class _JudgeDim:
        criterion: str = "ascf"
        process: float = 1.0
        outcome: float = 1.0
        score: float = 1.0
        evidence: str = ""
        reason: str = ""

    @dataclass
    class _JudgeResult:
        dimensions: list = field(default_factory=lambda: [_JudgeDim()])
        model: str = "stub-judge"
        latency_ms: float = 0.0

    def fake_judge(text, ctx, *, provider, model):
        captured["text"] = text
        return _JudgeResult()

    session_result = _StubSessionResult(
        synthesis=_StubSynthesis(
            text=FULL_ARTIFACT["headline"],
            eval_context={"artifact": FULL_ARTIFACT},
        )
    )
    prov = RunProvenance(run_id="r", git_sha="sha", git_dirty=False)
    row = build_do_row(session_result, {}, fake_judge, prov)

    # The judged text is the full render -- focus areas + exercises, not headline only.
    assert "Focus areas:" in captured["text"]
    assert "Suggested exercises:" in captured["text"]
    assert "Focus areas:" in row["synthesis_text"]  # row records what was judged


def test_build_do_row_falls_back_to_headline_without_artifact() -> None:
    """No artifact in eval_context -> judge the headline (backward compatible)."""
    captured: dict = {}

    @dataclass
    class _JudgeResult:
        dimensions: list = field(default_factory=list)
        model: str = "stub-judge"
        latency_ms: float = 0.0

    def fake_judge(text, ctx, *, provider, model):
        captured["text"] = text
        return _JudgeResult()

    session_result = _StubSessionResult(
        synthesis=_StubSynthesis(text="Just a headline.", eval_context={})
    )
    prov = RunProvenance(run_id="r", git_sha="sha", git_dirty=False)
    row = build_do_row(session_result, {}, fake_judge, prov)

    assert captured["text"] == "Just a headline."
    assert row["synthesis_text"] == "Just a headline."
