"""Tests for render_artifact_text (#28: the judge grades the artifact, not the headline)."""

from pipeline_client import SynthesisResult, render_artifact_text


def _artifact(**overrides) -> dict:
    base = {
        "session_id": "sess_1",
        "synthesis_scope": "session",
        "strengths": [{"dimension": "dynamics", "one_liner": "bars 3-6 shaped the phrase"}],
        "focus_areas": [
            {"dimension": "timing", "one_liner": "rushed the left hand", "severity": "moderate"}
        ],
        "prescribed_exercise": None,
        "dominant_dimension": "timing",
        "recurring_pattern": None,
        "next_session_focus": "steady pulse in the left hand",
        "diagnosis_refs": ["diag_1"],
        "headline": "You held the phrase shape well today.",
        "assigned_loops": [],
    }
    base.update(overrides)
    return base


def test_includes_fields_the_headline_omits():
    out = render_artifact_text(
        SynthesisResult(text="You held the phrase shape well today.", is_fallback=False, artifact=_artifact())
    )

    assert "You held the phrase shape well today." in out
    assert "bars 3-6 shaped the phrase" in out
    assert "rushed the left hand" in out
    assert "moderate" in out
    assert "steady pulse in the left hand" in out


def test_falls_back_to_headline_when_no_artifact():
    out = render_artifact_text(SynthesisResult(text="just the headline", is_fallback=False, artifact=None))

    assert out == "just the headline"


def test_omits_null_optional_sections():
    out = render_artifact_text(
        SynthesisResult(
            text="h",
            is_fallback=False,
            artifact=_artifact(recurring_pattern=None, next_session_focus=None),
        )
    )

    assert "Recurring pattern" not in out
    assert "Next session focus" not in out


def test_renders_recurring_pattern_when_present():
    out = render_artifact_text(
        SynthesisResult(text="h", is_fallback=False, artifact=_artifact(recurring_pattern="rushes under pressure"))
    )

    assert "rushes under pressure" in out
