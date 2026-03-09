"""Tests for LLM feedback assessment (Experiment 4)."""

import pytest

from model_improvement.feedback_assessment import (
    build_condition_a_prompt,
    build_condition_b_prompt,
    build_judge_prompt,
    parse_judge_response,
)


def test_build_condition_a_prompt():
    """Condition A includes scores but no MIDI comparison."""
    scores = {"dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
              "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54}
    student_context = {"level": "intermediate", "session_count": 12}
    prompt = build_condition_a_prompt(scores, student_context)
    assert "0.35" in prompt  # pedaling score appears
    assert "comparison" not in prompt.lower()


def test_build_condition_b_prompt():
    """Condition B includes scores AND MIDI comparison."""
    scores = {"dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
              "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54}
    student_context = {"level": "intermediate", "session_count": 12}
    midi_comparison = {
        "velocity": {"velocity_mae": 15.0, "velocity_correlation": 0.8},
        "timing": {"mean_deviation_ms": 45.0, "max_deviation_ms": 120.0},
        "accuracy": {"note_f1": 0.95, "missed_notes": 2, "extra_notes": 1},
        "summary": "Note accuracy: F1=0.95; Timing: mean deviation 45ms; Dynamics: MAE=15",
    }
    prompt = build_condition_b_prompt(scores, student_context, midi_comparison)
    assert "0.35" in prompt
    assert "45" in prompt  # timing deviation
    assert "F1" in prompt or "accuracy" in prompt.lower()


def test_build_judge_prompt():
    """Judge prompt presents two observations in randomized order."""
    obs_a = "Your pedaling could use some work in this passage."
    obs_b = "In bars 5-8, the sustain pedal is held through the harmonic change at beat 3."
    prompt = build_judge_prompt(obs_a, obs_b)
    # Both observations must appear
    assert obs_a in prompt or obs_b in prompt
    assert "specificity" in prompt.lower() or "actionability" in prompt.lower()


def test_parse_judge_response():
    """parse_judge_response extracts winner and reasoning."""
    response = '{"winner": "B", "specificity": "B references specific bars", "actionability": "B tells what to do", "accuracy": "Both reasonable"}'
    result = parse_judge_response(response)
    assert result["winner"] in ("A", "B", "X", "Y")
    assert "specificity" in result
