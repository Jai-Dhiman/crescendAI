"""LLM feedback assessment for Experiment 4 (MIDI-as-context test).

Generates teacher observations under two conditions:
  A: A1 dimension scores + student context only
  B: A1 dimension scores + student context + structured MIDI comparison

Then uses an LLM judge to determine which produces more specific, actionable feedback.
"""

from __future__ import annotations

import json
import logging
import random

logger = logging.getLogger(__name__)

_SUBAGENT_SYSTEM = """You are a piano teaching assistant analyzing a practice moment.
Given the student's context and performance data, generate a brief, specific
teaching observation (2-3 sentences) that a piano teacher would say.

Focus on one dimension. Be specific about what you hear and what to try next.
Reference passage location if possible. Speak warmly but directly."""


def build_condition_a_prompt(
    scores: dict[str, float],
    student_context: dict,
) -> str:
    """Build teacher prompt with A1 scores only (no MIDI data).

    Args:
        scores: {dimension_name: float} for 6 dimensions.
        student_context: {level, session_count, goals, ...}.

    Returns:
        Complete prompt string for the teacher LLM.
    """
    scores_text = "\n".join(f"  {dim}: {val:.2f}" for dim, val in scores.items())
    weakest_dim = min(scores, key=scores.get)

    return f"""{_SUBAGENT_SYSTEM}

## Student Context
- Level: {student_context.get('level', 'unknown')}
- Session count: {student_context.get('session_count', 0)}
- Goals: {student_context.get('goals', 'none specified')}

## Performance Scores (0-1 scale, higher = better)
{scores_text}

The weakest dimension is **{weakest_dim}** ({scores[weakest_dim]:.2f}).

Generate a teaching observation focused on {weakest_dim}."""


def build_condition_b_prompt(
    scores: dict[str, float],
    student_context: dict,
    midi_data: dict,
) -> str:
    """Build teacher prompt with A1 scores AND structured MIDI data.

    Args:
        scores: {dimension_name: float} for 6 dimensions.
        student_context: {level, session_count, goals, ...}.
        midi_data: Output of structured_midi_comparison().

    Returns:
        Complete prompt string for the teacher LLM.
    """
    scores_text = "\n".join(f"  {dim}: {val:.2f}" for dim, val in scores.items())
    weakest_dim = min(scores, key=scores.get)

    midi_summary = midi_data.get("summary", "")
    velocity_info = midi_data.get("velocity", {})
    timing_info = midi_data.get("timing", {})
    accuracy_info = midi_data.get("accuracy", {})

    midi_detail = f"""## MIDI Analysis (performance vs score reference)
{midi_summary}

### Velocity (dynamics)
- MAE: {velocity_info.get('velocity_mae', 0):.0f}
- Correlation with score dynamics: {velocity_info.get('velocity_correlation', 0):.2f}
- Performance mean velocity: {velocity_info.get('perf_velocity_mean', 0):.0f}

### Timing
- Mean onset deviation: {timing_info.get('mean_deviation_ms', 0):.0f}ms
- Max onset deviation: {timing_info.get('max_deviation_ms', 0):.0f}ms
- Systematic tendency: {timing_info.get('mean_signed_deviation_ms', 0):+.0f}ms (positive = rushing)

### Note Accuracy
- F1: {accuracy_info.get('note_f1', 0):.2f}
- Missed notes: {accuracy_info.get('missed_notes', 0)}
- Extra notes: {accuracy_info.get('extra_notes', 0)}"""

    return f"""{_SUBAGENT_SYSTEM}

## Student Context
- Level: {student_context.get('level', 'unknown')}
- Session count: {student_context.get('session_count', 0)}
- Goals: {student_context.get('goals', 'none specified')}

## Performance Scores (0-1 scale, higher = better)
{scores_text}

The weakest dimension is **{weakest_dim}** ({scores[weakest_dim]:.2f}).

{midi_detail}

Use the MIDI analysis data to make your observation more specific.
Reference concrete details (e.g., velocity differences, timing deviations).
Generate a teaching observation focused on {weakest_dim}."""


def build_judge_prompt(
    observation_a: str,
    observation_b: str,
    randomize: bool = True,
) -> str:
    """Build LLM judge prompt comparing two observations.

    Presents observations in randomized order to avoid position bias.

    Args:
        observation_a: Teacher observation from Condition A.
        observation_b: Teacher observation from Condition B.
        randomize: Whether to randomize presentation order.

    Returns:
        Judge prompt string. The caller tracks which is which.
    """
    if randomize and random.random() < 0.5:
        first, second = observation_b, observation_a
    else:
        first, second = observation_a, observation_b

    return f"""You are judging two piano teaching observations for the same practice moment.
Rate which observation is better on three criteria:

1. **Specificity**: Does it reference particular passages, bars, or musical details?
2. **Actionability**: Does it tell the student what to do differently?
3. **Accuracy**: Does the observation sound musically plausible and precise?

## Observation X
{first}

## Observation Y
{second}

Respond with JSON only:
{{
    "winner": "X" or "Y",
    "specificity": "Which is more specific and why (1 sentence)",
    "actionability": "Which is more actionable and why (1 sentence)",
    "accuracy": "Which sounds more accurate and why (1 sentence)",
    "confidence": "high" or "medium" or "low"
}}"""


def parse_judge_response(response: str) -> dict:
    """Parse judge LLM response into structured result.

    Args:
        response: Raw LLM response (should be JSON).

    Returns:
        Parsed dict with winner, specificity, actionability, accuracy.

    Raises:
        ValueError: If response cannot be parsed as JSON.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    parsed = json.loads(text)

    required = {"winner", "specificity", "actionability", "accuracy"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Judge response missing keys: {missing}")

    return parsed
