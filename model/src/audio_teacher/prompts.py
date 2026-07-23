"""Structured elicitation for contrast axes.

The question always asks which clip exhibits the DEGRADATION on the
axis, so downstream a response is correct iff the parsed choice equals
ContrastPair.degraded.
"""
from __future__ import annotations

import re

AXIS_QUESTIONS = {
    "pedaling": (
        "One of these two piano recordings is over-pedaled: excessive sustain "
        "pedal blurs harmonies and note attacks together. Which one?"
    ),
    "dynamics": (
        "One of these two piano recordings has flat, unshaped dynamics: little "
        "contrast between loud and soft, and no dynamic direction across the "
        "phrase. Which one?"
    ),
    "phrasing": (
        "One of these two piano recordings has weak phrasing: no breathing "
        "between phrases, uniform note weight, and no sense of line. Which one?"
    ),
}

ANSWER_INSTRUCTION = (
    "Listen to Clip A, then Clip B. Explain briefly, then end with a final "
    'line of exactly "ANSWER: A" or "ANSWER: B".'
)


def build_question(axis: str) -> str:
    if axis not in AXIS_QUESTIONS:
        raise KeyError(
            f"no elicitation question for axis {axis!r}; known: {sorted(AXIS_QUESTIONS)}"
        )
    return f"{AXIS_QUESTIONS[axis]}\n\n{ANSWER_INSTRUCTION}"


_ANSWER_RE = re.compile(r"ANSWER:\s*([AB])\b", re.IGNORECASE)


def parse_choice(text: str) -> str | None:
    """Extract the forced A/B choice from a model response.

    The last ANSWER: line wins (models sometimes revise). Returns "a",
    "b", or None when no well-formed answer exists -- the scorer counts
    None as unparseable, which pushes the gate toward FAIL (closed).
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()
