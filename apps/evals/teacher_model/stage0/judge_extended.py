"""Extended judge: 7 base rubric dims + Taste defensibility + Adaptation specificity.

Reuses the existing apps/evals/shared/judge.py infrastructure (DimensionScore,
LLMClient, JSON-fence stripping). The only differences vs judge_synthesis_v2
are (a) the prompt file and (b) the parser tolerates 9 dims instead of 7.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the canonical DimensionScore + LLMClient + fence stripper.
from shared.judge import DimensionScore  # noqa: F401  (re-exported for callers)
from teaching_knowledge.llm_client import LLMClient, strip_json_fences

EXTENDED_DIMS: list[str] = [
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

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_v2_extended.txt"


@dataclass
class JudgeResultV2Extended:
    dimensions: list[DimensionScore]
    model: str
    prompt_version: str
    latency_ms: float


class JudgeParseError(RuntimeError):
    """Raised when the extended judge response cannot be parsed into 9 dims."""


def _coerce_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().upper() in ("N/A", "NA", ""):
            return None
        try:
            value = int(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if 0 <= iv <= 3:
            return iv
    return None


def parse_extended_judge_response(response_text: str) -> list[DimensionScore]:
    """Parse a v2-extended judge JSON-array response into 9 DimensionScore rows."""
    text = strip_json_fences(response_text)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"judge response was not valid JSON: {e}") from e
    if not isinstance(raw, list):
        raise JudgeParseError(f"judge response must be a JSON array, got {type(raw).__name__}")

    by_criterion: dict[str, dict] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("criterion"), str):
            by_criterion[item["criterion"]] = item

    out: list[DimensionScore] = []
    for dim_name in EXTENDED_DIMS:
        item = by_criterion.get(dim_name)
        if item is None:
            raise JudgeParseError(f"missing dimension in judge response: {dim_name!r}")
        process = _coerce_score(item.get("process"))
        outcome = _coerce_score(item.get("outcome"))
        if process is not None and outcome is not None:
            score = min(process, outcome)
        else:
            score = process if process is not None else outcome
        out.append(
            DimensionScore(
                criterion=dim_name,
                score=score,
                evidence=str(item.get("evidence", ""))[:500],
                reason=str(item.get("reason", ""))[:1000],
                process=process,
                outcome=outcome,
            )
        )
    return out


def judge_extended(
    synthesis_text: str,
    context: dict[str, Any],
    provider: str = "workers-ai",
    model: str | None = None,
) -> JudgeResultV2Extended:
    """Judge a synthesis using the v2-extended rubric (9 dimensions)."""
    template = _PROMPT_PATH.read_text()
    user_message = (
        f"{template}\n\n"
        f"## Context\n"
        f"Piece: {context.get('piece_name', 'Unknown')} by {context.get('composer', 'Unknown')}\n"
        f"Student skill level: {context.get('skill_level', 'Unknown')}\n\n"
        f"## AI Teacher Output to Evaluate\n"
        f"{synthesis_text}"
    )

    client = LLMClient(provider=provider, model=model, tier="judge")
    start = time.monotonic()
    response_text = client.complete_json(user_message, max_tokens=4000)
    latency_ms = (time.monotonic() - start) * 1000

    dimensions = parse_extended_judge_response(response_text)
    return JudgeResultV2Extended(
        dimensions=dimensions,
        model=client.model,
        prompt_version="judge_v2_extended",
        latency_ms=latency_ms,
    )
