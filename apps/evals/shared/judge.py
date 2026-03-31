from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Auto-load ANTHROPIC_API_KEY from .dev.vars if not in environment
if not os.environ.get("ANTHROPIC_API_KEY"):
    _dev_vars = Path(__file__).parents[2] / "api" / ".dev.vars"
    if _dev_vars.exists():
        for line in _dev_vars.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                break

import anthropic

# Import LLMClient for Workers AI support (v2 judge functions)
from teaching_knowledge.llm_client import LLMClient, strip_json_fences


JUDGE_SYSTEM_MESSAGE = (
    "You are an evaluation judge. For each criterion, you MUST format your "
    "response exactly as:\n\n"
    "**[Criterion Name]:** YES or NO\n"
    "Evidence: \"your evidence here\"\n\n"
    "Do not deviate from this format."
)

DEFAULT_MODEL = "claude-sonnet-4-6"

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class CriterionScore:
    criterion: str
    passed: bool | None
    evidence: str
    raw_response: str


@dataclass
class DimensionScore:
    """Score for a v2 rubric dimension (0-3 scale, or None for N/A)."""
    criterion: str
    score: int | None
    evidence: str
    reason: str


@dataclass
class JudgeResult:
    scores: list[CriterionScore] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    latency_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        evaluated = [s for s in self.scores if s.passed is not None]
        if not evaluated:
            return 0.0
        return sum(1 for s in evaluated if s.passed) / len(evaluated)


@dataclass
class JudgeResultV2:
    """Result from v2 rubric judge (0-3 scale per dimension)."""
    dimensions: list[DimensionScore] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    latency_ms: float = 0.0

    @property
    def mean_score(self) -> float:
        scores = [d.score for d in self.dimensions if d.score is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @property
    def scored_dimension_count(self) -> int:
        return sum(1 for d in self.dimensions if d.score is not None)

    @property
    def scores_by_dimension(self) -> dict[str, int | None]:
        return {d.criterion: d.score for d in self.dimensions}


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the shared/prompts/ directory."""
    path = _PROMPTS_DIR / prompt_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def judge_observation(
    observation_text: str,
    context: dict[str, Any],
    prompt_file: str = "observation_quality_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge a teacher observation using the specified prompt and model."""
    template = load_prompt(prompt_file)

    # Build format kwargs from context, with defaults for standard fields
    format_kwargs = {
        "piece_name": context.get("piece_name", "Unknown"),
        "bar_range": context.get("bar_range", "Unknown"),
        "predictions": context.get("predictions", "{}"),
        "baselines": context.get("baselines", "{}"),
        "recent_observations": context.get("recent_observations", "[]"),
        "analysis_facts": context.get("analysis_facts", "None"),
        "observation_text": observation_text,
    }
    # Forward any additional context keys (for v2 prompts)
    for key, value in context.items():
        if key not in format_kwargs:
            format_kwargs[key] = value

    user_message = template.format(**format_kwargs)

    client = anthropic.Anthropic()
    start = time.monotonic()
    response_text = _call_with_retry(client, model, user_message)
    latency_ms = (time.monotonic() - start) * 1000

    scores = _parse_judge_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def judge_subagent(
    subagent_output: dict[str, Any],
    context: dict[str, Any],
    prompt_file: str = "subagent_reasoning_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge a subagent's teaching moment analysis."""
    template = load_prompt(prompt_file)
    user_message = template.format(
        predictions=context.get("predictions", "{}"),
        baselines=context.get("baselines", "{}"),
        recent_observations=context.get("recent_observations", "[]"),
        dimension=subagent_output.get("dimension", "Unknown"),
        framing=subagent_output.get("framing", "Unknown"),
        reasoning_trace=subagent_output.get("reasoning_trace", ""),
    )

    client = anthropic.Anthropic()
    start = time.monotonic()
    response_text = _call_with_retry(client, model, user_message)
    latency_ms = (time.monotonic() - start) * 1000

    scores = _parse_judge_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def judge_synthesis(
    synthesis_text: str,
    context: dict[str, Any],
    prompt_file: str = "synthesis_quality_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge a post-session synthesis using the specified prompt and model."""
    template = load_prompt(prompt_file)

    format_kwargs = {
        "piece_name": context.get("piece_name", "Unknown"),
        "composer": context.get("composer", "Unknown"),
        "skill_level": context.get("skill_level", "Unknown"),
        "synthesis_text": synthesis_text,
        "drilling_detected": context.get("drilling_detected", False),
        "drilling_passage": context.get("drilling_passage", "None"),
    }
    for key, value in context.items():
        if key not in format_kwargs:
            format_kwargs[key] = value

    user_message = template.format(**format_kwargs)

    client = anthropic.Anthropic()
    start = time.monotonic()
    response_text = _call_with_retry(client, model, user_message)
    latency_ms = (time.monotonic() - start) * 1000

    scores = _parse_judge_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def judge_teaching_moment(
    context: dict[str, Any],
    prompt_file: str = "teaching_moment_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge teaching moment selection using the specified prompt and model."""
    template = load_prompt(prompt_file)

    format_kwargs = {
        "piece_name": context.get("piece_name", "Unknown"),
        "composer": context.get("composer", "Unknown"),
        "all_moments": context.get("all_moments", "[]"),
        "selected_dimension": context.get("selected_dimension", "Unknown"),
        "deviation": context.get("deviation", "Unknown"),
        "score": context.get("score", "Unknown"),
    }
    for key, value in context.items():
        if key not in format_kwargs:
            format_kwargs[key] = value

    user_message = template.format(**format_kwargs)

    client = anthropic.Anthropic()
    start = time.monotonic()
    response_text = _call_with_retry(client, model, user_message)
    latency_ms = (time.monotonic() - start) * 1000

    scores = _parse_judge_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def judge_differentiation(
    syntheses: list[tuple[int, str]],
    piece_name: str,
    composer: str,
    prompt_file: str = "differentiation_judge_v1.txt",
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Judge differentiation across skill levels using the specified prompt and model.

    Args:
        syntheses: List of (skill_level, synthesis_text) tuples, expected for levels 1, 3, 5.
        piece_name: Name of the piece being evaluated.
        composer: Composer of the piece.
        prompt_file: Prompt template filename.
        model: Anthropic model to use.
    """
    template = load_prompt(prompt_file)

    # Build kwargs from the 3 expected skill levels
    format_kwargs: dict[str, Any] = {
        "piece_name": piece_name,
        "composer": composer,
    }
    for skill_level, synthesis_text in syntheses:
        format_kwargs[f"synthesis_{skill_level}"] = synthesis_text
        format_kwargs[f"skill_level_{skill_level}"] = skill_level

    user_message = template.format(**format_kwargs)

    client = anthropic.Anthropic()
    start = time.monotonic()
    response_text = _call_with_retry(client, model, user_message)
    latency_ms = (time.monotonic() - start) * 1000

    scores = _parse_judge_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def judge_synthesis_v2(
    synthesis_text: str,
    context: dict[str, Any],
    prompt_file: str = "synthesis_quality_judge_v2.txt",
    provider: str = "workers-ai",
    model: str | None = None,
) -> JudgeResultV2:
    """Judge a synthesis using the v2 rubric (0-3 scale, 7 dimensions).

    Uses Workers AI by default. Pass provider="anthropic" to use Claude.
    The v2 prompt outputs a JSON array of {criterion, score, evidence, reason}.
    """
    template = load_prompt(prompt_file)

    # v2 prompt is self-contained (rubric baked in), just needs the synthesis
    user_message = (
        f"{template}\n\n"
        f"## Context\n"
        f"Piece: {context.get('piece_name', 'Unknown')} by {context.get('composer', 'Unknown')}\n"
        f"Student skill level: {context.get('skill_level', 'Unknown')}\n\n"
        f"## AI Teacher Output to Evaluate\n"
        f"{synthesis_text}"
    )

    client = LLMClient(provider=provider, model=model, tier="quality")
    start = time.monotonic()
    response_text = client.complete_json(user_message, max_tokens=4000)
    latency_ms = (time.monotonic() - start) * 1000

    dimensions = _parse_v2_response(response_text)
    prompt_version = prompt_file.replace(".txt", "")

    return JudgeResultV2(
        dimensions=dimensions,
        model=client.model,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


def _parse_v2_response(response_text: str) -> list[DimensionScore]:
    """Parse v2 judge JSON response into DimensionScore list.

    Handles score values of 0-3 (int) or "N/A" (mapped to None).
    """
    try:
        data = json.loads(response_text)
        if not isinstance(data, list):
            data = [data]
        dimensions = []
        for entry in data:
            raw_score = entry.get("score")
            if isinstance(raw_score, str) and raw_score.strip().upper() == "N/A":
                score: int | None = None
            else:
                score = int(raw_score) if raw_score is not None else None
            dimensions.append(
                DimensionScore(
                    criterion=entry.get("criterion", "unknown"),
                    score=score,
                    evidence=entry.get("evidence", ""),
                    reason=entry.get("reason", ""),
                )
            )
        return dimensions
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        return [
            DimensionScore(
                criterion="parse_failure",
                score=0,
                evidence=f"Could not parse v2 judge response: {e}",
                reason=response_text[:200],
            )
        ]


def _call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    user_message: str,
    max_retries: int = 3,
) -> str:
    """Call Claude API with exponential backoff on 429/529 errors."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=JUDGE_SYSTEM_MESSAGE,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = 2 ** (attempt + 1)
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                delay = 2 ** (attempt + 1)
                time.sleep(delay)
            else:
                raise

    raise RuntimeError("Unreachable: exhausted retries without returning or raising")


def _parse_judge_response(response_text: str) -> list[CriterionScore]:
    """Parse judge response into CriterionScore list.

    Expects lines like:
        **Criterion Name:** YES
        Evidence: "some evidence here"
    """
    scores: list[CriterionScore] = []

    # Match criterion blocks: **Name:** YES/NO/HIGH/MEDIUM/LOW/N/A followed by Evidence line
    pattern = re.compile(
        r"\*\*([^*]+?):\*\*\s*(YES|NO|HIGH|MEDIUM|LOW|N/A)\s*\n"
        r'(?:Evidence:\s*"?(.+?)"?\s*(?:\n|$))',
        re.IGNORECASE,
    )

    matches = pattern.findall(response_text)

    if not matches:
        scores.append(
            CriterionScore(
                criterion="parse_failure",
                passed=None,
                evidence=f"Could not parse judge response: {response_text[:200]}",
                raw_response=response_text,
            )
        )
        return scores

    for criterion, verdict, evidence in matches:
        upper_verdict = verdict.upper()
        if upper_verdict in ("YES", "HIGH", "MEDIUM"):
            passed = True
        elif upper_verdict == "N/A":
            passed = None
        else:
            # NO or LOW
            passed = False
        scores.append(
            CriterionScore(
                criterion=criterion.strip(),
                passed=passed,
                evidence=evidence.strip(),
                raw_response=response_text,
            )
        )

    return scores
