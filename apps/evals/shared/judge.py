from __future__ import annotations

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

    # Match criterion blocks: **Name:** YES/NO followed by Evidence line
    pattern = re.compile(
        r"\*\*([^*]+?):\*\*\s*(YES|NO)\s*\n"
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
        scores.append(
            CriterionScore(
                criterion=criterion.strip(),
                passed=verdict.upper() == "YES",
                evidence=evidence.strip(),
                raw_response=response_text,
            )
        )

    return scores
