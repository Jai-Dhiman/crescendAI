"""Subagent reasoning evaluation.

Tests the Workers AI subagent's ability to select the right dimension,
choose appropriate framing, and produce coherent reasoning from
hand-crafted scenarios.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

# Add apps/evals/ to path for shared imports
sys.path.insert(0, str(Path(__file__).parents[2]))

import requests

from shared.judge import judge_subagent
from shared.reporting import EvalReport, MetricResult

SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "scenarios.json"

_DEV_VARS_PATH = Path(__file__).parents[3] / "api" / ".dev.vars"
DEFAULT_CF_ACCOUNT_ID = "5df63f40beeab277db407f1ecbd6e1ec"
DEFAULT_GATEWAY_ID = "crescendai-background"
_WORKERS_AI_MODEL = "@cf/google/gemma-4-26b-a4b-it"


def _load_cf_token() -> str:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if token:
        return token
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("CLOUDFLARE_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("CLOUDFLARE_API_TOKEN not found in env or apps/api/.dev.vars")


def load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def build_subagent_prompt(scenario: dict) -> str:
    """Build the subagent prompt matching production format."""
    predictions = scenario["predictions"]
    baselines = scenario.get("baselines", {})
    recent = scenario.get("recent_observations", [])
    recent_dims = ", ".join(o["dimension"] for o in recent) if recent else "none"

    deviations = {}
    for dim, score in predictions.items():
        baseline = baselines.get(dim)
        if baseline is not None:
            deviations[dim] = round(score - baseline, 3)

    return f"""You are a piano teaching assistant analyzing a student's performance.

## Performance Scores (0-1, higher = better)
{json.dumps(predictions, indent=2)}

## Student Baseline (running average)
{json.dumps(baselines, indent=2) if baselines else "No baseline yet (first session)"}

## Deviations from Baseline
{json.dumps(deviations, indent=2) if deviations else "N/A (no baseline)"}

## Recently Covered Dimensions (last 3 observations)
{recent_dims}

## Your Task
Select ONE dimension to teach about and provide:
1. The dimension name
2. A framing: "correction" (student declining), "recognition" (student improving), "encouragement" (stable/first session), or "question" (exploratory)
3. A brief reasoning trace explaining your choice

Respond in this exact JSON format:
{{
  "dimension": "<dimension_name>",
  "framing": "<correction|recognition|encouragement|question>",
  "reasoning_trace": "<1-2 sentences explaining your selection>"
}}"""


def call_subagent(prompt: str) -> dict:
    """Call Workers AI subagent with the subagent prompt."""
    token = _load_cf_token()
    account_id = os.environ.get("CF_ACCOUNT_ID", DEFAULT_CF_ACCOUNT_ID)
    gateway_id = os.environ.get("CF_GATEWAY_ID", DEFAULT_GATEWAY_ID)
    url = (
        f"https://gateway.ai.cloudflare.com/v1/"
        f"{account_id}/{gateway_id}/workers-ai/v1/chat/completions"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": _WORKERS_AI_MODEL,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    if content is None:
        raise RuntimeError(
            f"Workers AI returned null content: {json.dumps(data)[:300]}"
        )
    text = content.strip()

    # Parse JSON from response (handle markdown code blocks)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


def main(reports_dir: Path) -> EvalReport:
    """Run the subagent reasoning eval."""
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    dimension_correct = []
    framing_correct = []
    reasoning_coherent = []
    total_judge_calls = 0

    for i, scenario in enumerate(scenarios):
        print(f"  [{i + 1}/{len(scenarios)}] {scenario['id']}...", end=" ", flush=True)

        prompt = build_subagent_prompt(scenario)
        try:
            output = call_subagent(prompt)
        except (json.JSONDecodeError, Exception) as e:
            print(f"SKIP (subagent error: {e})")
            continue

        # Dimension selection (deterministic check)
        expected_dim = scenario.get("expected_dimension")
        dim_match = True  # default for null expected_dimension (any is acceptable)
        if expected_dim is not None:
            dim_match = output.get("dimension") == expected_dim
        dimension_correct.append(dim_match)

        # Framing match (deterministic check)
        expected_framing = scenario.get("expected_framing")
        if expected_framing:
            framing_match = output.get("framing") == expected_framing
            framing_correct.append(framing_match)

        # Reasoning coherence (LLM judge)
        judge_result = judge_subagent(output, scenario)
        total_judge_calls += 1

        coherence_scores = [
            s for s in judge_result.scores if "coherence" in s.criterion.lower()
        ]
        if coherence_scores and coherence_scores[0].passed is not None:
            reasoning_coherent.append(coherence_scores[0].passed)

        status = "PASS" if (expected_dim is None or dim_match) else "FAIL"
        print(
            f"{status} (picked: {output.get('dimension')}, framing: {output.get('framing')})"
        )

    report = EvalReport(
        eval_name="subagent_reasoning",
        eval_version="1.0",
        dataset=f"scenarios_{len(scenarios)}",
        metrics={},
    )

    if dimension_correct:
        report.metrics["dimension_selection"] = MetricResult(
            mean=sum(dimension_correct) / len(dimension_correct),
            std=statistics.stdev(dimension_correct)
            if len(dimension_correct) > 1
            else 0.0,
            n=len(dimension_correct),
            pass_threshold=0.80,
        )
    if framing_correct:
        report.metrics["framing_match"] = MetricResult(
            mean=sum(framing_correct) / len(framing_correct),
            std=statistics.stdev(framing_correct) if len(framing_correct) > 1 else 0.0,
            n=len(framing_correct),
            pass_threshold=0.75,
        )
    if reasoning_coherent:
        report.metrics["reasoning_coherence"] = MetricResult(
            mean=sum(reasoning_coherent) / len(reasoning_coherent),
            std=statistics.stdev(reasoning_coherent)
            if len(reasoning_coherent) > 1
            else 0.0,
            n=len(reasoning_coherent),
            pass_threshold=0.70,
        )

    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": total_judge_calls * 0.003,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report.save(reports_dir / "subagent_reasoning.json")
    report.print_summary()
    return report
