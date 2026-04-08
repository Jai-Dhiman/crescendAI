"""Downstream impact evaluation: A/B with/without memory context.

For each scenario's final observation, generates subagent output in two variants:
- no_memory: Teaching moment + student baselines + recent observations only
- with_memory: Same + formatted Student Memory section

Calls Workers AI subagent for both, then evaluates with LLM-as-Judge (Claude Sonnet).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .eval_synthesis import _call_workers_ai
from .memory_db import MemoryDB
from .scenarios import MemoryEvalScenario, load_scenarios

DATA_DIR = Path(__file__).parents[1] / "data"
_DEV_VARS_PATH = Path(__file__).parents[3] / "api" / ".dev.vars"

# Simplified subagent prompt for downstream comparison
SUBAGENT_SYSTEM = """You are a piano pedagogy analyst. You receive structured data about a student's practice session.

Your job is to reason about which teaching moment matters most for this student right now and decide how to frame it.

Output a brief narrative paragraph (3-5 sentences) that:
1. Identifies the key issue or improvement
2. Explains why it matters for this student
3. Suggests how to frame feedback (correction, recognition, encouragement, or question)

Be specific about what you heard and reference the student's history when available."""


JUDGE_SYSTEM = """You are evaluating two responses from a piano teaching AI system.
Both responses analyze the same teaching moment for the same student.
One response had access to the student's memory (synthesized patterns, engagement history)
and one did not, but you do NOT know which is which.

Rate EACH response on 5 axes (1-5 scale):

1. Continuity: Awareness of student's history and patterns (1=generic, 5=deeply personalized)
2. Specificity: References specific moments, dimensions, trends (1=vague, 5=precise)
3. Non-repetition: Avoids repeating recent feedback (1=repetitive, 5=fresh insight)
4. Approach fit: Framing matches student's engagement patterns (1=wrong tone, 5=ideal fit)
5. Accuracy: Correct diagnosis given scores and context (1=wrong, 5=spot-on)

Output ONLY valid JSON:
```json
{
  "response_a": {"continuity": N, "specificity": N, "non_repetition": N, "approach_fit": N, "accuracy": N},
  "response_b": {"continuity": N, "specificity": N, "non_repetition": N, "approach_fit": N, "accuracy": N},
  "preferred": "a" or "b" or "tie",
  "reasoning": "Brief explanation"
}
```"""


def _build_downstream_prompt(
    scenario: MemoryEvalScenario,
    memory_context: str = "",
) -> str:
    """Build a simplified subagent prompt for downstream comparison."""
    final_obs = scenario.observations[-1]
    prompt = f"## Teaching Moment\n\n"
    prompt += f"Dimension flagged: {final_obs.dimension} (score: {final_obs.dimension_score or 0:.2f})\n"
    prompt += f'Observation: "{final_obs.observation_text}"\n\n'

    prompt += "## Student Context\n\n"
    prompt += "Baselines:\n"
    for dim, val in scenario.baselines.items():
        prompt += f"- {dim}: {val:.2f}\n"
    prompt += "\n"

    # Recent observations (last 5)
    recent = scenario.observations[-6:-1] if len(scenario.observations) > 5 else scenario.observations[:-1]
    if recent:
        prompt += "## Recent Observations\n\n"
        for obs in reversed(recent):
            engaged_label = " [student engaged]" if obs.engaged else ""
            prompt += f"- [{obs.session_date[:10]}] {obs.dimension}: \"{obs.observation_text}\" (framing: {obs.framing}{engaged_label})\n"
        prompt += "\n"

    if memory_context:
        prompt += memory_context

    prompt += "## Task\n\nAnalyze this teaching moment. What observation should the teacher make? Output a brief narrative."
    return prompt


def _call_subagent(system: str, user: str) -> str:
    return _call_workers_ai(system, user, max_tokens=400)


def _load_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_API_KEY not found in env or apps/api/.dev.vars")


def _call_judge(scenario_context: str, response_a: str, response_b: str) -> dict | None:
    import anthropic
    client = anthropic.Anthropic(api_key=_load_anthropic_key())

    user_prompt = f"""## Scenario Context

{scenario_context}

## Response A

{response_a}

## Response B

{response_b}

Rate both responses on the 5 axes. Output ONLY the JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = response.content[0].text
    # Extract JSON
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        raw = text[start:end].strip() if end > start else text.strip()
    elif "{" in text:
        start = text.find("{")
        end = text.rfind("}")
        raw = text[start:end + 1] if end > start else text.strip()
    else:
        raw = text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


@dataclass
class DownstreamResult:
    scenario_id: str
    no_memory_response: str = ""
    with_memory_response: str = ""
    judge_raw: dict | None = None
    # A/B randomization: if flipped, response_a=with_memory
    ab_flipped: bool = False
    # Derived scores
    no_memory_scores: dict = field(default_factory=dict)
    with_memory_scores: dict = field(default_factory=dict)
    memory_lift: float = 0.0
    continuity_delta: float = 0.0
    non_repetition_delta: float = 0.0
    with_memory_wins: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def run_downstream_assessment(
    scenarios: list[MemoryEvalScenario],
    live: bool = False,
    cache_path: Path | None = None,
    judge_path: Path | None = None,
) -> list[DownstreamResult]:
    if cache_path is None:
        cache_path = DATA_DIR / "downstream_cache.jsonl"
    if judge_path is None:
        judge_path = DATA_DIR / "judge_results.jsonl"

    # Load caches
    response_cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                response_cache[entry["key"]] = entry

    judge_cache: dict[str, dict] = {}
    if judge_path.exists():
        with open(judge_path) as f:
            for line in f:
                entry = json.loads(line)
                judge_cache[entry["scenario_id"]] = entry

    db = MemoryDB()
    results: list[DownstreamResult] = []
    rng = random.Random(42)

    for scenario in scenarios:
        # Build memory context from scenario data
        db.reset()
        db.insert_student(scenario.student_id, scenario.baselines)

        for obs in scenario.observations:
            db.insert_observation(
                id=obs.id, student_id=scenario.student_id,
                session_id=obs.session_id, dimension=obs.dimension,
                observation_text=obs.observation_text, framing=obs.framing,
                dimension_score=obs.dimension_score, student_baseline=obs.student_baseline,
                reasoning_trace=obs.reasoning_trace, piece_context=obs.piece_context,
                created_at=obs.session_date,
            )
            if obs.engaged:
                db.insert_teaching_approach(
                    student_id=scenario.student_id, observation_id=obs.id,
                    dimension=obs.dimension, framing=obs.framing,
                    approach_summary=f"Feedback on {obs.dimension}: {obs.observation_text[:50]}",
                    engaged=True, created_at=obs.session_date,
                )

        # Insert gold facts for with_memory variant
        for ef in scenario.expected_facts:
            if not ef.id:
                continue
            fact_text = ef.gold_fact_text if ef.gold_fact_text else ef.fact_text_pattern
            db.insert_fact(
                id=ef.id, student_id=scenario.student_id,
                fact_text=fact_text,
                fact_type=ef.fact_type, dimension=ef.dimension,
                valid_at=ef.valid_at or "2026-02-01", trend=ef.trend,
                confidence=ef.confidence, created_at=ef.valid_at or "2026-02-01",
            )

        memory_text = db.format_memory_context(scenario.student_id)

        # Generate or load responses
        cache_key_no = f"{scenario.id}_no_memory"
        cache_key_with = f"{scenario.id}_with_memory"

        if cache_key_no in response_cache and cache_key_with in response_cache:
            no_memory_response = response_cache[cache_key_no]["response"]
            with_memory_response = response_cache[cache_key_with]["response"]
        elif live:
            prompt_no = _build_downstream_prompt(scenario, memory_context="")
            prompt_with = _build_downstream_prompt(scenario, memory_context=memory_text)

            no_memory_response = _call_subagent(SUBAGENT_SYSTEM, prompt_no)
            with_memory_response = _call_subagent(SUBAGENT_SYSTEM, prompt_with)

            response_cache[cache_key_no] = {"key": cache_key_no, "response": no_memory_response}
            response_cache[cache_key_with] = {"key": cache_key_with, "response": with_memory_response}

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                f.write(json.dumps(response_cache[cache_key_no], ensure_ascii=False) + "\n")
                f.write(json.dumps(response_cache[cache_key_with], ensure_ascii=False) + "\n")
        else:
            continue

        # A/B randomization
        flipped = rng.random() < 0.5

        # Judge (or load from cache)
        if scenario.id in judge_cache:
            judge_raw = judge_cache[scenario.id].get("judge_raw")
        elif live:
            scenario_ctx = f"Student practicing piano. Category: {scenario.category}. Name: {scenario.name}."
            if flipped:
                judge_raw = _call_judge(scenario_ctx, with_memory_response, no_memory_response)
            else:
                judge_raw = _call_judge(scenario_ctx, no_memory_response, with_memory_response)

            judge_entry = {
                "scenario_id": scenario.id,
                "flipped": flipped,
                "judge_raw": judge_raw,
            }
            judge_cache[scenario.id] = judge_entry
            judge_path.parent.mkdir(parents=True, exist_ok=True)
            with open(judge_path, "a") as f:
                f.write(json.dumps(judge_entry, ensure_ascii=False) + "\n")
        else:
            judge_raw = None

        # Derive scores
        no_scores: dict = {}
        with_scores: dict = {}
        memory_lift = 0.0
        continuity_delta = 0.0
        non_rep_delta = 0.0
        with_wins = False

        if judge_raw:
            a_scores = judge_raw.get("response_a", {})
            b_scores = judge_raw.get("response_b", {})

            if flipped:
                with_scores = a_scores
                no_scores = b_scores
            else:
                no_scores = a_scores
                with_scores = b_scores

            axes = ["continuity", "specificity", "non_repetition", "approach_fit", "accuracy"]
            with_mean = sum(with_scores.get(a, 3) for a in axes) / len(axes)
            no_mean = sum(no_scores.get(a, 3) for a in axes) / len(axes)
            memory_lift = with_mean - no_mean
            continuity_delta = with_scores.get("continuity", 3) - no_scores.get("continuity", 3)
            non_rep_delta = with_scores.get("non_repetition", 3) - no_scores.get("non_repetition", 3)

            with_total = sum(with_scores.get(a, 0) for a in axes)
            no_total = sum(no_scores.get(a, 0) for a in axes)
            with_wins = with_total > no_total

        results.append(DownstreamResult(
            scenario_id=scenario.id,
            no_memory_response=no_memory_response,
            with_memory_response=with_memory_response,
            judge_raw=judge_raw,
            ab_flipped=flipped,
            no_memory_scores=no_scores,
            with_memory_scores=with_scores,
            memory_lift=memory_lift,
            continuity_delta=continuity_delta,
            non_repetition_delta=non_rep_delta,
            with_memory_wins=with_wins,
        ))

    return results


def print_results(results: list[DownstreamResult]) -> None:
    print("\n=== Downstream Impact Assessment ===\n")

    if not results:
        print("  No results. Run with --live to generate.")
        return

    judged = [r for r in results if r.judge_raw]
    if not judged:
        print("  No judge results. Run with --live to judge.")
        for r in results:
            print(f"  {r.scenario_id}: responses cached (no_memory: {len(r.no_memory_response)} chars, with_memory: {len(r.with_memory_response)} chars)")
        return

    win_count = 0
    all_lift = []
    all_cont = []
    all_nonrep = []

    for r in judged:
        status = "WIN" if r.with_memory_wins else "LOSS"
        print(f"  [{status}] {r.scenario_id}: lift={r.memory_lift:+.2f} cont={r.continuity_delta:+.1f} nonrep={r.non_repetition_delta:+.1f}")
        if r.with_memory_wins:
            win_count += 1
        all_lift.append(r.memory_lift)
        all_cont.append(r.continuity_delta)
        all_nonrep.append(r.non_repetition_delta)

    n = len(judged)
    print(f"\n--- Aggregate (n={n}) ---")
    print(f"  Mean memory lift:      {sum(all_lift)/n:+.3f}")
    print(f"  Mean continuity delta: {sum(all_cont)/n:+.3f}")
    print(f"  Mean non-rep delta:    {sum(all_nonrep)/n:+.3f}")
    print(f"  Win rate:              {win_count}/{n} ({win_count/n:.1%})")


def main() -> None:
    import sys
    live = "--live" in sys.argv
    scenarios_path = DATA_DIR / "scenarios.jsonl"
    scenarios = load_scenarios(scenarios_path)
    results = run_downstream_assessment(scenarios, live=live)
    print_results(results)


if __name__ == "__main__":
    main()
