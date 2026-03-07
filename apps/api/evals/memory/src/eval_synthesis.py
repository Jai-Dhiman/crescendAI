"""Synthesis evaluation: test fact creation against gold-standard expected facts.

Replicates build_synthesis_prompt from prompts.rs in Python. Calls Groq
(or uses cached responses). Compares output against gold-standard expected facts.

Two modes: offline (cached JSONL) and live (Groq API, populates cache).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .memory_db import MemoryDB, SynthesizedFact
from .scenarios import MemoryEvalScenario, load_scenarios

DATA_DIR = Path(__file__).parents[1] / "data"
_DEV_VARS_PATH = Path(__file__).parents[3] / ".dev.vars"

# Exact replica of prompts.rs SYNTHESIS_SYSTEM
SYNTHESIS_SYSTEM = """You are a memory consolidation system for a piano teaching app. You receive:
1. Current active facts about a student (what the system currently believes)
2. New observations since the last synthesis (what was recently observed)
3. Teaching approach records (what feedback was given and whether the student engaged)
4. Student baselines (current dimension scores)

Your job is to update the student's fact base. Output ONLY valid JSON with three arrays:

```json
{
  "new_facts": [
    {
      "fact_text": "One sentence describing the pattern or insight",
      "fact_type": "dimension|approach|arc|student_reported",
      "dimension": "dynamics|timing|pedaling|articulation|phrasing|interpretation|null",
      "piece_context": {"composer": "...", "title": "..."} or null,
      "trend": "improving|stable|declining|new|resolved",
      "confidence": "high|medium|low",
      "evidence": ["obs_id_1", "obs_id_2"]
    }
  ],
  "invalidated_facts": [
    {
      "fact_id": "id of fact to invalidate",
      "reason": "Why this fact is no longer true",
      "invalid_at": "ISO date when it stopped being true"
    }
  ],
  "unchanged_facts": ["fact_id_1", "fact_id_2"]
}
```

Rules:
- Every current active fact must appear in either invalidated_facts or unchanged_facts
- Create approach facts when engagement patterns are clear (e.g., "student engages most with correction-framed feedback")
- Invalidate facts that are contradicted by new evidence (e.g., a "persistent weakness" that has improved for 3+ sessions)
- Set trend to "resolved" when a previously flagged issue is no longer appearing
- Be conservative: only create high-confidence facts when supported by 3+ observations
- Review student_reported facts for staleness (goals older than 90 days with no related observations)"""


def build_synthesis_prompt(
    active_facts: list[SynthesizedFact],
    new_observations: list[dict],
    teaching_approaches: list[dict],
    baselines: dict,
) -> str:
    """Python replica of prompts.rs build_synthesis_prompt."""
    prompt = "## Current Active Facts\n\n"

    if not active_facts:
        prompt += "No facts yet (first synthesis).\n\n"
    else:
        for fact in active_facts:
            dim = fact.dimension or "general"
            trend = fact.trend or "unknown"
            prompt += (
                f"- [id: {fact.id}, type: {fact.fact_type}, dim: {dim}, "
                f"trend: {trend}, confidence: {fact.confidence}, "
                f"since: {fact.valid_at}] {fact.fact_text}\n"
            )
        prompt += "\n"

    prompt += "## New Observations Since Last Synthesis\n\n"
    if not new_observations:
        prompt += "No new observations.\n\n"
    else:
        for obs in new_observations:
            obs_id = obs.get("id", "")
            dim = obs.get("dimension", "")
            text = obs.get("observation_text", "")
            framing = obs.get("framing", "")
            score = obs.get("dimension_score")
            baseline = obs.get("student_baseline")
            created = obs.get("created_at", "")
            trace = obs.get("reasoning_trace", "")

            prompt += f"- [id: {obs_id}, dim: {dim}, framing: {framing}, date: {created}]\n"
            prompt += f'  Text: "{text}"\n'
            if score is not None and baseline is not None:
                delta = score - baseline
                prompt += f"  Score: {score:.2f} (baseline: {baseline:.2f}, delta: {delta:+.2f})\n"
            if trace and trace != "{}":
                prompt += f"  Reasoning: {trace}\n"
        prompt += "\n"

    if teaching_approaches:
        prompt += "## Teaching Approaches\n\n"
        for ta in teaching_approaches:
            dim = ta.get("dimension", "")
            summary = ta.get("approach_summary", "")
            engaged = ta.get("engaged", 0) == 1
            prompt += f"- {dim}: {summary} (engaged: {'yes' if engaged else 'no'})\n"
        prompt += "\n"

    prompt += "## Student Baselines\n\n"
    for dim in ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]:
        val = baselines.get(f"baseline_{dim}")
        if val is not None:
            prompt += f"- {dim}: {val:.2f}\n"
    prompt += "\n"

    prompt += "## Task\n\nAnalyze the new observations against current facts and baselines. Output the JSON update."

    return prompt


def _load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not found in env or apps/api/.dev.vars")


def _call_groq(system: str, user: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
    """Call Groq API (OpenAI-compatible)."""
    import groq
    client = groq.Groq(api_key=_load_groq_key())
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _extract_json(output: str) -> dict | None:
    """Extract JSON from LLM output (may be in code fences)."""
    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            raw = output[start:end].strip()
        else:
            raw = output.strip()
    elif "{" in output:
        start = output.find("{")
        end = output.rfind("}")
        if end > start:
            raw = output[start:end + 1]
        else:
            raw = output.strip()
    else:
        raw = output.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


@dataclass
class SynthesisResult:
    scenario_id: str
    checkpoint_index: int
    json_parsed: bool = False
    raw_output: str = ""
    new_fact_recall: float = 0.0
    new_fact_precision: float = 0.0
    hallucination_rate: float = 0.0
    invalidation_recall: float = 0.0
    invalidation_precision: float = 0.0
    fact_type_accuracy: float = 0.0
    trend_accuracy: float = 0.0
    evidence_grounded: bool = True
    produced_facts: list[dict] = field(default_factory=list)
    matched_expected: list[str] = field(default_factory=list)
    unmatched_expected: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _match_fact(produced: dict, expected_facts: list, observation_ids: set[str]) -> tuple[str | None, bool, bool]:
    """Match a produced fact against expected facts.

    Returns (matched_expected_id, type_correct, trend_correct).
    """
    fact_text = produced.get("fact_text", "")
    fact_type = produced.get("fact_type", "")

    for ef in expected_facts:
        try:
            if re.search(ef.fact_text_pattern, fact_text, re.IGNORECASE):
                type_ok = ef.fact_type == fact_type
                trend_ok = ef.trend is None or ef.trend == produced.get("trend")
                return ef.id, type_ok, trend_ok
        except re.error:
            continue

    return None, False, False


def _is_grounded(produced: dict, observation_ids: set[str]) -> bool:
    """Check if evidence array contains valid observation IDs."""
    evidence = produced.get("evidence", [])
    if not isinstance(evidence, list):
        return False
    return all(eid in observation_ids for eid in evidence)


def run_synthesis_assessment(
    scenarios: list[MemoryEvalScenario],
    live: bool = False,
    cache_path: Path | None = None,
) -> list[SynthesisResult]:
    """Run synthesis assessment.

    Args:
        scenarios: Scenarios to test.
        live: If True, call Groq API. If False, use cached responses.
        cache_path: Path to cache JSONL file.
    """
    if cache_path is None:
        cache_path = DATA_DIR / "synthesis_cache.jsonl"

    # Load cache
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["key"]] = entry

    db = MemoryDB()
    results: list[SynthesisResult] = []

    for scenario in scenarios:
        db.reset()
        db.insert_student(scenario.student_id, scenario.baselines)

        observation_ids: set[str] = set()

        for cp_idx, checkpoint in enumerate(scenario.checkpoints):
            # Insert observations up to this checkpoint
            for obs in scenario.observations[:checkpoint.after_observation_index]:
                if obs.id not in observation_ids:
                    db.insert_observation(
                        id=obs.id,
                        student_id=scenario.student_id,
                        session_id=obs.session_id,
                        dimension=obs.dimension,
                        observation_text=obs.observation_text,
                        framing=obs.framing,
                        dimension_score=obs.dimension_score,
                        student_baseline=obs.student_baseline,
                        reasoning_trace=obs.reasoning_trace,
                        piece_context=obs.piece_context,
                        created_at=obs.session_date,
                    )
                    if obs.engaged:
                        db.insert_teaching_approach(
                            student_id=scenario.student_id,
                            observation_id=obs.id,
                            dimension=obs.dimension,
                            framing=obs.framing,
                            approach_summary=f"Feedback on {obs.dimension}: {obs.observation_text[:50]}",
                            engaged=True,
                            created_at=obs.session_date,
                        )
                    observation_ids.add(obs.id)

            # Handle te-04 stale goal
            if scenario.id == "te-04" and cp_idx == 0:
                db.insert_fact(
                    id="stale-goal-1",
                    student_id=scenario.student_id,
                    fact_text="Student wants to learn jazz piano",
                    fact_type="student_reported",
                    valid_at="2025-10-01",
                    confidence="high",
                    source_type="student_reported",
                    created_at="2025-10-01T00:00:00Z",
                )

            # Build prompt
            active_facts = db.query_active_facts(scenario.student_id)
            new_obs = db.get_new_observations_since(scenario.student_id, "1970-01-01T00:00:00Z")
            teaching_approaches = db.get_teaching_approaches_since(scenario.student_id, "1970-01-01T00:00:00Z")
            baselines = db.get_baselines(scenario.student_id)

            user_prompt = build_synthesis_prompt(active_facts, new_obs, teaching_approaches, baselines)

            cache_key = f"{scenario.id}_cp{cp_idx}"

            if live or cache_key not in cache:
                if not live:
                    # Skip if not in cache and not live
                    results.append(SynthesisResult(
                        scenario_id=scenario.id,
                        checkpoint_index=cp_idx,
                        raw_output="[NOT CACHED]",
                    ))
                    continue

                raw_output = _call_groq(SYNTHESIS_SYSTEM, user_prompt)

                # Cache it
                cache[cache_key] = {
                    "key": cache_key,
                    "scenario_id": scenario.id,
                    "checkpoint_index": cp_idx,
                    "prompt": user_prompt,
                    "raw_output": raw_output,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "a") as f:
                    f.write(json.dumps(cache[cache_key], ensure_ascii=False) + "\n")
            else:
                raw_output = cache[cache_key]["raw_output"]

            # Parse JSON
            parsed = _extract_json(raw_output)
            if parsed is None:
                results.append(SynthesisResult(
                    scenario_id=scenario.id,
                    checkpoint_index=cp_idx,
                    json_parsed=False,
                    raw_output=raw_output,
                ))
                continue

            # Analyze results
            produced_facts = parsed.get("new_facts", [])
            invalidated = parsed.get("invalidated_facts", [])

            # Get expected facts for this checkpoint
            expected_fact_ids = set(checkpoint.expected_new_facts)
            expected_facts = [ef for ef in scenario.expected_facts if ef.id in expected_fact_ids]
            expected_invalidation_ids = set(checkpoint.expected_invalidations)

            # Match produced facts against expected
            matched = []
            unmatched_expected = list(expected_fact_ids)
            hallucinated = []
            type_correct_count = 0
            trend_correct_count = 0
            total_matched = 0
            all_grounded = True

            for pf in produced_facts:
                match_id, type_ok, trend_ok = _match_fact(pf, expected_facts, observation_ids)
                if match_id:
                    matched.append(match_id)
                    if match_id in unmatched_expected:
                        unmatched_expected.remove(match_id)
                    total_matched += 1
                    if type_ok:
                        type_correct_count += 1
                    if trend_ok:
                        trend_correct_count += 1
                else:
                    hallucinated.append(pf.get("fact_text", ""))

                if not _is_grounded(pf, observation_ids):
                    all_grounded = False

            # Metrics
            n_expected = len(expected_facts)
            n_produced = len(produced_facts)

            new_fact_recall = len(matched) / n_expected if n_expected > 0 else 1.0
            new_fact_precision = total_matched / n_produced if n_produced > 0 else 1.0
            hallucination_rate = len(hallucinated) / n_produced if n_produced > 0 else 0.0

            fact_type_acc = type_correct_count / total_matched if total_matched > 0 else 1.0
            trend_acc = trend_correct_count / total_matched if total_matched > 0 else 1.0

            # Invalidation metrics
            produced_invalidation_ids = {inv.get("fact_id", "") for inv in invalidated}
            inv_tp = expected_invalidation_ids & produced_invalidation_ids
            inv_recall = len(inv_tp) / len(expected_invalidation_ids) if expected_invalidation_ids else 1.0
            inv_precision = len(inv_tp) / len(produced_invalidation_ids) if produced_invalidation_ids else 1.0

            # Apply invalidations to DB for next checkpoint
            for inv in invalidated:
                fact_id = inv.get("fact_id", "")
                invalid_at = inv.get("invalid_at", "2026-03-01")
                if fact_id:
                    db.invalidate_fact(fact_id, scenario.student_id, invalid_at, invalid_at)

            # Insert new facts into DB for next checkpoint
            for pf in produced_facts:
                import uuid
                db.insert_fact(
                    id=str(uuid.uuid4()),
                    student_id=scenario.student_id,
                    fact_text=pf.get("fact_text", ""),
                    fact_type=pf.get("fact_type", "dimension"),
                    dimension=pf.get("dimension"),
                    piece_context=json.dumps(pf["piece_context"]) if pf.get("piece_context") else None,
                    valid_at=scenario.observations[checkpoint.after_observation_index - 1].session_date[:10] if scenario.observations else "2026-02-01",
                    trend=pf.get("trend"),
                    confidence=pf.get("confidence", "medium"),
                    evidence=json.dumps(pf.get("evidence", [])),
                    created_at=scenario.observations[checkpoint.after_observation_index - 1].session_date if scenario.observations else "2026-02-01",
                )

            results.append(SynthesisResult(
                scenario_id=scenario.id,
                checkpoint_index=cp_idx,
                json_parsed=True,
                raw_output=raw_output,
                new_fact_recall=new_fact_recall,
                new_fact_precision=new_fact_precision,
                hallucination_rate=hallucination_rate,
                invalidation_recall=inv_recall,
                invalidation_precision=inv_precision,
                fact_type_accuracy=fact_type_acc,
                trend_accuracy=trend_acc,
                evidence_grounded=all_grounded,
                produced_facts=produced_facts,
                matched_expected=matched,
                unmatched_expected=unmatched_expected,
                hallucinated=hallucinated,
            ))

    return results


def print_results(results: list[SynthesisResult]) -> None:
    print("\n=== Synthesis Assessment ===\n")

    json_parse_count = 0
    all_recall = []
    all_precision = []
    all_hallucination = []
    all_inv_recall = []
    all_inv_precision = []
    all_type_acc = []
    all_trend_acc = []
    all_grounded = 0

    for r in results:
        if r.raw_output == "[NOT CACHED]":
            print(f"  [SKIP] {r.scenario_id}/cp{r.checkpoint_index} (not cached, run with --live)")
            continue

        status = "PASS" if r.json_parsed and r.new_fact_recall >= 0.8 else "FAIL"
        print(f"  [{status}] {r.scenario_id}/cp{r.checkpoint_index}")
        print(f"    parsed={r.json_parsed} recall={r.new_fact_recall:.2f} prec={r.new_fact_precision:.2f} hall={r.hallucination_rate:.2f}")
        if r.unmatched_expected:
            print(f"    Missing expected: {r.unmatched_expected}")
        if r.hallucinated:
            for h in r.hallucinated[:3]:
                print(f"    Hallucinated: {h[:80]}...")

        if r.json_parsed:
            json_parse_count += 1
        all_recall.append(r.new_fact_recall)
        all_precision.append(r.new_fact_precision)
        all_hallucination.append(r.hallucination_rate)
        all_inv_recall.append(r.invalidation_recall)
        all_inv_precision.append(r.invalidation_precision)
        all_type_acc.append(r.fact_type_accuracy)
        all_trend_acc.append(r.trend_accuracy)
        if r.evidence_grounded:
            all_grounded += 1

    n = len([r for r in results if r.raw_output != "[NOT CACHED]"])
    if n == 0:
        print("\n  No results to aggregate. Run with --live to populate cache.")
        return

    print(f"\n--- Aggregate (n={n}) ---")
    print(f"  JSON parse rate:      {json_parse_count}/{n} ({json_parse_count/n:.2f})")
    print(f"  New fact recall:      {sum(all_recall)/n:.3f}")
    print(f"  New fact precision:   {sum(all_precision)/n:.3f}")
    print(f"  Hallucination rate:   {sum(all_hallucination)/n:.3f}")
    print(f"  Invalidation recall:  {sum(all_inv_recall)/n:.3f}")
    print(f"  Invalidation prec:    {sum(all_inv_precision)/n:.3f}")
    print(f"  Fact type accuracy:   {sum(all_type_acc)/n:.3f}")
    print(f"  Trend accuracy:       {sum(all_trend_acc)/n:.3f}")
    print(f"  Evidence grounded:    {all_grounded}/{n}")


def main() -> None:
    import sys
    live = "--live" in sys.argv
    scenarios_path = DATA_DIR / "scenarios.jsonl"
    scenarios = load_scenarios(scenarios_path)

    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        sid = sys.argv[idx + 1]
        scenarios = [s for s in scenarios if s.id == sid]

    results = run_synthesis_assessment(scenarios, live=live)
    print_results(results)


if __name__ == "__main__":
    main()
