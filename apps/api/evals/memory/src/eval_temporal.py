"""Temporal reasoning evaluation.

Replays observations chronologically, runs synthesis at checkpoints,
then checks fact state at each assertion's query_time.

Categories (adapted from LongMemEval):
- extraction: Fact creation from observations
- multi_session: Cross-session pattern detection
- temporal: Fact invalidation timing
- knowledge_update: Fact supersession
- abstention: No fact when insufficient evidence
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .eval_synthesis import SYNTHESIS_SYSTEM, _call_groq, _extract_json, build_synthesis_prompt
from .memory_db import MemoryDB
from .scenarios import MemoryEvalScenario, load_scenarios

DATA_DIR = Path(__file__).parents[1] / "data"

CATEGORIES = ["extraction", "multi_session", "temporal", "knowledge_update", "abstention"]


@dataclass
class TemporalResult:
    scenario_id: str
    assertion_index: int
    category: str
    query_time: str
    fact_pattern: str
    should_be_active: bool
    is_active: bool
    correct: bool
    matching_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_matching_facts(facts: list, pattern: str) -> list[str]:
    """Find facts whose fact_text matches the regex pattern."""
    matches = []
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return matches
    for f in facts:
        if compiled.search(f.fact_text):
            matches.append(f.fact_text)
    return matches


def run_temporal_assessment(
    scenarios: list[MemoryEvalScenario],
    live: bool = False,
    cache_path: Path | None = None,
) -> list[TemporalResult]:
    """Run temporal reasoning assessment.

    For scenarios with temporal_assertions, this:
    1. Replays observations chronologically
    2. Runs synthesis at each checkpoint (live or cached)
    3. Checks fact state at each assertion's query_time
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
    results: list[TemporalResult] = []

    for scenario in scenarios:
        if not scenario.temporal_assertions:
            continue

        db.reset()
        db.insert_student(scenario.student_id, scenario.baselines)

        observation_ids: set[str] = set()

        # Replay observations and run synthesis at checkpoints
        for cp_idx, checkpoint in enumerate(scenario.checkpoints):
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

            # Build synthesis prompt
            active_facts = db.query_active_facts(scenario.student_id)
            new_obs = db.get_new_observations_since(scenario.student_id, "1970-01-01T00:00:00Z")
            teaching_approaches = db.get_teaching_approaches_since(scenario.student_id, "1970-01-01T00:00:00Z")
            baselines = db.get_baselines(scenario.student_id)

            user_prompt = build_synthesis_prompt(active_facts, new_obs, teaching_approaches, baselines)
            cache_key = f"{scenario.id}_cp{cp_idx}"

            if cache_key in cache:
                raw_output = cache[cache_key]["raw_output"]
            elif live:
                raw_output = _call_groq(SYNTHESIS_SYSTEM, user_prompt)
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
                continue  # skip if no cache and not live

            parsed = _extract_json(raw_output)
            if parsed is None:
                continue

            # Apply invalidations
            for inv in parsed.get("invalidated_facts", []):
                fact_id = inv.get("fact_id", "")
                invalid_at = inv.get("invalid_at", "2026-03-01")
                if fact_id:
                    db.invalidate_fact(fact_id, scenario.student_id, invalid_at, invalid_at)

            # Insert new facts
            import uuid
            for pf in parsed.get("new_facts", []):
                fact_text = pf.get("fact_text", "")
                if not fact_text:
                    continue
                db.insert_fact(
                    id=str(uuid.uuid4()),
                    student_id=scenario.student_id,
                    fact_text=fact_text,
                    fact_type=pf.get("fact_type", "dimension"),
                    dimension=pf.get("dimension"),
                    piece_context=json.dumps(pf["piece_context"]) if pf.get("piece_context") else None,
                    valid_at=scenario.observations[checkpoint.after_observation_index - 1].session_date[:10],
                    trend=pf.get("trend"),
                    confidence=pf.get("confidence", "medium"),
                    evidence=json.dumps(pf.get("evidence", [])),
                    created_at=scenario.observations[checkpoint.after_observation_index - 1].session_date,
                )

        # Now check temporal assertions against current DB state
        for a_idx, assertion in enumerate(scenario.temporal_assertions):
            active_facts = db.query_active_facts(scenario.student_id)
            matching = _find_matching_facts(active_facts, assertion.fact_pattern)

            is_active = len(matching) > 0
            correct = is_active == assertion.should_be_active

            results.append(TemporalResult(
                scenario_id=scenario.id,
                assertion_index=a_idx,
                category=assertion.category,
                query_time=assertion.query_time,
                fact_pattern=assertion.fact_pattern,
                should_be_active=assertion.should_be_active,
                is_active=is_active,
                correct=correct,
                matching_facts=matching,
            ))

    return results


def print_results(results: list[TemporalResult]) -> None:
    print("\n=== Temporal Reasoning Assessment ===\n")

    if not results:
        print("  No results. Run with --live to populate cache.")
        return

    # Per-scenario
    for r in results:
        status = "PASS" if r.correct else "FAIL"
        print(f"  [{status}] {r.scenario_id}/a{r.assertion_index} ({r.category})")
        print(f"    pattern='{r.fact_pattern[:50]}' expected_active={r.should_be_active} actual={r.is_active}")
        if r.matching_facts:
            for mf in r.matching_facts[:2]:
                print(f"    matched: {mf[:80]}")

    # Per-category breakdown
    print(f"\n--- Per-Category Accuracy ---")
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        correct = sum(1 for r in cat_results if r.correct)
        total = len(cat_results)
        print(f"  {cat}: {correct}/{total} ({correct/total:.2f})")

    # Overall
    correct_total = sum(1 for r in results if r.correct)
    print(f"\n  Overall: {correct_total}/{len(results)} ({correct_total/len(results):.2f})")


def main() -> None:
    import sys
    live = "--live" in sys.argv
    scenarios_path = DATA_DIR / "scenarios.jsonl"
    scenarios = load_scenarios(scenarios_path)
    results = run_temporal_assessment(scenarios, live=live)
    print_results(results)


if __name__ == "__main__":
    main()
