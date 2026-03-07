"""Retrieval evaluation: deterministic precision/recall/F1 for memory queries.

No LLM calls. For each scenario:
1. Populate MemoryDB with observations + pre-computed gold facts
2. Run each RetrievalQuery
3. Compare returned vs expected fact IDs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .memory_db import MemoryDB
from .scenarios import MemoryEvalScenario, load_scenarios

DATA_DIR = Path(__file__).parents[1] / "data"


@dataclass
class RetrievalResult:
    scenario_id: str
    query_id: str
    query_type: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    returned_ids: list[str] = field(default_factory=list)
    expected_ids: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    limit_respected: bool = True


def _populate_db(db: MemoryDB, scenario: MemoryEvalScenario) -> dict[str, str]:
    """Populate DB with scenario data. Returns mapping of expected_fact_id -> db_fact_id."""
    db.reset()
    db.insert_student(scenario.student_id, scenario.baselines)

    # Insert observations
    for obs in scenario.observations:
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

    # Insert expected facts as gold-standard synthesized facts
    fact_id_map = {}
    for ef in scenario.expected_facts:
        if not ef.id:
            continue
        fact_text = ef.fact_text_pattern.replace(r"(?i)", "").replace("(", "").replace(")", "").replace("|", "/").replace(".*", " ").replace(r"\s", " ")
        db.insert_fact(
            id=ef.id,
            student_id=scenario.student_id,
            fact_text=f"[Gold] {fact_text}",
            fact_type=ef.fact_type,
            dimension=ef.dimension,
            valid_at=ef.valid_at or "2026-02-01",
            trend=ef.trend,
            confidence=ef.confidence,
            created_at=ef.valid_at or "2026-02-01",
        )
        fact_id_map[ef.id] = ef.id

    # Handle te-04 stale goal scenario
    if scenario.id == "te-04":
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

    return fact_id_map


def run_retrieval_assessment(scenarios: list[MemoryEvalScenario]) -> list[RetrievalResult]:
    """Run retrieval assessment on all scenarios."""
    db = MemoryDB()
    results: list[RetrievalResult] = []

    for scenario in scenarios:
        _populate_db(db, scenario)

        for query in scenario.retrieval_queries:
            if query.query_type == "active_facts":
                facts = db.query_active_facts(scenario.student_id)
                returned_ids = [f.id for f in facts]
                limit = 15
            elif query.query_type == "recent_observations":
                obs = db.query_recent_observations_with_engagement(scenario.student_id)
                returned_ids = [f"{o.dimension}:{o.created_at}" for o in obs]
                limit = 5
            elif query.query_type == "piece_facts":
                if query.piece_title:
                    ctx = db.build_memory_context(scenario.student_id, query.piece_title)
                    returned_ids = [f.id for f in ctx["piece_facts"]]
                else:
                    returned_ids = []
                limit = None
            else:
                continue

            expected_ids = set(query.expected_fact_ids)
            absent_ids = set(query.expected_absent_ids)
            returned_set = set(returned_ids)

            duplicate_count = len(returned_ids) - len(returned_set)
            limit_ok = limit is None or len(returned_ids) <= limit

            if not expected_ids and not returned_set:
                precision = recall = f1 = 1.0
                false_positives_list: list[str] = []
                false_negatives_list: list[str] = []
            elif not expected_ids:
                false_positives_list = [fid for fid in returned_set if fid in absent_ids]
                false_negatives_list = []
                precision = 1.0 if not false_positives_list else 0.0
                recall = 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            elif not returned_set:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                false_positives_list = []
                false_negatives_list = list(expected_ids)
            else:
                true_positives = expected_ids & returned_set
                fp_set = returned_set - expected_ids
                false_positives_list = [fid for fid in fp_set if fid in absent_ids]
                false_negatives_list = list(expected_ids - returned_set)

                tp = len(true_positives)
                fp = len(false_positives_list)
                fn = len(false_negatives_list)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append(RetrievalResult(
                scenario_id=scenario.id,
                query_id=query.id,
                query_type=query.query_type,
                precision=precision,
                recall=recall,
                f1=f1,
                returned_ids=returned_ids,
                expected_ids=list(expected_ids),
                false_positives=false_positives_list,
                false_negatives=false_negatives_list,
                duplicate_count=duplicate_count,
                limit_respected=limit_ok,
            ))

    return results


def print_results(results: list[RetrievalResult]) -> None:
    """Print per-scenario and aggregate metrics."""
    print("\n=== Retrieval Assessment ===\n")

    all_precision = []
    all_recall = []
    all_f1 = []
    total_duplicates = 0
    limit_violations = 0

    for r in results:
        status = "PASS" if r.f1 >= 0.95 and r.duplicate_count == 0 and r.limit_respected else "FAIL"
        print(f"  [{status}] {r.scenario_id}/{r.query_id} ({r.query_type})")
        print(f"    P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f} dupes={r.duplicate_count} limit_ok={r.limit_respected}")
        if r.false_negatives:
            print(f"    Missing: {r.false_negatives}")
        if r.false_positives:
            print(f"    Unexpected: {r.false_positives}")

        all_precision.append(r.precision)
        all_recall.append(r.recall)
        all_f1.append(r.f1)
        total_duplicates += r.duplicate_count
        if not r.limit_respected:
            limit_violations += 1

    n = len(results)
    print(f"\n--- Aggregate (n={n}) ---")
    print(f"  Mean Precision: {sum(all_precision)/n:.3f}")
    print(f"  Mean Recall:    {sum(all_recall)/n:.3f}")
    print(f"  Mean F1:        {sum(all_f1)/n:.3f}")
    print(f"  Total duplicates: {total_duplicates}")
    print(f"  Limit violations: {limit_violations}")


def main() -> None:
    scenarios_path = DATA_DIR / "scenarios.jsonl"
    scenarios = load_scenarios(scenarios_path)
    results = run_retrieval_assessment(scenarios)
    print_results(results)


if __name__ == "__main__":
    main()
