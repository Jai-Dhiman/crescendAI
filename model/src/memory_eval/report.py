"""Benchmark comparison report.

Maps CrescendAI metrics to LongMemEval's 5 dimensions and outputs
a comparison table with caveats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data" / "memory_eval"

# Industry benchmark scores for comparison
INDUSTRY_BENCHMARKS = {
    "LongMemEval": {
        "source": "ICLR 2025",
        "top_system": "Emergence AI",
        "extraction": 0.86,
        "multi_session": 0.86,
        "temporal": 0.86,
        "knowledge_update": 0.86,
        "abstention": 0.86,
    },
    "Zep": {
        "source": "Deep Memory Retrieval",
        "extraction": 0.948,
        "multi_session": None,
        "temporal": None,
        "knowledge_update": None,
        "abstention": None,
    },
    "Mem0": {
        "source": "LoCoMo benchmark",
        "extraction": 0.669,
        "multi_session": 0.669,
        "temporal": None,
        "knowledge_update": None,
        "abstention": None,
    },
    "MemGPT": {
        "source": "Deep Memory Retrieval",
        "extraction": 0.934,
        "multi_session": None,
        "temporal": None,
        "knowledge_update": None,
        "abstention": None,
    },
}

LONGMEMEVAL_DIMENSIONS = [
    "extraction",
    "multi_session",
    "temporal",
    "knowledge_update",
    "abstention",
]

DIMENSION_DESCRIPTIONS = {
    "extraction": "Information extraction from observations",
    "multi_session": "Cross-session pattern reasoning",
    "temporal": "Fact invalidation timing",
    "knowledge_update": "Fact supersession accuracy",
    "abstention": "Abstaining when evidence is insufficient",
}

CRESCENDAI_METRIC_MAP = {
    "extraction": "New fact recall (synthesis)",
    "multi_session": "Multi-session temporal accuracy",
    "temporal": "Temporal assertion accuracy",
    "knowledge_update": "Invalidation recall + precision",
    "abstention": "1 - hallucination + abstention accuracy",
}


@dataclass
class CrescendAIScores:
    extraction: float | None = None
    multi_session: float | None = None
    temporal: float | None = None
    knowledge_update: float | None = None
    abstention: float | None = None
    n_scenarios: int = 30

    # Detailed metrics
    retrieval_f1: float | None = None
    synthesis_recall: float | None = None
    synthesis_precision: float | None = None
    hallucination_rate: float | None = None
    invalidation_recall: float | None = None
    invalidation_precision: float | None = None
    json_parse_rate: float | None = None
    fact_type_accuracy: float | None = None
    trend_accuracy: float | None = None
    downstream_memory_lift: float | None = None
    downstream_win_rate: float | None = None


def load_retrieval_results() -> dict:
    """Load retrieval assessment results if available."""
    # Import inline to avoid circular deps at module load
    from .eval_retrieval import run_retrieval_assessment
    from .scenarios import load_scenarios

    scenarios_path = DATA_DIR / "scenarios.jsonl"
    if not scenarios_path.exists():
        return {}

    scenarios = load_scenarios(scenarios_path)
    results = run_retrieval_assessment(scenarios)

    n = len(results)
    if n == 0:
        return {}

    return {
        "mean_precision": sum(r.precision for r in results) / n,
        "mean_recall": sum(r.recall for r in results) / n,
        "mean_f1": sum(r.f1 for r in results) / n,
        "total_duplicates": sum(r.duplicate_count for r in results),
        "limit_violations": sum(0 if r.limit_respected else 1 for r in results),
    }


def load_synthesis_results() -> dict:
    """Load synthesis assessment results from cache."""
    cache_path = DATA_DIR / "synthesis_cache.jsonl"
    if not cache_path.exists():
        return {}

    from .eval_synthesis import run_synthesis_assessment
    from .scenarios import load_scenarios

    scenarios = load_scenarios(DATA_DIR / "scenarios.jsonl")
    results = run_synthesis_assessment(scenarios, live=False)

    assessed = [r for r in results if r.raw_output != "[NOT CACHED]"]
    if not assessed:
        return {}

    n = len(assessed)
    return {
        "json_parse_rate": sum(1 for r in assessed if r.json_parsed) / n,
        "new_fact_recall": sum(r.new_fact_recall for r in assessed) / n,
        "new_fact_precision": sum(r.new_fact_precision for r in assessed) / n,
        "hallucination_rate": sum(r.hallucination_rate for r in assessed) / n,
        "invalidation_recall": sum(r.invalidation_recall for r in assessed) / n,
        "invalidation_precision": sum(r.invalidation_precision for r in assessed) / n,
        "fact_type_accuracy": sum(r.fact_type_accuracy for r in assessed) / n,
        "trend_accuracy": sum(r.trend_accuracy for r in assessed) / n,
        "evidence_grounded": sum(1 for r in assessed if r.evidence_grounded) / n,
        "n": n,
    }


def load_temporal_results() -> dict:
    """Load temporal assessment results from cache."""
    cache_path = DATA_DIR / "synthesis_cache.jsonl"
    if not cache_path.exists():
        return {}

    from .eval_temporal import run_temporal_assessment, CATEGORIES
    from .scenarios import load_scenarios

    scenarios = load_scenarios(DATA_DIR / "scenarios.jsonl")
    results = run_temporal_assessment(scenarios, live=False)

    if not results:
        return {}

    overall = sum(1 for r in results if r.correct) / len(results)
    per_category = {}
    for cat in CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            per_category[cat] = sum(1 for r in cat_results if r.correct) / len(cat_results)

    return {
        "overall": overall,
        "per_category": per_category,
        "n": len(results),
    }


def load_downstream_results() -> dict:
    """Load downstream judge results."""
    judge_path = DATA_DIR / "judge_results.jsonl"
    if not judge_path.exists():
        return {}

    entries = []
    with open(judge_path) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return {}

    judged = [e for e in entries if e.get("judge_raw")]
    if not judged:
        return {}

    axes = ["continuity", "specificity", "non_repetition", "approach_fit", "accuracy"]
    win_count = 0
    all_lift = []

    for e in judged:
        j = e["judge_raw"]
        flipped = e.get("flipped", False)
        a_scores = j.get("response_a", {})
        b_scores = j.get("response_b", {})

        if flipped:
            with_scores, no_scores = a_scores, b_scores
        else:
            no_scores, with_scores = a_scores, b_scores

        with_mean = sum(with_scores.get(a, 3) for a in axes) / len(axes)
        no_mean = sum(no_scores.get(a, 3) for a in axes) / len(axes)
        lift = with_mean - no_mean
        all_lift.append(lift)

        with_total = sum(with_scores.get(a, 0) for a in axes)
        no_total = sum(no_scores.get(a, 0) for a in axes)
        if with_total > no_total:
            win_count += 1

    n = len(judged)
    return {
        "memory_lift": sum(all_lift) / n,
        "win_rate": win_count / n,
        "n": n,
    }


def build_scores() -> CrescendAIScores:
    """Aggregate all available results into CrescendAI benchmark scores."""
    scores = CrescendAIScores()

    retrieval = load_retrieval_results()
    synthesis = load_synthesis_results()
    temporal = load_temporal_results()
    downstream = load_downstream_results()

    if retrieval:
        scores.retrieval_f1 = retrieval["mean_f1"]

    if synthesis:
        scores.synthesis_recall = synthesis["new_fact_recall"]
        scores.synthesis_precision = synthesis["new_fact_precision"]
        scores.hallucination_rate = synthesis["hallucination_rate"]
        scores.invalidation_recall = synthesis["invalidation_recall"]
        scores.invalidation_precision = synthesis["invalidation_precision"]
        scores.json_parse_rate = synthesis["json_parse_rate"]
        scores.fact_type_accuracy = synthesis["fact_type_accuracy"]
        scores.trend_accuracy = synthesis["trend_accuracy"]

        # Map to LongMemEval dimensions
        scores.extraction = synthesis["new_fact_recall"]

    if temporal:
        per_cat = temporal.get("per_category", {})
        scores.temporal = per_cat.get("temporal")
        scores.multi_session = per_cat.get("multi_session")
        scores.knowledge_update = per_cat.get("knowledge_update")
        scores.abstention = per_cat.get("abstention")

    if downstream:
        scores.downstream_memory_lift = downstream["memory_lift"]
        scores.downstream_win_rate = downstream["win_rate"]

    return scores


def print_report(scores: CrescendAIScores) -> None:
    """Print the benchmark comparison report."""
    print("\n" + "=" * 80)
    print("  CRESCENDAI MEMORY SYSTEM - BENCHMARK COMPARISON REPORT")
    print("=" * 80)

    # Comparison table
    print("\n--- LongMemEval Dimension Mapping ---\n")
    print(f"  {'Dimension':<20} {'CrescendAI':<12} {'Top (LME)':<12} {'Zep':<12} {'Mem0':<12} {'MemGPT':<12}")
    print(f"  {'-'*20} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for dim in LONGMEMEVAL_DIMENSIONS:
        crescend_val = getattr(scores, dim)
        crescend_str = f"{crescend_val:.2f}" if crescend_val is not None else "--"
        lme_val = INDUSTRY_BENCHMARKS["LongMemEval"].get(dim)
        lme_str = f"{lme_val:.2f}" if lme_val is not None else "--"
        zep_val = INDUSTRY_BENCHMARKS["Zep"].get(dim)
        zep_str = f"{zep_val:.2f}" if zep_val is not None else "--"
        mem0_val = INDUSTRY_BENCHMARKS["Mem0"].get(dim)
        mem0_str = f"{mem0_val:.2f}" if mem0_val is not None else "--"
        memgpt_val = INDUSTRY_BENCHMARKS["MemGPT"].get(dim)
        memgpt_str = f"{memgpt_val:.2f}" if memgpt_val is not None else "--"

        print(f"  {dim:<20} {crescend_str:<12} {lme_str:<12} {zep_str:<12} {mem0_str:<12} {memgpt_str:<12}")

    # CrescendAI metric source mapping
    print(f"\n--- CrescendAI Metric Sources ---\n")
    for dim in LONGMEMEVAL_DIMENSIONS:
        print(f"  {dim}: {CRESCENDAI_METRIC_MAP[dim]}")

    # Detailed metrics
    print(f"\n--- Detailed CrescendAI Metrics ---\n")

    if scores.retrieval_f1 is not None:
        print(f"  Retrieval layer:")
        print(f"    F1: {scores.retrieval_f1:.3f}")

    if scores.synthesis_recall is not None:
        print(f"  Synthesis layer:")
        print(f"    New fact recall:      {scores.synthesis_recall:.3f}")
        print(f"    New fact precision:   {scores.synthesis_precision:.3f}")
        print(f"    Hallucination rate:   {scores.hallucination_rate:.3f}")
        print(f"    Invalidation recall:  {scores.invalidation_recall:.3f}")
        print(f"    Invalidation prec:    {scores.invalidation_precision:.3f}")
        print(f"    JSON parse rate:      {scores.json_parse_rate:.3f}")
        print(f"    Fact type accuracy:   {scores.fact_type_accuracy:.3f}")
        print(f"    Trend accuracy:       {scores.trend_accuracy:.3f}")

    if scores.temporal is not None:
        print(f"  Temporal layer:")
        print(f"    Temporal accuracy:    {scores.temporal:.3f}")
    if scores.multi_session is not None:
        print(f"    Multi-session:        {scores.multi_session:.3f}")
    if scores.knowledge_update is not None:
        print(f"    Knowledge update:     {scores.knowledge_update:.3f}")
    if scores.abstention is not None:
        print(f"    Abstention:           {scores.abstention:.3f}")

    if scores.downstream_memory_lift is not None:
        print(f"  Downstream impact:")
        print(f"    Memory lift (judge):  {scores.downstream_memory_lift:+.3f}")
        print(f"    Win rate:             {scores.downstream_win_rate:.1%}")

    # Caveats
    print(f"\n--- Caveats ---\n")
    print("  1. CrescendAI domain is narrower (6 dims, known ontology) -- easier than open-domain")
    print("  2. Scenario count is smaller (30 vs 500+ in LongMemEval)")
    print("  3. Different LLM for synthesis (Llama 3.3 70B vs GPT-4/Claude in competitors)")
    print("  4. Scores test equivalent cognitive abilities, not identical tasks")
    print("  5. Industry scores from published papers; CrescendAI scores from internal eval")
    print()


def main() -> None:
    scores = build_scores()
    print_report(scores)


if __name__ == "__main__":
    main()
