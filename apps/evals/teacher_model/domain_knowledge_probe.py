"""Domain Knowledge Probe for CPT Gate Evaluation (Gate 4).

Tests whether a model has absorbed piano pedagogy knowledge from CPT training data.
Gate criterion: model must score >= 60% overall to pass.

Usage:
    uv run python -m teacher_model.domain_knowledge_probe --provider workers-ai
    uv run python -m teacher_model.domain_knowledge_probe --provider anthropic
    uv run python -m teacher_model.domain_knowledge_probe --provider workers-ai --topic technique
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make parent package importable when run as __main__
_EVALS_DIR = Path(__file__).resolve().parents[1]
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))

from teaching_knowledge.llm_client import LLMClient

QUESTIONS_PATH = Path(__file__).parent / "data" / "domain_probe.json"

GATE_THRESHOLD = 0.60

SYSTEM_PROMPT = (
    "You are taking a piano pedagogy knowledge quiz. "
    "For each question, respond with ONLY the letter of the correct answer "
    "(A, B, C, or D). Do not explain your reasoning."
)


@dataclass
class ProbeResult:
    question_id: str
    topic: str
    correct_answer: str
    model_answer: str | None
    is_correct: bool


def load_questions(topic_filter: str | None = None) -> list[dict]:
    """Load questions from domain_probe.json, optionally filtered by topic."""
    with QUESTIONS_PATH.open() as f:
        questions = json.load(f)

    if topic_filter:
        questions = [q for q in questions if q["topic"] == topic_filter]

    return questions


def _format_question(q: dict) -> str:
    """Format a question dict into a prompt string."""
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in q["choices"].items()
    )
    return f"{q['question']}\n\n{choices}"


def extract_answer(response: str) -> str | None:
    """Extract a single letter answer (A/B/C/D) from the model response.

    Handles common response patterns:
    - "B"
    - "B."
    - "The answer is B"
    - "B) Controlled arm weight..."
    - Leading/trailing whitespace
    """
    if not response:
        return None

    text = response.strip()

    # Direct single-letter response (most common with strict system prompt)
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()

    # Letter followed by punctuation: "B." or "B)"
    if len(text) == 2 and text[0].upper() in "ABCD" and text[1] in ".):":
        return text[0].upper()

    # First line is just a letter
    first_line = text.splitlines()[0].strip()
    if first_line.upper() in ("A", "B", "C", "D"):
        return first_line.upper()

    # "The answer is B" or "Answer: B"
    match = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def run_probe(
    client: LLMClient,
    questions: list[dict] | None = None,
) -> list[ProbeResult]:
    """Run all questions through the model and return results.

    Args:
        client: Initialized LLMClient instance.
        questions: Questions to run. If None, loads all questions.

    Returns:
        List of ProbeResult for each question.
    """
    if questions is None:
        questions = load_questions()

    results: list[ProbeResult] = []

    for i, q in enumerate(questions, 1):
        prompt = _format_question(q)
        print(f"  [{i:>3}/{len(questions)}] {q['id']} ({q['topic']})", end="", flush=True)

        response = client.complete(
            user=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=16,
        )

        answer = extract_answer(response)
        correct = q["correct"]
        is_correct = answer == correct

        result = ProbeResult(
            question_id=q["id"],
            topic=q["topic"],
            correct_answer=correct,
            model_answer=answer,
            is_correct=is_correct,
        )
        results.append(result)

        mark = "+" if is_correct else f"-  (got {answer!r}, expected {correct!r})"
        print(f"  {mark}")

    return results


def summarize_results(results: list[ProbeResult]) -> dict:
    """Compute overall accuracy, by-topic breakdown, and gate pass/fail.

    Returns:
        dict with keys:
          - total: int
          - correct: int
          - accuracy: float (0-1)
          - gate_pass: bool
          - gate_threshold: float
          - by_topic: dict[str, dict] with total/correct/accuracy per topic
          - failures: list[dict] for incorrect answers
    """
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0.0

    topics: dict[str, dict] = {}
    for r in results:
        t = r.topic
        if t not in topics:
            topics[t] = {"total": 0, "correct": 0}
        topics[t]["total"] += 1
        if r.is_correct:
            topics[t]["correct"] += 1

    by_topic = {
        t: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
        }
        for t, v in sorted(topics.items())
    }

    failures = [
        {
            "id": r.question_id,
            "topic": r.topic,
            "expected": r.correct_answer,
            "got": r.model_answer,
        }
        for r in results
        if not r.is_correct
    ]

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "gate_pass": accuracy >= GATE_THRESHOLD,
        "gate_threshold": GATE_THRESHOLD,
        "by_topic": by_topic,
        "failures": failures,
    }


def _print_summary(summary: dict, model_name: str) -> None:
    """Print a human-readable summary to stdout."""
    acc_pct = summary["accuracy"] * 100
    gate = "PASS" if summary["gate_pass"] else "FAIL"
    threshold_pct = summary["gate_threshold"] * 100

    print()
    print("=" * 60)
    print(f"Domain Knowledge Probe Results")
    print(f"Model : {model_name}")
    print(f"Score : {summary['correct']}/{summary['total']}  ({acc_pct:.1f}%)")
    print(f"Gate 4: {gate}  (threshold {threshold_pct:.0f}%)")
    print()
    print("By topic:")
    for topic, stats in summary["by_topic"].items():
        pct = stats["accuracy"] * 100
        bar_filled = int(pct / 5)
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"  {topic:<18} {stats['correct']:>3}/{stats['total']:<3}  [{bar}]  {pct:.0f}%")

    if summary["failures"]:
        print()
        print(f"Incorrect answers ({len(summary['failures'])}):")
        for f in summary["failures"]:
            print(f"  {f['id']} [{f['topic']}]  expected={f['expected']}  got={f['got']!r}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run domain knowledge probe for CPT gate evaluation."
    )
    parser.add_argument(
        "--provider",
        choices=["workers-ai", "anthropic"],
        default="workers-ai",
        help="LLM provider to use (default: workers-ai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (e.g. '@cf/openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--tier",
        choices=["cheap", "quality", "default"],
        default="default",
        help="Model tier when no explicit --model given (default: default)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        choices=["technique", "repertoire", "pedagogy", "psychology", "concepts"],
        help="Run only questions from this topic",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON summary to this file path",
    )
    args = parser.parse_args()

    client = LLMClient(provider=args.provider, model=args.model, tier=args.tier)
    print(f"Provider : {client.provider}")
    print(f"Model    : {client.model}")

    questions = load_questions(topic_filter=args.topic)
    print(f"Questions: {len(questions)}")
    if args.topic:
        print(f"Topic    : {args.topic}")
    print()

    results = run_probe(client, questions)
    summary = summarize_results(results)

    _print_summary(summary, model_name=client.model)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {output_path}")

    sys.exit(0 if summary["gate_pass"] else 1)


if __name__ == "__main__":
    main()
