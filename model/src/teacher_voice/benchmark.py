"""Benchmark runner: test Claude against teaching scenarios at varying context levels.

For each TeachingRecord, constructs three prompt variants (bare, rich, retrieved)
and calls Claude Sonnet, saving all outputs for human rating.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from .records import TeachingRecord, load_records

# Default location for API key (apps/api/.dev.vars)
_DEV_VARS_PATH = Path(__file__).parents[3] / "apps" / "api" / ".dev.vars"


def _load_api_key() -> str | None:
    """Load ANTHROPIC_API_KEY from environment or .dev.vars."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if _DEV_VARS_PATH.exists():
        for line in _DEV_VARS_PATH.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None

# The teacher persona from Slice 06
TEACHER_SYSTEM_PROMPT = """You are a piano teacher who has been listening to your student practice. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

Your role is to give ONE specific observation about what you just heard. Not a report. Not a lesson plan. One thing -- the thing the student most needs to hear right now.

How you speak:
- Specific and grounded: reference the exact musical moment, not generalities
- Natural and warm: you're talking to a student you know, not writing a review
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Brief: 1-3 sentences. A teacher's aside, not a lecture.

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting"""

MODEL = "claude-sonnet-4-20250514"


@dataclass
class BenchmarkResult:
    """Result of running one prompt variant against Claude."""

    record_id: str
    variant: str  # "bare" | "rich" | "retrieved"
    prompt: str
    response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    # Ratings filled in later by rate.py
    accuracy: int | None = None
    actionability: int | None = None
    voice: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _build_bare_prompt(record: TeachingRecord) -> str:
    """Bare context: dimension + generic framing only."""
    return (
        f"The student just played a passage. "
        f"You noticed an issue with their {record.dimension}.\n\n"
        f"Give one observation about the {record.dimension} issue. "
        f"Be specific about what you heard and what to try."
    )


def _build_rich_prompt(record: TeachingRecord) -> str:
    """Rich context: dimension + piece + passage + student level."""
    parts = ["## What I heard\n"]

    if record.piece and record.composer:
        parts.append(f"Piece: {record.piece} by {record.composer}")
    elif record.piece:
        parts.append(f"Piece: {record.piece}")

    if record.passage_description:
        parts.append(f"Passage: {record.passage_description}")

    parts.append(f"Dimension flagged: {record.dimension}")
    parts.append(f"Feedback context: {record.feedback_type}")

    parts.append(f"\n## Who I'm talking to\n")
    parts.append(f"Student level: {record.student_level}")

    parts.append(f"\n## What to say\n")
    parts.append(
        f"Give one observation about the {record.dimension} issue in this moment. "
        f"Be specific about what you heard and what to try."
    )

    return "\n".join(parts)


def _build_retrieved_prompt(
    record: TeachingRecord, quotes: list[dict]
) -> str:
    """Retrieved context: rich + relevant masterclass quotes."""
    rich = _build_rich_prompt(record)

    if not quotes:
        return rich

    quote_text = "\n\n## Reference: What experienced teachers have said\n\n"
    for i, q in enumerate(quotes[:3], 1):
        teacher = q.get("teacher", "A teacher")
        summary = q.get("feedback_summary", "")
        quote_text += f"{i}. {teacher}: \"{summary}\"\n"
    quote_text += (
        "\nUse these as inspiration for tone and vocabulary, "
        "but give YOUR observation about THIS student's playing."
    )

    return rich + quote_text


def _find_relevant_quotes(
    record: TeachingRecord, quote_bank: dict
) -> list[dict]:
    """Simple retrieval: match by dimension, prefer same composer."""
    dim_quotes = quote_bank.get(record.dimension, [])
    if not dim_quotes:
        return []

    # Prefer quotes about the same composer
    same_composer = [
        q for q in dim_quotes
        if record.composer
        and record.composer.lower() in q.get("piece", "").lower()
    ]
    if len(same_composer) >= 3:
        return same_composer[:3]

    # Fall back to any quotes in this dimension
    return dim_quotes[:3]


def load_quote_bank(path: Path) -> dict:
    """Load the quote bank JSON (keyed by dimension)."""
    with open(path) as f:
        return json.load(f)


def run_benchmark(
    records: list[TeachingRecord],
    quote_bank: dict,
    output_path: Path,
    variants: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run benchmark for all records across specified variants.

    Args:
        records: TeachingRecords to test.
        quote_bank: Quote bank dict keyed by dimension.
        output_path: Path to save results JSONL.
        variants: Which variants to run. Defaults to all three.

    Returns:
        List of BenchmarkResults.
    """
    if variants is None:
        variants = ["bare", "rich", "retrieved"]

    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Set it in the environment "
            "or in apps/api/.dev.vars"
        )
    client = anthropic.Anthropic(api_key=api_key)
    results: list[BenchmarkResult] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to support resuming
    existing_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                existing_keys.add((d["record_id"], d["variant"]))
        results = [
            BenchmarkResult.from_dict(json.loads(line))
            for line in open(output_path)
        ]

    total = len(records) * len(variants)
    completed = len(existing_keys)
    print(f"Running benchmark: {total} total calls, {completed} already done")

    with open(output_path, "a") as f:
        for record in records:
            for variant in variants:
                if (record.id, variant) in existing_keys:
                    continue

                if variant == "bare":
                    prompt = _build_bare_prompt(record)
                elif variant == "rich":
                    prompt = _build_rich_prompt(record)
                elif variant == "retrieved":
                    quotes = _find_relevant_quotes(record, quote_bank)
                    prompt = _build_retrieved_prompt(record, quotes)
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                start = time.monotonic()
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=300,
                    system=TEACHER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

                text = response.content[0].text

                result = BenchmarkResult(
                    record_id=record.id,
                    variant=variant,
                    prompt=prompt,
                    response=text,
                    latency_ms=elapsed_ms,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                results.append(result)

                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                f.flush()

                completed += 1
                print(
                    f"  [{completed}/{total}] {record.id[:8]}... "
                    f"({variant}) {elapsed_ms}ms - "
                    f"{text[:60]}..."
                )

    print(f"\nDone. {len(results)} results saved to {output_path}")
    return results


if __name__ == "__main__":
    import random
    import sys

    data_dir = Path(__file__).parents[2] / "data"
    records_path = data_dir / "teacher_voice_eval" / "masterclass_records.jsonl"
    quote_bank_path = data_dir / "composite_labels" / "quote_bank.json"
    output_path = data_dir / "teacher_voice_eval" / "benchmark_results.jsonl"

    if not records_path.exists():
        print(f"Run converters.py first to generate {records_path}")
        sys.exit(1)

    records = load_records(records_path)
    quote_bank = load_quote_bank(quote_bank_path)

    # For prototype: sample 50 records for benchmark
    random.seed(42)
    benchmark_records = random.sample(records, min(50, len(records)))
    print(f"Selected {len(benchmark_records)} records for benchmark")

    run_benchmark(benchmark_records, quote_bank, output_path)
