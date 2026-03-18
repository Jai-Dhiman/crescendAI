"""Extract feedback quality descriptors from masterclass teaching moments.

Step 1 of criteria derivation: for each of the 2,136 masterclass moments,
ask an LLM to identify 2-5 qualities that make the intervention effective
or ineffective.

Usage:
    cd apps/evals/
    uv run python -m pipeline.criteria_derivation.extract_qualities
    uv run python -m pipeline.criteria_derivation.extract_qualities --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from paths import MODEL_DATA

import anthropic

MOMENTS_DIR = MODEL_DATA / "raw" / "masterclass" / "teaching_moments"
OUTPUT_DIR = Path(__file__).parent / "data"

EXTRACTION_PROMPT = """Given this piano masterclass teaching moment:

Teacher: {teacher}
Piece: {composer} - {piece}
What the teacher said: {feedback_summary}
Transcript excerpt: {transcript_excerpt}
Feedback type: {feedback_type}
Teacher demonstrated: {demonstrated}
Severity: {severity}

What qualities make this teaching intervention effective or ineffective?
List 2-5 specific qualities, each in 2-5 words.
Focus on qualities expressible in text (not physical demonstration).

Format: one quality per line, no numbering."""


def load_all_moments() -> list[dict]:
    """Load all teaching moments from JSONL files."""
    moments = []
    for jsonl_path in sorted(MOMENTS_DIR.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    moments.append(json.loads(line))
    return moments


def build_prompt(moment: dict) -> str:
    """Build the extraction prompt for a single moment."""
    transcript = moment.get("transcript_text", "")
    if len(transcript) > 500:
        transcript = transcript[:500] + "..."

    return EXTRACTION_PROMPT.format(
        teacher=moment.get("teacher", "Unknown"),
        composer=moment.get("composer", "Unknown"),
        piece=moment.get("piece", "Unknown"),
        feedback_summary=moment.get("feedback_summary", ""),
        transcript_excerpt=transcript,
        feedback_type=moment.get("feedback_type", "Unknown"),
        demonstrated=moment.get("demonstrated", False),
        severity=moment.get("severity", "Unknown"),
    )


def load_existing_results() -> dict[str, dict]:
    """Load already-processed moment IDs for resume support."""
    output_path = OUTPUT_DIR / "qualities_raw.jsonl"
    if not output_path.exists():
        return {}
    existing = {}
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            existing[r["moment_id"]] = r
    return existing


def extract_qualities(
    moments: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    limit: int | None = None,
) -> list[dict]:
    """Extract quality descriptors for each moment via LLM."""
    client = anthropic.Anthropic()
    existing = load_existing_results()
    results = list(existing.values())
    processed_ids = set(existing.keys())

    if limit:
        moments = moments[:limit]

    new_count = 0
    for i, moment in enumerate(moments):
        moment_id = moment.get("moment_id", f"moment_{i}")
        if moment_id in processed_ids:
            continue

        prompt = build_prompt(moment)
        qualities: list[str] = []

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                qualities = [
                    q.strip().lstrip("- ") for q in text.split("\n")
                    if q.strip()
                    and len(q.strip()) > 3
                    and not q.strip().startswith("#")
                    and not q.strip().startswith("*")
                ]
                break
            except anthropic.RateLimitError:
                time.sleep(2 ** (attempt + 1))
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise

        result = {
            "moment_id": moment_id,
            "video_id": moment.get("video_id", ""),
            "teacher": moment.get("teacher", ""),
            "feedback_type": moment.get("feedback_type", ""),
            "severity": moment.get("severity", ""),
            "time_spent_seconds": moment.get("time_spent_seconds", 0),
            "stop_group": moment.get("stop_group", 0),
            "musical_dimension": moment.get("musical_dimension", ""),
            "passage_description": moment.get("passage_description", ""),
            "demonstrated": moment.get("demonstrated", False),
            "qualities": qualities,
        }
        results.append(result)
        new_count += 1

        # Save incrementally every 50 new moments
        if new_count % 50 == 0 or i == len(moments) - 1:
            save_results(results)
            print(f"  [{i+1}/{len(moments)}] {sum(len(r['qualities']) for r in results)} total qualities")

    return results


def save_results(results: list[dict]) -> Path:
    """Save extraction results to JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "qualities_raw.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract feedback quality descriptors")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N moments")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    print("Loading masterclass moments...")
    moments = load_all_moments()
    print(f"  Loaded {len(moments)} moments from {len(list(MOMENTS_DIR.glob('*.jsonl')))} videos")

    existing = load_existing_results()
    if existing:
        print(f"  Resuming: {len(existing)} already processed")

    print("Extracting quality descriptors...")
    results = extract_qualities(moments, model=args.model, limit=args.limit)

    total_qualities = sum(len(r["qualities"]) for r in results)
    avg = total_qualities / len(results) if results else 0
    print(f"\n  Total: {total_qualities} qualities from {len(results)} moments "
          f"({avg:.1f} avg per moment)")

    save_results(results)
    print(f"  Saved to {OUTPUT_DIR / 'qualities_raw.jsonl'}")


if __name__ == "__main__":
    main()
