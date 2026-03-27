"""Derive eval rubrics from the Teaching Playbook.

Reads the playbook, identifies quality dimensions that distinguish great from
mediocre teaching, and generates judge prompt templates with real examples.

Usage:
  uv run python -m apps.evals.teaching_knowledge.derive_rubrics \
    --playbook data/playbook.yaml \
    --output-dir ../shared/prompts/
"""
import argparse
import json
from pathlib import Path

import anthropic

RUBRIC_PROMPT = """You are designing an evaluation rubric for an AI piano teacher.

Here is the Teaching Playbook (derived from real masterclass analysis and pedagogy research):
{playbook}

TASK: Design evaluation dimensions and a scoring rubric for judging whether the AI
teacher's synthesis output is good. The dimensions and scales should emerge from the
playbook -- what does the playbook say distinguishes great teaching from mediocre?

For each dimension:
1. Name it (something specific, not generic like "quality")
2. Define what each score level looks like (with concrete examples from the playbook)
3. Include a "zero" score that flags critical failures (would damage student trust)
4. Include calibration examples: one good and one bad synthesis example per dimension

Also produce:
- A system prompt for the LLM judge that includes the rubric
- Instructions for how to use the playbook's piece-style rules in scoring
- The format for judge output (criterion name, score, evidence)

Return a JSON object with:
{{
  "dimensions": [
    {{
      "name": "string",
      "description": "string",
      "scale": {{"0": "string", "1": "string", "2": "string", "3": "string"}},
      "good_example": {{"context": "string", "synthesis": "string", "score": 3, "why": "string"}},
      "bad_example": {{"context": "string", "synthesis": "string", "score": 0, "why": "string"}}
    }}
  ],
  "judge_system_prompt": "string",
  "scoring_format": "string"
}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("apps/evals/shared/prompts"))
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20250514")
    args = parser.parse_args()

    playbook_text = args.playbook.read_text()
    client = anthropic.Anthropic()

    print("Deriving eval rubrics from playbook...")
    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(playbook=playbook_text[:30000])}],
    )

    try:
        rubrics = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        print("ERROR: Could not parse rubrics as JSON")
        print(response.content[0].text[:2000])
        return

    # Write judge system prompt
    judge_prompt_path = args.output_dir / "synthesis_quality_judge_v2.txt"
    judge_prompt_path.write_text(rubrics["judge_system_prompt"])
    print(f"Judge prompt saved to {judge_prompt_path}")

    # Write full rubric definition (for reference and calibration)
    rubric_path = args.output_dir / "rubric_definition.json"
    rubric_path.write_text(json.dumps(rubrics, indent=2))
    print(f"Full rubric definition saved to {rubric_path}")

    # Summary
    dims = rubrics.get("dimensions", [])
    print(f"\nDerived {len(dims)} evaluation dimensions:")
    for d in dims:
        print(f"  - {d['name']}: {d['description'][:80]}")


if __name__ == "__main__":
    main()
