"""Synthesize piano pedagogy principles from literature and web research.

Produces a structured summary of teaching frameworks, principles, and
evidence-based strategies for piano instruction across skill levels.

Usage:
  uv run python -m apps.evals.teaching_knowledge.pedagogy_research \
    --output data/pedagogy_principles.json
"""
import argparse
import json
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent / "data"

RESEARCH_PROMPT = """You are a music education researcher. Synthesize the key principles
from piano pedagogy literature and teaching methodology. Cover these frameworks:

1. Suzuki Method -- key principles, how teachers interact with students
2. ABRSM / RCM exam structure -- how skill levels are defined, what's tested at each grade
3. Taubman Approach -- technical teaching methodology
4. Deliberate Practice (Ericsson) -- how skill acquisition works, what practice is effective
5. Zone of Proximal Development (Vygotsky) -- how to pitch feedback at the right level
6. Motor Learning Theory -- how physical skills are acquired, role of feedback timing
7. Self-Determination Theory (Deci/Ryan) -- intrinsic motivation in music learning
8. Flow Theory (Csikszentmihalyi) -- optimal challenge level for engagement

For each framework, extract:
- Core teaching principles (what the teacher should DO)
- What the teacher should NOT do (common mistakes)
- How feedback should differ by student skill level
- Evidence for effectiveness

Also synthesize cross-framework patterns:
- What do all great teaching methodologies agree on?
- Where do they disagree?
- What's the consensus on feedback timing, frequency, and type?

Return a JSON object with this structure:
{
  "frameworks": [
    {
      "name": "string",
      "core_principles": ["string"],
      "teacher_should_not": ["string"],
      "skill_level_adaptation": {"beginner": "string", "intermediate": "string", "advanced": "string"},
      "evidence": "string"
    }
  ],
  "cross_framework_patterns": {
    "consensus": ["string"],
    "disagreements": ["string"],
    "feedback_principles": ["string"]
  }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "pedagogy_principles.json")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6-20250514")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    print("Synthesizing pedagogy literature...")

    response = client.messages.create(
        model=args.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": RESEARCH_PROMPT}],
    )

    try:
        result = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        # Save raw text if JSON parsing fails
        result = {"raw_response": response.content[0].text}
        print("WARNING: Response was not valid JSON. Saving raw text.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {len(result.get('frameworks', []))} frameworks to {args.output}")


if __name__ == "__main__":
    main()
