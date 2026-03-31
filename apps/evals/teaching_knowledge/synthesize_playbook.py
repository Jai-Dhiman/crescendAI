"""Synthesize Teaching Playbook from raw teaching database + pedagogy principles.

Three rounds:
  Round 1: Cluster teaching moments by skill level and feedback type
  Round 2: Extract patterns, dimension priorities, language patterns
  Round 3: Founder review checkpoint (manual)

Usage:
  uv run python -m apps.evals.teaching_knowledge.synthesize_playbook \
    --teaching-db data/raw_teaching_db.json \
    --pedagogy data/pedagogy_principles.json \
    --output data/playbook.yaml
"""
import argparse
import json
from pathlib import Path

import yaml

from teaching_knowledge.llm_client import LLMClient

DATA_DIR = Path(__file__).parent / "data"

ROUND1_PROMPT = """You are analyzing a database of {n_moments} teaching moments extracted
from real piano masterclasses and lessons, plus {n_frameworks} pedagogical frameworks.

TEACHING MOMENTS (sample of {sample_size}):
{moments_sample}

PEDAGOGY PRINCIPLES:
{pedagogy_summary}

TASK: Group these teaching moments into natural clusters. Do NOT pre-assume an organizing
principle (tiers, moves, etc.). Let the patterns emerge from the data.

For each cluster you identify:
1. Name it descriptively (what teaching behavior defines this cluster?)
2. List 3-5 representative examples from the data
3. Note which skill levels this cluster appears at
4. Note which dimensions this cluster focuses on
5. Note the distribution of feedback types (corrective vs encouraging etc.)

Return a JSON object with clusters as an array.
"""

ROUND2_PROMPT = """You are building a Teaching Playbook for an AI piano teacher.

Round 1 identified these clusters:
{clusters}

Raw teaching data ({n_moments} moments) and pedagogy research ({n_frameworks} frameworks)
are available.

FULL TEACHING DATA:
{all_moments}

For each cluster, now extract:
1. Dominant teaching strategies and when to use them
2. Dimension priorities by repertoire style (e.g., pedaling for Chopin vs articulation for Bach)
3. Language patterns and register (warm, direct, technical, metaphorical)
4. Good feedback examples (with evidence citations from the data)
5. Bad feedback patterns (what mediocre teachers do instead)
6. What distinguishes great feedback from mediocre in this cluster

Also produce cross-cluster insights:
- What do great piano teachers do that mediocre ones don't?
- What should the AI teacher NEVER say at each skill level?
- Piece-style dimension rules (which dimensions matter for which composers/styles?)

Return a YAML document that could serve as a Teaching Playbook.
"""


def sample_moments(moments: list[dict], max_sample: int = 50) -> str:
    """Sample moments for prompt context, preserving diversity."""
    if len(moments) <= max_sample:
        return json.dumps(moments, indent=2)

    # Stratified sample by skill level
    by_skill = {}
    for m in moments:
        skill = m.get("student_skill_estimate", "unknown")
        by_skill.setdefault(skill, []).append(m)

    per_skill = max_sample // max(len(by_skill), 1)
    sample = []
    for skill_moments in by_skill.values():
        sample.extend(skill_moments[:per_skill])

    return json.dumps(sample[:max_sample], indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teaching-db", type=Path, default=DATA_DIR / "raw_teaching_db.json")
    parser.add_argument("--pedagogy", type=Path, default=DATA_DIR / "pedagogy_principles.json")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "playbook.yaml")
    parser.add_argument("--provider", type=str, default="workers-ai", choices=["workers-ai", "anthropic"])
    parser.add_argument("--model", type=str, default=None, help="Override model (default: provider's quality model)")
    args = parser.parse_args()

    moments = json.loads(args.teaching_db.read_text())
    pedagogy = json.loads(args.pedagogy.read_text())
    client = LLMClient(provider=args.provider, model=args.model, tier="quality")

    print(f"Loaded {len(moments)} teaching moments, {len(pedagogy.get('frameworks', []))} frameworks")
    print(f"Using: {client}")

    # Round 1: Clustering
    print("\n--- Round 1: Clustering ---")
    r1_prompt = ROUND1_PROMPT.format(
        n_moments=len(moments),
        n_frameworks=len(pedagogy.get("frameworks", [])),
        sample_size=min(50, len(moments)),
        moments_sample=sample_moments(moments),
        pedagogy_summary=json.dumps(pedagogy.get("cross_framework_patterns", {}), indent=2),
    )
    clusters = client.complete(r1_prompt, max_tokens=8000)
    print(f"Clusters identified. Response length: {len(clusters)} chars")

    # Save Round 1 output for review
    r1_path = DATA_DIR / "playbook_round1_clusters.json"
    r1_path.write_text(clusters)
    print(f"Round 1 saved to {r1_path}")

    # Round 2: Pattern extraction
    print("\n--- Round 2: Pattern Extraction ---")
    r2_prompt = ROUND2_PROMPT.format(
        clusters=clusters,
        n_moments=len(moments),
        n_frameworks=len(pedagogy.get("frameworks", [])),
        all_moments=json.dumps(moments, indent=2)[:50000],  # Cap at 50K chars
    )
    playbook_text = client.complete(r2_prompt, max_tokens=16000)
    print(f"Playbook draft generated. Response length: {len(playbook_text)} chars")

    # Try to parse as YAML, fall back to saving raw text
    try:
        # Strip markdown code fences if present
        clean = playbook_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        playbook = yaml.safe_load(clean)
        args.output.write_text(yaml.dump(playbook, default_flow_style=False, allow_unicode=True))
    except yaml.YAMLError:
        args.output.write_text(playbook_text)
        print("WARNING: Could not parse as YAML. Saved raw text.")

    print(f"\nPlaybook saved to {args.output}")
    print("\n--- Round 3: FOUNDER REVIEW REQUIRED ---")
    print("Review the playbook at:")
    print(f"  {args.output}")
    print("\nQuality gate checklist:")
    print("  [ ] At least 3 distinct feedback patterns per skill level")
    print("  [ ] At least 1 pattern that surprises you (something the pipeline doesn't do)")
    print("  [ ] Piece-style dimension rules present (Bach vs Chopin etc.)")
    print("  [ ] Good AND bad examples for each pattern")


if __name__ == "__main__":
    main()
