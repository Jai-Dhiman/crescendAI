# Memory Eval Improvement + Supermemory Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the memory eval suite with realistic scenarios and machine-readable output, then run autoresearch experiments to improve the memory system.

**Architecture:** Phase 1 upgrades eval infrastructure (scenarios, CLI, JSON output). Phase 2 establishes the autoresearch baseline. Phase 3 runs 6 experiments one at a time against a frozen composite metric. Each experiment modifies scope files (prompts.rs, memory.rs, or eval_synthesis.py), measures, and keeps or reverts.

**Tech Stack:** Python (eval suite, sentence-transformers, groq SDK), Rust (memory system on Cloudflare Workers), D1 (SQLite), autoresearch loop pattern.

**Spec:** `docs/superpowers/specs/2026-03-23-memory-eval-improvement-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `apps/evals/memory/src/build_realistic_scenarios.py` | LLM-powered generator for messy, realistic eval scenarios |
| `apps/evals/memory/data/realistic_scenarios.jsonl` | 15-20 simulated-realistic synthesis scenarios |
| `apps/evals/memory/src/test_infrastructure.py` | Structural tests for run_all.py and scenario generator |
| `apps/evals/memory/results.tsv` | Autoresearch iteration tracking |
| `apps/evals/memory/changelog.md` | Autoresearch experiment log |
| `apps/api/migrations/0009_supersession.sql` | ALTER TABLE for superseded_by column (experiment 4) |

### Modified Files
| File | What changes |
|------|-------------|
| `apps/evals/memory/src/run_all.py` | `--json-output` flag, multi-layer CLI parsing, API health pre-check, composite calculation |
| `apps/evals/memory/src/eval_synthesis.py` | Load realistic scenarios alongside existing, optional LLM-as-judge (experiment 1) |
| `apps/evals/memory/src/eval_temporal.py` | Load realistic temporal scenarios |
| `apps/evals/memory/src/scenarios.py` | Add `version` field to MemoryEvalScenario |
| `apps/api/src/services/prompts.rs` | Synthesis prompt improvements (experiments 2-3) |
| `apps/api/src/services/memory.rs` | Supersession chains, staleness decay, semantic dedup (experiments 4-6) |

---

## Task 1: Add Version Field to Scenario Dataclass

**Files:**
- Modify: `apps/evals/memory/src/scenarios.py:83-97`

- [ ] **Step 1: Add version field to MemoryEvalScenario**

In `apps/evals/memory/src/scenarios.py`, add a `version` field to track scenario set versions across autoresearch iterations:

```python
# At line 97, after temporal_assertions, add:
    version: str = "1.0"  # Scenario set version for autoresearch tracking
```

- [ ] **Step 2: Verify existing scenarios still load**

Run: `cd apps/evals/memory && uv run python -c "from src.scenarios import MemoryEvalScenario; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apps/evals/memory/src/scenarios.py
git commit -m "feat(eval): add version field to MemoryEvalScenario for autoresearch tracking"
```

---

## Task 2: Fix Multi-Layer CLI Parsing + Add JSON Output

**Files:**
- Modify: `apps/evals/memory/src/run_all.py:105-154`
- Create: `apps/evals/memory/src/test_infrastructure.py`

- [ ] **Step 1: Write failing test for multi-layer CLI parsing**

Create `apps/evals/memory/src/test_infrastructure.py`:

```python
"""Structural tests for eval infrastructure."""
import json
import subprocess
import sys


def test_multi_layer_parsing():
    """Multiple --layer flags should all be recognized."""
    # Simulate argparse with multiple --layer flags
    from src.run_all import parse_args
    args = parse_args(["--layer", "synthesis", "--layer", "temporal"])
    assert "synthesis" in args.layers
    assert "temporal" in args.layers
    assert "retrieval" not in args.layers


def test_json_output_format():
    """--json-output should emit valid JSON with expected fields."""
    from src.run_all import parse_args
    args = parse_args(["--json-output"])
    assert args.json_output is True


def test_json_output_composite():
    """Composite formula should be 0.4*synth + 0.3*temp + 0.3*chat."""
    from src.run_all import compute_composite
    result = compute_composite(
        synthesis_recall=1.0,
        temporal_accuracy=1.0,
        chat_precision=1.0,
    )
    assert abs(result - 1.0) < 0.001

    result2 = compute_composite(
        synthesis_recall=0.5,
        temporal_accuracy=0.5,
        chat_precision=0.5,
    )
    assert abs(result2 - 0.5) < 0.001

    result3 = compute_composite(
        synthesis_recall=0.8,
        temporal_accuracy=0.6,
        chat_precision=0.7,
    )
    expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.7
    assert abs(result3 - expected) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/evals/memory && uv run python -m pytest src/test_infrastructure.py -v`
Expected: FAIL (parse_args and compute_composite don't exist yet)

- [ ] **Step 3: Refactor run_all.py CLI parsing**

In `apps/evals/memory/src/run_all.py`, replace the current CLI parsing (around lines 105-130) with proper argparse:

```python
import argparse

LAYERS = ["retrieval", "synthesis", "temporal", "downstream", "chat_extraction", "locomo", "report"]


def parse_args(argv=None):
    """Parse CLI arguments. Accepts multiple --layer flags."""
    parser = argparse.ArgumentParser(description="Memory eval runner")
    parser.add_argument(
        "--layer", dest="layers", action="append", choices=LAYERS,
        help="Layer(s) to run. Repeat for multiple. Omit for all.",
    )
    parser.add_argument("--live", action="store_true", help="Call live APIs instead of cache")
    parser.add_argument("--locomo-samples", type=int, default=10, help="Max LoCoMo samples")
    parser.add_argument("--json-output", action="store_true", help="Emit machine-readable JSON scores")
    args = parser.parse_args(argv)
    if args.layers is None:
        args.layers = LAYERS
    return args


def compute_composite(
    synthesis_recall: float,
    temporal_accuracy: float,
    chat_precision: float,
) -> float:
    """Frozen composite metric for autoresearch. DO NOT change weights during loop."""
    return 0.4 * synthesis_recall + 0.3 * temporal_accuracy + 0.3 * chat_precision
```

Update `main()` to use `parse_args()` and route layers accordingly. Add JSON output at the end:

```python
def main():
    args = parse_args()
    results = {}

    for layer in args.layers:
        try:
            if layer == "synthesis":
                results["synthesis"] = run_synthesis_layer(args)
            elif layer == "temporal":
                results["temporal"] = run_temporal_layer(args)
            elif layer == "chat_extraction":
                results["chat_extraction"] = run_chat_extraction_layer(args)
            elif layer == "retrieval":
                results["retrieval"] = run_retrieval_layer(args)
            # ... other layers
        except Exception as e:
            results[layer] = {"error": str(e)}

    if args.json_output:
        synth_recall = results.get("synthesis", {}).get("recall", 0.0)
        temp_accuracy = results.get("temporal", {}).get("assertion_accuracy", 0.0)
        chat_precision = results.get("chat_extraction", {}).get("precision", 0.0)
        composite = compute_composite(synth_recall, temp_accuracy, chat_precision)
        # Log scenario versions for autoresearch attribution
        scenario_versions = set()
        for layer_result in results.values():
            if isinstance(layer_result, dict):
                for v in layer_result.get("scenario_versions", []):
                    scenario_versions.add(v)
        output = {
            **results,
            "composite": composite,
            "scenario_versions": sorted(scenario_versions),
        }
        print(json.dumps(output, indent=2))
```

Note: The existing layer runner functions need to return dicts instead of just printing. Wrap each layer's existing logic to capture and return metrics. Keep the existing print output for non-JSON mode.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/evals/memory && uv run python -m pytest src/test_infrastructure.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Add API health pre-check**

In `run_all.py`, before running chat_extraction layer:

```python
import urllib.request

def check_api_health() -> bool:
    """Check if local API is running for chat extraction."""
    try:
        urllib.request.urlopen("http://localhost:8787/health", timeout=2)
        return True
    except Exception:
        return False
```

In the chat_extraction branch of the layer loop:

```python
elif layer == "chat_extraction":
    if not check_api_health():
        print("WARNING: localhost:8787 not reachable. Skipping chat_extraction.")
        print("Start API with: cd apps/api && npx wrangler dev")
        results["chat_extraction"] = {"error": "api_unavailable", "precision": 0.0}
    else:
        results["chat_extraction"] = run_chat_extraction_layer(args)
```

When computing composite with missing chat_extraction, reweight:

```python
if results.get("chat_extraction", {}).get("error") == "api_unavailable":
    # Reweight: 0.55 * synth + 0.45 * temporal
    composite = 0.55 * synth_recall + 0.45 * temp_accuracy
    output["composite_note"] = "chat_extraction unavailable, reweighted to 0.55/0.45"
```

- [ ] **Step 6: Commit**

```bash
git add apps/evals/memory/src/run_all.py apps/evals/memory/src/test_infrastructure.py
git commit -m "feat(eval): add --json-output, multi-layer CLI, API health pre-check"
```

---

## Task 3: Build Realistic Scenario Generator

**Files:**
- Create: `apps/evals/memory/src/build_realistic_scenarios.py`
- Create: `apps/evals/memory/data/realistic_scenarios.jsonl`

- [ ] **Step 1: Write the scenario generator**

Create `apps/evals/memory/src/build_realistic_scenarios.py`:

```python
"""Generate simulated-realistic eval scenarios using LLM.

Produces messy observation sequences that mimic real student behavior:
incomplete sessions, piece switches, multi-session arcs, vague engagement,
contradictory observations, and sparse data.

Expected facts are placeholders -- they MUST be hand-annotated by a human
after reviewing the generated observations.
"""
import json
import os
import sys
from dataclasses import asdict

from groq import Groq

from src.scenarios import (
    ExpectedFact,
    MemoryEvalScenario,
    Observation,
    SynthesisCheckpoint,
    TemporalAssertion,
)

GROQ_MODEL = "llama-3.3-70b-versatile"

# Scenario type configs: (category, count, obs_per_session_range, sessions, description)
# Synthesis scenario types
SYNTHESIS_TYPES = [
    ("incomplete", 3, (2, 3), 1, "Student plays 2-3 chunks then stops mid-session"),
    ("piece_switch", 3, (3, 5), 1, "Student switches pieces mid-practice session"),
    ("multi_session_arc", 4, (4, 8), 3, "Same piece over 3 sessions: struggles -> improves -> plateaus"),
    ("vague_engagement", 2, (4, 6), 2, "Student gives ambiguous responses like 'thanks' or 'ok'"),
    ("contradictory", 3, (5, 8), 3, "Observations contradict each other across sessions"),
    ("sparse", 2, (1, 2), 2, "Only 1-2 observations per session, insufficient for high-confidence facts"),
]

# Temporal scenario types (test fact lifecycle over time)
TEMPORAL_TYPES = [
    ("delayed_creation", 2, (3, 5), 3, "Pattern should only be recognized after session 3, not session 2"),
    ("cross_session_invalidation", 3, (4, 6), 3, "Contradictory evidence 2+ weeks later should invalidate existing fact"),
    ("abstention", 3, (1, 2), 2, "Insufficient evidence -- system should NOT create a fact"),
]

GENERATION_PROMPT = """You are generating realistic piano practice observations for an eval dataset.

Student persona: {persona}
Scenario type: {scenario_type} -- {description}
Number of sessions: {n_sessions}
Observations per session: {obs_range[0]}-{obs_range[1]}

Generate a JSON array of observations. Each observation:
{{
  "dimension": one of ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"],
  "observation_text": what the teacher noticed (1-2 sentences, natural language),
  "framing": one of ["correction", "recognition", "encouragement", "question"],
  "dimension_score": float 1.0-5.0 or null,
  "student_baseline": float 1.0-5.0 or null,
  "piece_context": {{"composer": "...", "title": "..."}} or null,
  "session_index": which session (0-indexed),
  "session_date": ISO date (start from 2026-02-01, sessions 3-7 days apart),
  "engaged": boolean (did student ask for more detail?)
}}

IMPORTANT:
- Make observations MESSY and realistic. Real students don't show clean patterns.
- For "{scenario_type}" scenarios: {specific_guidance}
- Include pauses, false starts, and ambiguous feedback.
- Vary the dimensions -- don't focus on just one.
- Use real piece names (Bach, Chopin, Debussy, Beethoven, etc.)

Output ONLY the JSON array, no commentary."""

SPECIFIC_GUIDANCE = {
    "incomplete": "Student stops after 2-3 observations. No clear pattern emerges. System should NOT create confident facts.",
    "piece_switch": "Student starts with one piece, switches to another mid-session. Facts should be piece-scoped, not blended.",
    "multi_session_arc": "Session 1: clear weakness. Session 2: some improvement. Session 3: plateau or slight regression. System should track the trend.",
    "vague_engagement": "Student responds with 'ok', 'thanks', 'got it' -- not 'tell me more'. Engaged should mostly be false.",
    "contradictory": "Session 1: dynamics are flat. Session 2: dynamics improved. Session 3: dynamics flat again. System should handle the contradiction.",
    "sparse": "Very few observations. System should abstain from creating high-confidence facts with only 1-2 data points.",
    "delayed_creation": "Pattern only becomes clear by session 3. Sessions 1-2 have hints but not enough. Synthesis after session 2 should produce NO fact. Synthesis after session 3 should produce one.",
    "cross_session_invalidation": "Session 1-2: clear weakness. Session 3 (2+ weeks later): contradictory evidence. The earlier fact should be invalidated.",
    "abstention": "Only 1-2 vague observations. System should NOT create any fact, or at most a low-confidence one.",
}

PERSONAS = [
    "Adult beginner, 6 months experience, learning Fur Elise",
    "Teenager, grade 5 ABRSM, working on Bach Inventions",
    "Adult returner, played as child, relearning Debussy Clair de Lune",
    "University student, intermediate, exploring Chopin Nocturnes",
    "Self-taught adult, 2 years, working through Beethoven Sonatinas",
    "Retired professional, advanced, polishing Rachmaninoff Prelude",
]


def generate_scenario(
    scenario_id: str,
    category: str,
    obs_range: tuple[int, int],
    n_sessions: int,
    description: str,
    persona: str,
    client: Groq,
) -> MemoryEvalScenario:
    """Generate a single realistic scenario using Groq."""
    prompt = GENERATION_PROMPT.format(
        persona=persona,
        scenario_type=category,
        description=description,
        n_sessions=n_sessions,
        obs_range=obs_range,
        specific_guidance=SPECIFIC_GUIDANCE[category],
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        obs_data = json.loads(raw)
    except json.JSONDecodeError:
        # Retry once on parse error
        response2 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid JSON. Please output ONLY a JSON array of observations."},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        raw2 = response2.choices[0].message.content.strip()
        if raw2.startswith("```"):
            raw2 = raw2.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        obs_data = json.loads(raw2)  # Let it raise if still invalid

    observations = []
    for i, obs in enumerate(obs_data):
        session_idx = obs.get("session_index", 0)
        observations.append(Observation(
            id=f"obs-{scenario_id}-{i:03d}",
            dimension=obs["dimension"],
            observation_text=obs["observation_text"],
            framing=obs.get("framing", "correction"),
            dimension_score=obs.get("dimension_score"),
            student_baseline=obs.get("student_baseline"),
            piece_context=json.dumps(obs["piece_context"]) if obs.get("piece_context") else None,
            session_id=f"sess-{scenario_id}-{session_idx:02d}",
            session_date=obs.get("session_date", "2026-02-01T10:00:00Z"),
            engaged=obs.get("engaged", False),
        ))

    # Create checkpoint at end of all observations
    checkpoints = [SynthesisCheckpoint(
        after_observation_index=len(observations),
        expected_new_facts=[],  # PLACEHOLDER: Must be hand-annotated
        expected_invalidations=[],
    )]

    return MemoryEvalScenario(
        id=scenario_id,
        name=f"Realistic: {category} - {persona[:30]}",
        category=f"realistic_{category}",
        observations=observations,
        checkpoints=checkpoints,
        expected_facts=[],  # PLACEHOLDER: Must be hand-annotated
        retrieval_queries=[],
        temporal_assertions=[],
        version="2.0",
    )


def build_all_realistic_scenarios() -> list[MemoryEvalScenario]:
    """Generate all realistic scenarios."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    scenarios = []
    persona_idx = 0

    all_types = SYNTHESIS_TYPES + TEMPORAL_TYPES
    for category, count, obs_range, n_sessions, description in all_types:
        for i in range(count):
            scenario_id = f"r-{category[:4]}-{i + 1:02d}"
            persona = PERSONAS[persona_idx % len(PERSONAS)]
            persona_idx += 1

            print(f"Generating {scenario_id}: {category} ({persona[:30]}...)")
            scenario = generate_scenario(
                scenario_id=scenario_id,
                category=category,
                obs_range=obs_range,
                n_sessions=n_sessions,
                description=description,
                persona=persona,
                client=client,
            )
            scenarios.append(scenario)
            print(f"  -> {len(scenario.observations)} observations generated")

    return scenarios


def main():
    scenarios = build_all_realistic_scenarios()

    output_path = "data/realistic_scenarios.jsonl"
    with open(output_path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(asdict(s)) + "\n")

    print(f"\nGenerated {len(scenarios)} realistic scenarios to {output_path}")
    print(f"Categories: {', '.join(set(s.category for s in scenarios))}")
    print(f"\nIMPORTANT: Expected facts are EMPTY placeholders.")
    print(f"You must hand-annotate expected_facts for each scenario before running evals.")
    print(f"Review each scenario's observations and write what synthesis SHOULD produce.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add structural validation test**

Append to `apps/evals/memory/src/test_infrastructure.py`:

```python
def test_realistic_scenario_schema():
    """Generated scenarios must have required fields."""
    from src.scenarios import MemoryEvalScenario, Observation
    from dataclasses import fields

    required_obs_fields = {"id", "dimension", "observation_text", "session_id", "session_date"}
    actual_obs_fields = {f.name for f in fields(Observation)}
    assert required_obs_fields.issubset(actual_obs_fields)

    required_scenario_fields = {"id", "name", "category", "observations", "checkpoints", "expected_facts", "version"}
    actual_scenario_fields = {f.name for f in fields(MemoryEvalScenario)}
    assert required_scenario_fields.issubset(actual_scenario_fields)
```

- [ ] **Step 3: Run test**

Run: `cd apps/evals/memory && uv run python -m pytest src/test_infrastructure.py::test_realistic_scenario_schema -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add apps/evals/memory/src/build_realistic_scenarios.py apps/evals/memory/src/test_infrastructure.py
git commit -m "feat(eval): add realistic scenario generator with LLM-powered observation sequences"
```

---

## Task 4: Generate Scenarios + Hand-Annotate Expected Facts

**Files:**
- Create: `apps/evals/memory/data/realistic_scenarios.jsonl`

This task has a HUMAN step -- Jai must review generated observations and write expected facts.

- [ ] **Step 1: Run the generator**

Run: `cd apps/evals/memory && GROQ_API_KEY=$GROQ_API_KEY uv run python -m src.build_realistic_scenarios`
Expected: 17 scenarios generated to `data/realistic_scenarios.jsonl`

- [ ] **Step 2: HUMAN STEP -- Annotate expected facts**

For each scenario in `data/realistic_scenarios.jsonl`:
1. Read the observations
2. Write what synthesis SHOULD produce as `expected_facts`
3. Write `fact_text_pattern` as a regex (or `gold_fact_text` for LLM-as-judge later)
4. Set `trend`, `confidence`, `fact_type`, `dimension`
5. Update `checkpoints.expected_new_facts` with fact IDs
6. For sparse/incomplete scenarios, expected_facts should be EMPTY or low-confidence only
7. Add `temporal_assertions` for temporal scenarios

Estimated time: ~3 hours (10 min per scenario)

- [ ] **Step 3: Validate annotated scenarios**

Run: `cd apps/evals/memory && uv run python -c "
import json
with open('data/realistic_scenarios.jsonl') as f:
    for line in f:
        s = json.loads(line)
        n_obs = len(s['observations'])
        n_facts = len(s['expected_facts'])
        n_cp = len(s['checkpoints'])
        print(f\"{s['id']}: {n_obs} obs, {n_facts} facts, {n_cp} checkpoints\")
        assert n_obs > 0, f'{s[\"id\"]} has no observations'
        assert n_cp > 0, f'{s[\"id\"]} has no checkpoints'
print('All scenarios valid')
"`

- [ ] **Step 4: Commit**

```bash
git add apps/evals/memory/data/realistic_scenarios.jsonl
git commit -m "feat(eval): add 17 hand-annotated realistic scenarios for synthesis + temporal eval"
```

---

## Task 5: Wire Realistic Scenarios Into Eval Layers

**Files:**
- Modify: `apps/evals/memory/src/eval_synthesis.py`
- Modify: `apps/evals/memory/src/eval_temporal.py`

- [ ] **Step 1: Load realistic scenarios in synthesis eval**

In `apps/evals/memory/src/eval_synthesis.py`, find where `scenarios.jsonl` is loaded (near the top of the main evaluation function). Add loading of realistic scenarios:

```python
import os

def load_all_scenarios() -> list:
    """Load both synthetic and realistic scenarios."""
    scenarios = []

    # Existing synthetic scenarios
    with open("data/scenarios.jsonl") as f:
        for line in f:
            scenarios.append(json.loads(line))

    # Realistic scenarios (if file exists and has annotated facts)
    realistic_path = "data/realistic_scenarios.jsonl"
    if os.path.exists(realistic_path):
        with open(realistic_path) as f:
            for line in f:
                s = json.loads(line)
                if s.get("expected_facts"):  # Only include annotated scenarios
                    scenarios.append(s)

    return scenarios
```

Replace the existing scenario loading call with `load_all_scenarios()`.

- [ ] **Step 2: Load realistic temporal scenarios**

In `apps/evals/memory/src/eval_temporal.py`, apply the same pattern -- load both `scenarios.jsonl` and `realistic_scenarios.jsonl`, filtering for scenarios that have `temporal_assertions`.

- [ ] **Step 3: Run synthesis eval to verify it loads both**

Run: `cd apps/evals/memory && uv run python -m src.memory_eval.run_all --layer synthesis`
Expected: Output shows both `sd-*` and `r-*` scenario IDs

- [ ] **Step 4: Commit**

```bash
git add apps/evals/memory/src/eval_synthesis.py apps/evals/memory/src/eval_temporal.py
git commit -m "feat(eval): wire realistic scenarios into synthesis and temporal eval layers"
```

---

## Task 6: Establish Autoresearch Baseline (Iteration 0)

**Files:**
- Create: `apps/evals/memory/results.tsv`
- Create: `apps/evals/memory/changelog.md`

- [ ] **Step 1: Run full eval with JSON output to establish baseline**

Run: `cd apps/evals/memory && uv run python -m src.memory_eval.run_all --layer synthesis --layer temporal --layer chat_extraction --live --json-output 2>&1 | tee /tmp/baseline.json`

If chat_extraction is unavailable (API not running), that's OK -- the pre-check will reweight.

- [ ] **Step 2: Record baseline in results.tsv**

Create `apps/evals/memory/results.tsv`:

```tsv
iter	commit	composite	synth_recall	temp_accuracy	chat_precision	delta	status	description
0	COMMIT_HASH	COMPOSITE	SYNTH	TEMP	CHAT	--	baseline	Initial measurement on synthetic + realistic scenarios
```

Fill in actual values from the JSON output.

- [ ] **Step 3: Create changelog.md**

Create `apps/evals/memory/changelog.md`:

```markdown
# Memory Eval Autoresearch Changelog

## Iteration 0 -- BASELINE
**Date:** 2026-03-23
**Composite:** [value]
**Breakdown:** synth_recall=[value], temp_accuracy=[value], chat_precision=[value]
**Scenarios:** [N] synthetic + [M] realistic = [total]
**Notes:** Baseline measurement. Realistic scenarios added. Matching: regex+cosine (threshold 0.55).
```

- [ ] **Step 4: Commit**

```bash
git add apps/evals/memory/results.tsv apps/evals/memory/changelog.md
git commit -m "experiment: baseline measurement (iteration 0)"
```

---

## Task 7: Experiment 1 -- LLM-as-Judge Matching

**Files:**
- Modify: `apps/evals/memory/src/eval_synthesis.py:235-310`

- [ ] **Step 1: Add LLM-as-judge function**

In `apps/evals/memory/src/eval_synthesis.py`, add a new matching function alongside the existing `_match_facts_batch()`:

```python
JUDGE_PROMPT = """You are evaluating whether two facts about a piano student describe the same pattern.

Produced fact (from synthesis): "{produced}"
Expected fact (ground truth): "{expected}"

Do these describe the same musical pattern or observation about the student?
Consider: same dimension, same behavior, same trend direction.
"Over-pedals through harmony changes" and "sustain pedal bleeds across chord transitions" ARE the same.
"Dynamics improved" and "timing improved" are NOT the same.

Answer ONLY "yes" or "no"."""


def _judge_fact_match(produced: str, expected: str, client) -> bool:
    """Use LLM to judge semantic equivalence of two facts."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            produced=produced, expected=expected,
        )}],
        temperature=0.0,
        max_tokens=10,
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def _match_facts_llm_judge(
    produced_facts: list[dict],
    expected_facts: list,
    client,
) -> dict:
    """Match produced facts to expected facts using LLM-as-judge.

    Pre-filters with cosine similarity at 0.3 to reduce LLM calls.
    Falls back to cosine-only if LLM fails.
    """
    model = _get_sentence_model()
    results = {}
    matched_ef_ids = set()

    # Compute cosine similarity matrix for pre-filtering
    produced_texts = [pf.get("fact_text", "") for pf in produced_facts]
    expected_texts = [_strip_regex(ef.fact_text_pattern) for ef in expected_facts]

    if not produced_texts or not expected_texts:
        return results

    prod_emb = model.encode(produced_texts, convert_to_tensor=True)
    exp_emb = model.encode(expected_texts, convert_to_tensor=True)
    from sentence_transformers import util
    sim_matrix = util.cos_sim(prod_emb, exp_emb)

    # Build candidate pairs above cosine threshold 0.3
    candidates = []
    for pi in range(len(produced_facts)):
        for ei in range(len(expected_facts)):
            if sim_matrix[pi][ei].item() >= 0.3:
                candidates.append((sim_matrix[pi][ei].item(), pi, ei))

    # Sort by similarity descending (most likely matches first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_produced = set()
    for _sim, pi, ei in candidates:
        if pi in used_produced or expected_facts[ei].id in matched_ef_ids:
            continue

        ef = expected_facts[ei]
        pf = produced_facts[pi]

        try:
            is_match = _judge_fact_match(pf.get("fact_text", ""), _strip_regex(ef.fact_text_pattern), client)
        except Exception:
            # Fallback: accept if cosine >= 0.55 (existing threshold)
            is_match = _sim >= 0.55

        if is_match:
            type_ok = pf.get("fact_type", "") == ef.fact_type
            trend_ok = pf.get("trend", "") == ef.trend if ef.trend else True
            results[pi] = (ef.id, type_ok, trend_ok)
            matched_ef_ids.add(ef.id)
            used_produced.add(pi)

    return results
```

- [ ] **Step 2: Add a flag to toggle between matching strategies**

In `eval_synthesis.py`, add a config toggle (environment variable):

```python
USE_LLM_JUDGE = os.environ.get("EVAL_MATCH_STRATEGY", "regex") == "llm_judge"
```

In the evaluation function, use the appropriate matcher:

```python
if USE_LLM_JUDGE:
    matches = _match_facts_llm_judge(produced_facts, expected_facts, groq_client)
else:
    matches = _match_facts_batch(produced_facts, expected_facts)
```

- [ ] **Step 3: Run eval with LLM judge and compare to baseline**

Run: `cd apps/evals/memory && EVAL_MATCH_STRATEGY=llm_judge uv run python -m src.memory_eval.run_all --layer synthesis --live --json-output`

Compare synthesis_recall to baseline. If improved: keep. If same or worse: revert.

- [ ] **Step 4: Record result and keep/revert**

Update `results.tsv` and `changelog.md` with iteration 1 results.

If keeping:
```bash
git add apps/evals/memory/src/eval_synthesis.py apps/evals/memory/results.tsv apps/evals/memory/changelog.md
git commit -m "experiment: LLM-as-judge matching (iteration 1)"
```

If reverting:
```bash
git checkout -- apps/evals/memory/src/eval_synthesis.py
# Still update results.tsv and changelog.md with the revert
git add apps/evals/memory/results.tsv apps/evals/memory/changelog.md
git commit -m "experiment: LLM-as-judge matching - REVERTED (iteration 1)"
```

---

## Task 8: Experiment 2 -- Synthesis Prompt: Multi-Session Awareness

**Files:**
- Modify: `apps/api/src/services/prompts.rs:582-626`

- [ ] **Step 1: Add multi-session guidance to synthesis prompt**

In `apps/api/src/services/prompts.rs`, in the `SYNTHESIS_SYSTEM` constant (line 582), add after the existing rules (before the closing `#"`):

```rust
// Add after "Do NOT generalize beyond what the data shows..." line:
- Pay special attention to cross-session patterns:
  - If the same dimension shows a pattern across 2+ sessions, create a fact with appropriate trend
  - "improving" = consistent progress over sessions, "declining" = regression, "stable" = no change
  - Weight recent sessions more heavily than older ones when trends conflict
  - A single session's improvement does not override a multi-session decline pattern
```

- [ ] **Step 2: Run eval and compare**

Run: `cd apps/evals/memory && uv run python -m src.memory_eval.run_all --layer synthesis --layer temporal --live --json-output`

- [ ] **Step 3: Record and keep/revert**

Update `results.tsv` and `changelog.md`. Commit (keep) or revert.

---

## Task 9: Experiment 3 -- Synthesis Prompt: Abstention Guidance

**Files:**
- Modify: `apps/api/src/services/prompts.rs:582-626`

- [ ] **Step 1: Add abstention guidance**

In `SYNTHESIS_SYSTEM`, add:

```rust
- ABSTAIN from creating facts when evidence is insufficient:
  - Do NOT create high-confidence facts from fewer than 3 observations
  - Do NOT create facts from a single session unless the pattern is unambiguous
  - If only 1-2 observations mention a dimension, set confidence to "low" at most
  - It is better to create NO fact than to create a wrong one
```

- [ ] **Step 2: Run eval and compare**

Run: `cd apps/evals/memory && uv run python -m src.memory_eval.run_all --layer synthesis --layer temporal --live --json-output`

- [ ] **Step 3: Record and keep/revert**

---

## Task 10: Experiment 4 -- Supersession Chains

**Files:**
- Create: `apps/api/migrations/0009_supersession.sql`
- Modify: `apps/api/src/services/memory.rs:9-20` (SynthesizedFact struct)
- Modify: `apps/api/src/services/memory.rs:852-925` (run_synthesis invalidation + insertion)
- Modify: `apps/api/src/services/prompts.rs:582-626` (synthesis prompt)

- [ ] **Step 1: Create migration**

Create `apps/api/migrations/0009_supersession.sql`:

```sql
-- Add supersession tracking to synthesized facts.
-- When a fact is invalidated and replaced, superseded_by points to the new fact.
ALTER TABLE synthesized_facts ADD COLUMN superseded_by TEXT;
```

- [ ] **Step 2: Add field to Rust struct**

In `apps/api/src/services/memory.rs`, add to the `SynthesizedFact` struct (line 9):

```rust
pub struct SynthesizedFact {
    pub id: String,
    pub fact_text: String,
    pub fact_type: String,
    pub dimension: Option<String>,
    pub piece_context: Option<String>,
    pub valid_at: String,
    pub trend: Option<String>,
    pub confidence: String,
    pub source_type: String,
    pub superseded_by: Option<String>,  // NEW: links to replacement fact
}
```

- [ ] **Step 3: Update synthesis to link superseded facts**

In `run_synthesis()` (line 730), after processing invalidations and insertions, link superseded facts. The synthesis LLM output includes `invalidated_facts` and `new_facts`. When an invalidated fact has a corresponding new fact in the same dimension:

```rust
// After inserting new facts (around line 925):
// Link invalidated facts to their replacements
for inv in &invalidated_facts {
    let inv_dimension = &inv.dimension;
    let inv_piece = &inv.piece_context;
    // Find new fact in same dimension AND piece context
    if let Some(new_fact) = new_facts.iter().find(|nf| {
        &nf.dimension == inv_dimension && &nf.piece_context == inv_piece
    }) {
        db.prepare(
            "UPDATE synthesized_facts SET superseded_by = ?1 WHERE id = ?2 AND student_id = ?3",
        )
        .bind(&[
            JsValue::from_str(&new_fact.id),
            JsValue::from_str(&inv.fact_id),
            JsValue::from_str(student_id),
        ])?
        .run()
        .await
        .map_err(|e| format!("Failed to link supersession: {e}"))?;
    }
}
```

- [ ] **Step 4: Add prompt guidance**

In `SYNTHESIS_SYSTEM`, add:

```rust
- When invalidating a fact and creating its replacement, ensure both reference the same dimension.
  The system will automatically link the old fact to the new one for traceability.
```

- [ ] **Step 5: Apply migration locally and run eval**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --file=migrations/0009_supersession.sql`
Then: `cd apps/evals/memory && uv run python -m src.memory_eval.run_all --layer synthesis --layer temporal --live --json-output`

- [ ] **Step 6: Record and keep/revert**

Note: Even if reverted, the migration stays (forward-only). The nullable column is harmless.

---

## Task 11: Experiment 5 -- Proactive Staleness Decay

**Files:**
- Modify: `apps/api/src/services/memory.rs`

- [ ] **Step 1: Add decay_stale_facts function**

In `apps/api/src/services/memory.rs`, add before `run_synthesis()`:

```rust
/// Downgrade confidence of facts with no supporting evidence in 30+ days.
/// Called before synthesis to keep the active fact set fresh.
pub async fn decay_stale_facts(env: &Env, student_id: &str) -> Result<usize, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 error: {e}"))?;
    let cutoff = chrono::Utc::now()
        .checked_sub_signed(chrono::Duration::days(30))
        .unwrap()
        .to_rfc3339();

    // Find facts with valid_at > 30 days ago AND no observations in last 30 days
    // that reference their dimension
    let stale_facts = db
        .prepare(
            "SELECT sf.id, sf.confidence, sf.dimension FROM synthesized_facts sf \
             WHERE sf.student_id = ?1 \
               AND sf.invalid_at IS NULL AND sf.expired_at IS NULL \
               AND sf.valid_at < ?2 \
               AND sf.source_type = 'synthesized' \
               AND NOT EXISTS ( \
                 SELECT 1 FROM observations o \
                 WHERE o.student_id = sf.student_id \
                   AND o.dimension = sf.dimension \
                   AND o.created_at >= ?2 \
               )",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(&cutoff),
        ])?
        .all()
        .await
        .map_err(|e| format!("Stale fact query failed: {e}"))?;

    let mut decayed = 0;
    for row in stale_facts.results::<serde_json::Value>().unwrap_or_default() {
        let fact_id = row["id"].as_str().unwrap_or("");
        let current_confidence = row["confidence"].as_str().unwrap_or("medium");
        let new_confidence = match current_confidence {
            "high" => "medium",
            "medium" => "low",
            _ => continue, // already low, skip
        };

        db.prepare("UPDATE synthesized_facts SET confidence = ?1 WHERE id = ?2")
            .bind(&[JsValue::from_str(new_confidence), JsValue::from_str(fact_id)])?
            .run()
            .await
            .map_err(|e| format!("Confidence update failed: {e}"))?;
        decayed += 1;
    }

    if decayed > 0 {
        console_log!("Decayed confidence for {} stale facts", decayed);
    }
    Ok(decayed)
}
```

- [ ] **Step 2: Call decay before synthesis**

In `run_synthesis()` (line 730), add after the `should_synthesize()` check but before building the prompt:

```rust
// Decay stale facts before synthesis
if let Err(e) = decay_stale_facts(env, student_id).await {
    console_error!("Staleness decay failed (non-fatal): {}", e);
}
```

- [ ] **Step 3: Run eval and compare**

- [ ] **Step 4: Record and keep/revert**

---

## Task 12: Experiment 6 -- Semantic Dedup Before Synthesis

**Files:**
- Modify: `apps/api/src/services/memory.rs`

- [ ] **Step 1: Add dedup_facts_for_synthesis function**

In `apps/api/src/services/memory.rs`, add:

```rust
/// Deduplicate active facts by semantic similarity before sending to synthesis LLM.
/// In-memory only -- D1 facts are NOT modified.
/// Returns a deduplicated list of facts for the synthesis prompt.
pub async fn dedup_facts_for_synthesis(
    env: &Env,
    facts: Vec<SynthesizedFact>,
) -> Vec<SynthesizedFact> {
    if facts.len() < 2 {
        return facts;
    }

    // Call Workers AI for embeddings
    let texts: Vec<String> = facts.iter().map(|f| f.fact_text.clone()).collect();
    let embeddings = match get_embeddings(env, &texts).await {
        Ok(emb) => emb,
        Err(e) => {
            console_error!("Embedding failed for dedup (non-fatal): {}", e);
            return facts; // Fallback: return undeduped
        }
    };

    // Pairwise cosine similarity, merge above 0.85
    let mut merged_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut result = Vec::new();

    for i in 0..facts.len() {
        if merged_indices.contains(&i) {
            continue;
        }
        let mut best = facts[i].clone();
        for j in (i + 1)..facts.len() {
            if merged_indices.contains(&j) {
                continue;
            }
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim >= 0.85 {
                // Keep the more recent fact, but this is in-memory only
                // so we just skip the duplicate
                merged_indices.insert(j);
                console_log!(
                    "Dedup: merged '{}' into '{}' (sim={:.2})",
                    facts[j].fact_text, best.fact_text, sim,
                );
            }
        }
        result.push(best);
    }

    result
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

Note: `get_embeddings()` does not exist yet. Add it to `apps/api/src/services/llm.rs` alongside existing `call_workers_ai()` (line 385). Use the Workers AI embedding model `@cf/baai/bge-small-en-v1.5`:

```rust
// In apps/api/src/services/llm.rs, add after call_workers_ai():
pub async fn get_embeddings(env: &Env, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    let ai = env.ai("AI").map_err(|e| format!("AI binding error: {e}"))?;
    let input = serde_json::json!({ "text": texts });
    let result = ai
        .run("@cf/baai/bge-small-en-v1.5", &input)
        .await
        .map_err(|e| format!("Workers AI embedding failed: {e}"))?;
    // Workers AI returns {"data": [[f32, ...], [f32, ...], ...]}
    let data = result["data"]
        .as_array()
        .ok_or("Missing data field in embedding response")?;
    data.iter()
        .map(|v| {
            v.as_array()
                .ok_or("Invalid embedding vector".to_string())
                .and_then(|arr| {
                    arr.iter()
                        .map(|x| x.as_f64().map(|f| f as f32).ok_or("Invalid float".to_string()))
                        .collect()
                })
        })
        .collect()
}
```

Also add `AI` binding to `wrangler.toml` if not already present:
```toml
[ai]
binding = "AI"
```

- [ ] **Step 2: Call dedup before building synthesis prompt**

In `run_synthesis()`, after querying active facts but before building the prompt:

```rust
// Dedup facts in-memory before synthesis
let deduped_facts = dedup_facts_for_synthesis(env, active_facts).await;
// Use deduped_facts instead of active_facts for prompt building
let prompt = build_synthesis_prompt(&deduped_facts, &new_observations, &approaches, &baselines);
```

- [ ] **Step 3: Run eval and compare**

- [ ] **Step 4: Record and keep/revert**

---

## Task 13: Final Summary + Cleanup

- [ ] **Step 1: Review results.tsv and changelog.md**

After all 6 experiments, review which were kept and which were reverted.

- [ ] **Step 2: Update MEMORY.md**

Update the memory eval entry with final metrics and which experiments landed.

- [ ] **Step 3: Final commit**

```bash
git add apps/evals/memory/results.tsv apps/evals/memory/changelog.md
git commit -m "docs: final autoresearch results - memory eval improvement"
```
