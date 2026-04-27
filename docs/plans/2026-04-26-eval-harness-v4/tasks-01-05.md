# Tasks 01-05 (Group A — parallel)

---

## Task 1: Move and augment `playbook.yaml` with `triggers:` blocks
**Group:** A

**Behavior being verified:** the canonical playbook lives at `shared/teacher-style/playbook.yaml` and contains a parseable `triggers.score` formula in each of 5 clusters.

**Interface under test:** YAML file structure (loaded via `yaml.safe_load`).

**Files:**
- Move (git mv): `apps/evals/teaching_knowledge/data/playbook.yaml` -> `shared/teacher-style/playbook.yaml`
- Modify: `shared/teacher-style/playbook.yaml`
- Create: `shared/teacher-style/test_playbook_shape.py`

- [ ] **Step 1: Write the failing test**

```python
# shared/teacher-style/test_playbook_shape.py
from pathlib import Path
import yaml

PLAYBOOK = Path(__file__).parent / "playbook.yaml"

EXPECTED = {
    "Technical-corrective feedback",
    "Artifact-based teaching",
    "Positive-encouragement / praise",
    "Motivational / autonomy-supportive statements",
    "Guided-discovery / scaffolding feedback",
}


def _norm(name: str) -> str:
    return (name.replace("‑", "-").replace("–", "-").replace("—", "-")
            .replace("“", "").replace("”", "").strip())


def test_playbook_loads():
    data = yaml.safe_load(PLAYBOOK.read_text())
    assert "teaching_playbook" in data
    assert "clusters" in data["teaching_playbook"]


def test_five_clusters_present():
    data = yaml.safe_load(PLAYBOOK.read_text())
    names = {_norm(c["name"]) for c in data["teaching_playbook"]["clusters"]}
    expected = {_norm(n) for n in EXPECTED}
    assert names == expected


def test_each_cluster_has_triggers_score():
    data = yaml.safe_load(PLAYBOOK.read_text())
    for cluster in data["teaching_playbook"]["clusters"]:
        assert "triggers" in cluster
        assert "score" in cluster["triggers"]
        assert isinstance(cluster["triggers"]["score"], str)
        assert len(cluster["triggers"]["score"]) > 0
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd shared/teacher-style && uv run --with pyyaml --with pytest pytest test_playbook_shape.py -v
```
Expected: FAIL — `FileNotFoundError: shared/teacher-style/playbook.yaml`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```
mkdir -p shared/teacher-style
git mv apps/evals/teaching_knowledge/data/playbook.yaml shared/teacher-style/playbook.yaml
```

Then append a `triggers:` block to each of the 5 clusters at the same indentation as `language_patterns:`:

For "Technical-corrective feedback":
```yaml
      triggers:
        score: "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)"
```

For "Artifact-based teaching":
```yaml
      triggers:
        score: "(1.0 * max_neg_dev if max_neg_dev >= 0.15 else 0) + 0.5 * (1 if duration_min < 20 else 0) + 0.5 * (1 if has_piece else 0)"
```

For "Positive-encouragement / praise":
```yaml
      triggers:
        score: "1.5 * max_pos_dev + 1.5 * (1 if drilling_improved else 0) + 0.5 * (1 if max_neg_dev < 0.1 else 0)"
```

For "Motivational / autonomy-supportive statements":
```yaml
      triggers:
        score: "0.5 * (1 if duration_min < 10 else 0) + 0.5 * (1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0)"
```

For "Guided-discovery / scaffolding feedback":
```yaml
      triggers:
        score: "0.5 * (1 if duration_min > 30 else 0) + 0.5 * (1 if mode_count >= 3 else 0) + 1.0 * (1 if drilling_present and not drilling_improved else 0)"
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd shared/teacher-style && uv run --with pyyaml --with pytest pytest test_playbook_shape.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add shared/teacher-style/playbook.yaml shared/teacher-style/test_playbook_shape.py
git commit -m "feat(teacher-style): move playbook to shared, add cluster triggers"
```

---

## Task 2: `corrupt_signals.shuffle` — deterministic cross-session reassignment
**Group:** A

**Behavior being verified:** `corrupt(top_moments, mode="shuffle", seed=...)` returns top_moments from a different session deterministically.

**Interface under test:** `apps/evals/teaching_knowledge/ablation/corrupt_signals.py::corrupt`.

**Files:**
- Create: `apps/evals/teaching_knowledge/ablation/__init__.py` (empty)
- Create: `apps/evals/teaching_knowledge/ablation/corrupt_signals.py`
- Create: `apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py
import pytest
from teaching_knowledge.ablation.corrupt_signals import corrupt

CORPUS = [
    [{"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"}],
    [{"dimension": "timing", "score": 0.3, "deviation_from_mean": -0.18, "direction": "below_average"}],
    [{"dimension": "pedaling", "score": 0.6, "deviation_from_mean": 0.14, "direction": "above_average"}],
]


def test_shuffle_deterministic_same_seed():
    src = CORPUS[0]
    a = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    b = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    assert a == b


def test_shuffle_returns_other_session_signals():
    src = CORPUS[0]
    out = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    assert out in [CORPUS[1], CORPUS[2]]


def test_shuffle_raises_on_singleton_corpus():
    with pytest.raises(ValueError, match="cannot shuffle"):
        corrupt(CORPUS[0], mode="shuffle", seed=42, all_top_moments=[CORPUS[0]])
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.ablation.corrupt_signals'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teaching_knowledge/ablation/__init__.py
```

```python
# apps/evals/teaching_knowledge/ablation/corrupt_signals.py
"""Three deterministic corruption functions for the white-noise ablation eval."""
from __future__ import annotations
import random
from typing import Any

Mode = str  # "shuffle" | "marginal" | "flip"


def corrupt(
    top_moments: list[dict[str, Any]],
    mode: Mode,
    seed: int,
    all_top_moments: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if mode == "shuffle":
        return _shuffle(top_moments, seed, all_top_moments)
    raise ValueError(f"unknown mode: {mode}")


def _shuffle(src, seed, corpus):
    others = [tm for tm in corpus if tm is not src]
    if not others:
        raise ValueError("cannot shuffle: corpus has no other sessions")
    rng = random.Random(seed)
    return rng.choice(others)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/__init__.py apps/evals/teaching_knowledge/ablation/corrupt_signals.py apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py
git commit -m "feat(ablation): add deterministic shuffle corruption"
```

---

## Task 3: `corrupt_signals.marginal` — per-dim marginal sampling
**Group:** A

**Behavior being verified:** `corrupt(..., mode="marginal", ...)` returns top_moments where each dim's score is sampled from that dim's empirical distribution across the corpus.

**Interface under test:** same `corrupt` function, `mode="marginal"` branch.

**Files:**
- Modify: `apps/evals/teaching_knowledge/ablation/corrupt_signals.py`
- Modify: `apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py`

- [ ] **Step 1: Write the failing test**

Append to `test_corrupt_signals.py`:

```python
def test_marginal_deterministic_same_seed():
    src = CORPUS[0]
    a = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    b = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    assert a == b


def test_marginal_preserves_dim_names():
    src = CORPUS[0]
    out = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    assert {m["dimension"] for m in out} == {m["dimension"] for m in src}


def test_marginal_scores_come_from_corpus():
    src = CORPUS[0]
    out = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    for moment in out:
        observed = [m["score"] for tm in CORPUS for m in tm if m["dimension"] == moment["dimension"]]
        assert moment["score"] in observed
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: FAIL — `ValueError: unknown mode: marginal`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `corrupt_signals.py`, extend `corrupt`:

```python
def corrupt(top_moments, mode, seed, all_top_moments):
    if mode == "shuffle":
        return _shuffle(top_moments, seed, all_top_moments)
    if mode == "marginal":
        return _marginal(top_moments, seed, all_top_moments)
    raise ValueError(f"unknown mode: {mode}")


def _marginal(src, seed, corpus):
    rng = random.Random(seed)
    pools: dict[str, list[float]] = {}
    for tm in corpus:
        for moment in tm:
            pools.setdefault(moment["dimension"], []).append(float(moment["score"]))
    out = []
    for moment in src:
        dim = moment["dimension"]
        pool = pools.get(dim, [moment["score"]])
        new_score = rng.choice(pool)
        out.append({
            "dimension": dim,
            "score": new_score,
            "deviation_from_mean": round(new_score - 0.5, 3),
            "direction": "above_average" if new_score > 0.5 else "below_average",
        })
    return out
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/corrupt_signals.py apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py
git commit -m "feat(ablation): add per-dim marginal corruption"
```

---

## Task 4: `corrupt_signals.flip` — score inversion
**Group:** A

**Behavior being verified:** `corrupt(..., mode="flip", ...)` returns top_moments with `score := 1.0 - score` and direction inverted.

**Interface under test:** same `corrupt` function, `mode="flip"` branch.

**Files:**
- Modify: `apps/evals/teaching_knowledge/ablation/corrupt_signals.py`
- Modify: `apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py`

- [ ] **Step 1: Write the failing test**

Append to `test_corrupt_signals.py`:

```python
def test_flip_inverts_scores():
    src = [
        {"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"},
        {"dimension": "timing", "score": 0.3, "deviation_from_mean": -0.18, "direction": "below_average"},
    ]
    out = corrupt(src, mode="flip", seed=0, all_top_moments=[src])
    assert out[0]["score"] == pytest.approx(0.2)
    assert out[1]["score"] == pytest.approx(0.7)
    assert out[0]["direction"] == "below_average"
    assert out[1]["direction"] == "above_average"


def test_flip_deterministic():
    src = [{"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"}]
    a = corrupt(src, mode="flip", seed=0, all_top_moments=[src])
    b = corrupt(src, mode="flip", seed=999, all_top_moments=[src])
    assert a == b
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: FAIL — `ValueError: unknown mode: flip`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
def corrupt(top_moments, mode, seed, all_top_moments):
    if mode == "shuffle":
        return _shuffle(top_moments, seed, all_top_moments)
    if mode == "marginal":
        return _marginal(top_moments, seed, all_top_moments)
    if mode == "flip":
        return _flip(top_moments)
    raise ValueError(f"unknown mode: {mode}")


def _flip(src):
    out = []
    for moment in src:
        new_score = round(1.0 - float(moment["score"]), 4)
        out.append({
            "dimension": moment["dimension"],
            "score": new_score,
            "deviation_from_mean": round(new_score - 0.5, 3),
            "direction": "above_average" if new_score > 0.5 else "below_average",
        })
    return out
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_corrupt_signals.py -v
```
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/corrupt_signals.py apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py
git commit -m "feat(ablation): add score-flip corruption"
```

---

## Task 5: Atomic-skill rubrics content + schema test
**Group:** A

**Behavior being verified:** `atomic_skill_rubrics.json` contains 8 moves with 5 binary outcome criteria each.

**Interface under test:** static JSON file structure.

**Files:**
- Create: `apps/evals/shared/prompts/atomic_skill_rubrics.json`
- Create: `apps/evals/shared/test_atomic_rubrics_shape.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/shared/test_atomic_rubrics_shape.py
import json
from pathlib import Path

RUBRICS = Path(__file__).parent / "prompts" / "atomic_skill_rubrics.json"

EXPECTED_MOVE_IDS = {
    "voicing_diagnosis", "pedal_triage", "rubato_coaching",
    "phrasing_arc_analysis", "tempo_stability_triage",
    "dynamic_range_audit", "articulation_clarity_check", "exercise_proposal",
}


def _load():
    return json.loads(RUBRICS.read_text())


def test_eight_moves_present():
    data = _load()
    assert isinstance(data, list)
    assert len(data) == 8
    assert {entry["move_id"] for entry in data} == EXPECTED_MOVE_IDS


def test_each_move_has_five_criteria():
    data = _load()
    for entry in data:
        assert "criteria" in entry
        assert len(entry["criteria"]) == 5
        for crit in entry["criteria"]:
            assert "id" in crit
            assert "text" in crit
            assert isinstance(crit["text"], str)
            assert len(crit["text"]) > 10


def test_each_move_has_applies_when():
    data = _load()
    for entry in data:
        assert "applies_when" in entry
        assert isinstance(entry["applies_when"], str)
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_atomic_rubrics_shape.py -v
```
Expected: FAIL — `FileNotFoundError: ...prompts/atomic_skill_rubrics.json`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/evals/shared/prompts/atomic_skill_rubrics.json`:

```json
[
  {
    "move_id": "voicing_diagnosis",
    "applies_when": "synthesis addresses balance between voices or hands",
    "criteria": [
      {"id": "names_voice", "text": "Names a specific voice or hand (melody / inner / bass / LH / RH)."},
      {"id": "specific_locus", "text": "References a specific bar number or time range (e.g., 'bars 3-6')."},
      {"id": "concrete_target", "text": "Prescribes a concrete dynamic or touch target (e.g., 'below mp', 'lighter')."},
      {"id": "avoids_vague", "text": "Avoids vague balance language without specifics ('better balance', 'more musical')."},
      {"id": "one_strategy", "text": "Names exactly one actionable practice strategy (e.g., 'play LH alone')."}
    ]
  },
  {
    "move_id": "pedal_triage",
    "applies_when": "synthesis addresses pedaling",
    "criteria": [
      {"id": "names_phenomenon", "text": "Names which pedal phenomenon (over-pedaling, late change, blurred harmony, dry, flutter)."},
      {"id": "specific_locus", "text": "References a specific bar / harmony change / beat where the issue lives."},
      {"id": "concrete_action", "text": "Prescribes a concrete pedal action (half-pedal, change on beat 3, release before next chord)."},
      {"id": "audible_outcome", "text": "Connects pedaling to an audible musical outcome (clarity, resonance, color)."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (no-pedal first, listen-only pass, mark the pedal in score)."}
    ]
  },
  {
    "move_id": "rubato_coaching",
    "applies_when": "synthesis addresses tempo flexibility or rubato",
    "criteria": [
      {"id": "rubato_vs_drift", "text": "Distinguishes expressive rubato from unintentional drift."},
      {"id": "phrase_locus", "text": "Names where in the phrase the rubato lives (peak, approach, recovery)."},
      {"id": "style_grounding", "text": "References style or composer expectation for rubato use."},
      {"id": "avoids_vague", "text": "Avoids generic 'play with feeling' framing; prescribes a measurable shape."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (metronome to freed, conduct the breath, sing the line)."}
    ]
  },
  {
    "move_id": "phrasing_arc_analysis",
    "applies_when": "synthesis addresses musical sentence shape or phrase",
    "criteria": [
      {"id": "names_arc_parts", "text": "Names at least 2 of {start, peak, resolution} of the phrase explicitly."},
      {"id": "specific_locus", "text": "References a specific bar or beat where the peak / breath sits."},
      {"id": "shape_to_decision", "text": "Connects phrase shape to a dynamic, agogic, or tonal-color decision."},
      {"id": "line_as_unit", "text": "Addresses the line as a unit (does not isolate individual notes)."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (sing the line, slur in groups, breathe at phrase end)."}
    ]
  },
  {
    "move_id": "tempo_stability_triage",
    "applies_when": "synthesis addresses pulse, timing instability, or tempo evenness",
    "criteria": [
      {"id": "drift_vs_rush", "text": "Distinguishes drift (gradual) from rushing (acute) from dragging (acute) - not just 'uneven'."},
      {"id": "specific_locus", "text": "Names where in the piece the instability appears (specific bars or transition)."},
      {"id": "names_cause", "text": "Identifies a likely cause (technical demand, hand-coordination, breath, performance anxiety)."},
      {"id": "tempo_target", "text": "Prescribes a tempo target or a stable reference (metronome BPM, internal subdivision)."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (subdivision, slow-fast-slow, hands-separately at tempo)."}
    ]
  },
  {
    "move_id": "dynamic_range_audit",
    "applies_when": "synthesis addresses dynamics or volume",
    "criteria": [
      {"id": "names_range", "text": "Names the actual range observed (compressed, narrow, mostly mf, lacks pp, lacks ff)."},
      {"id": "specific_locus", "text": "References specific bars where the issue is sharpest."},
      {"id": "expressive_function", "text": "Connects range to expressive function (rising arc, climax, surprise contrast)."},
      {"id": "avoids_vague", "text": "Avoids 'play louder/softer' without target dynamic level or context."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (extreme contrasts as exercise, listen for ceiling/floor)."}
    ]
  },
  {
    "move_id": "articulation_clarity_check",
    "applies_when": "synthesis addresses note attack, release, or articulation",
    "criteria": [
      {"id": "names_articulation_type", "text": "Names the articulation type at issue (legato break, staccato length, accent placement, slur shape)."},
      {"id": "specific_locus", "text": "References specific bars or a specific figure (e.g., 'the 16th-note runs')."},
      {"id": "style_grounding", "text": "Connects articulation to era/style expectation (Baroque clarity, Romantic blend)."},
      {"id": "avoids_vague", "text": "Avoids vague 'clearer' framing; prescribes a touch or finger-level technique."},
      {"id": "practice_strategy", "text": "Includes a practice strategy (slow detached, dotted rhythms, hands separately listening)."}
    ]
  },
  {
    "move_id": "exercise_proposal",
    "applies_when": "synthesis proposes a concrete drill or practice exercise",
    "criteria": [
      {"id": "names_skill", "text": "Names the specific skill the exercise targets (one dim or a specific gesture)."},
      {"id": "passage_scope", "text": "Specifies passage scope (exact bars / hands / tempo)."},
      {"id": "progression", "text": "Specifies progression (start tempo / target tempo, or simple to complex chain)."},
      {"id": "stop_criterion", "text": "Specifies a stop criterion (% accuracy, comfort, 'until even')."},
      {"id": "avoids_vague", "text": "Avoids generic 'practice slowly' framing without measurable terms."}
    ]
  }
]
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_atomic_rubrics_shape.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/prompts/atomic_skill_rubrics.json apps/evals/shared/test_atomic_rubrics_shape.py
git commit -m "feat(eval): add atomic-skill rubrics content (8 moves x 5 binary criteria)"
```
