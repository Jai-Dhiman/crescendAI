# Tasks 17-21 (Group B — ablation orchestrator + atomic judge)

Tasks 17-19 in `apps/evals/teaching_knowledge/ablation/`. Tasks 20-21 in `apps/evals/shared/`.

---

## Task 17: `run_ablation.py` — orchestrator over fixture session
**Group:** B (depends on Tasks 2-4)

**Behavior:** Given a fixture session and a fake synthesis client, `run_ablation` produces 4 JSONL rows (1 real + 3 corrupted) with correct `condition` labels.

**Interface under test:** `apps/evals/teaching_knowledge/ablation/run_ablation.py::run_ablation`.

**Files:**
- Create: `apps/evals/teaching_knowledge/ablation/run_ablation.py`
- Create: `apps/evals/teaching_knowledge/ablation/test_run_ablation.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/ablation/test_run_ablation.py
import json
from pathlib import Path

from teaching_knowledge.ablation.run_ablation import run_ablation


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.calls.append({"user": user, "system": system})
        marker = "POS" if "above_average" in user else "NEG"
        return f"Practice was good. ({marker})"


SESSION = {
    "recording_id": "rec_001",
    "muq_means": {"dynamics": 0.7, "timing": 0.4},
    "duration_seconds": 600,
    "meta": {"piece_slug": "test_piece", "title": "Test Prelude",
             "composer": "Bach", "skill_bucket": 3},
}


def test_run_ablation_emits_four_rows(tmp_path: Path):
    out = tmp_path / "ablation.jsonl"
    client = FakeClient()
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    conditions = {r["condition"] for r in rows}
    assert conditions == {"real", "shuffle", "marginal", "flip"}
    for r in rows:
        assert r["recording_id"] == "rec_001"
        assert "synthesis_text" in r
        assert "top_moments_used" in r


def test_run_ablation_resume_safe(tmp_path: Path):
    out = tmp_path / "ablation.jsonl"
    client = FakeClient()
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    n_first = len(client.calls)
    run_ablation(sessions=[SESSION], out_path=out, synthesis_client=client, seed=42, skip_judge=True)
    assert len(client.calls) == n_first
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_run_ablation.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.ablation.run_ablation'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teaching_knowledge/ablation/run_ablation.py
"""4-condition signal ablation orchestrator."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Protocol

from teaching_knowledge.ablation.corrupt_signals import corrupt

CONDITIONS = ("real", "shuffle", "marginal", "flip")
SCALER_MEAN = {
    "dynamics": 0.545, "timing": 0.4848, "pedaling": 0.4594,
    "articulation": 0.5369, "phrasing": 0.5188, "interpretation": 0.5064,
}


class SynthesisClient(Protocol):
    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


SYNTHESIS_SYSTEM = (
    "You are a warm, perceptive piano teacher reviewing a practice session. "
    "Respond in 3-6 sentences."
)


def _muq_to_top_moments(muq_means: dict[str, float]) -> list[dict]:
    moments = []
    for dim, score in muq_means.items():
        dev = score - SCALER_MEAN.get(dim, 0.5)
        if abs(dev) >= 0.05:
            moments.append({
                "dimension": dim, "score": score,
                "deviation_from_mean": round(dev, 3),
                "direction": "above_average" if dev > 0 else "below_average",
            })
    moments.sort(key=lambda m: abs(m["deviation_from_mean"]), reverse=True)
    return moments[:4]


def _build_user_msg(top_moments, duration_seconds, meta) -> str:
    payload = {
        "duration_minutes": round(duration_seconds / 60, 1),
        "practice_pattern": "continuous_play",
        "top_moments": top_moments,
        "drilling_records": [],
        "piece": {"title": meta["title"], "composer": meta["composer"], "skill_level": meta["skill_bucket"]},
    }
    return f"<session_data>\n{json.dumps(payload, indent=2)}\n</session_data>\n<task>Write 3-6 sentences.</task>"


def _load_completed(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done.add((row["recording_id"], row["condition"]))
    return done


def run_ablation(*, sessions, out_path, synthesis_client, seed=42, skip_judge=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(out_path)
    all_real_top_moments = [_muq_to_top_moments(s["muq_means"]) for s in sessions]

    with out_path.open("a") as fout:
        for idx, session in enumerate(sessions):
            real_tm = all_real_top_moments[idx]
            for condition in CONDITIONS:
                key = (session["recording_id"], condition)
                if key in completed:
                    continue
                if condition == "real":
                    used_tm = real_tm
                else:
                    used_tm = corrupt(real_tm, mode=condition, seed=seed + idx,
                                      all_top_moments=all_real_top_moments)
                user_msg = _build_user_msg(used_tm, session["duration_seconds"], session["meta"])
                t0 = time.monotonic()
                synth = synthesis_client.complete(user=user_msg, system=SYNTHESIS_SYSTEM, max_tokens=1024)
                lat = round((time.monotonic() - t0) * 1000)
                row = {
                    "recording_id": session["recording_id"],
                    "condition": condition,
                    "top_moments_used": used_tm,
                    "synthesis_text": synth,
                    "synthesis_latency_ms": lat,
                    "judge_skipped": skip_judge,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/ablation/test_run_ablation.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/run_ablation.py apps/evals/teaching_knowledge/ablation/test_run_ablation.py
git commit -m "feat(ablation): 4-condition orchestrator with resume-safety"
```

---

## Task 18: `analyze.py` — cosine similarity
**Group:** B (depends on Task 17)

**Behavior:** `cosine_similarity(a, b)` returns ~1.0 for identical strings and lower for dissimilar.

**Interface under test:** `apps/evals/teaching_knowledge/ablation/analyze.py::cosine_similarity`.

**Files:**
- Create: `apps/evals/teaching_knowledge/ablation/analyze.py`
- Create: `apps/evals/teaching_knowledge/ablation/test_analyze.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/ablation/test_analyze.py
import pytest
from teaching_knowledge.ablation.analyze import cosine_similarity


def test_identical_strings_cosine_one():
    assert cosine_similarity("Practice was good.", "Practice was good.") == pytest.approx(1.0, abs=1e-3)


def test_orthogonal_strings_cosine_lower():
    sim = cosine_similarity(
        "The pedaling needs work in bars 8-12 on the half-pedal change.",
        "Discrete logarithm cryptography over elliptic curves.",
    )
    assert sim < 0.7
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run --with sentence-transformers pytest teaching_knowledge/ablation/test_analyze.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.ablation.analyze'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/teaching_knowledge/ablation/analyze.py
"""Ablation analysis: cosine similarity, four-quadrant binning, decision rule."""
from __future__ import annotations
from functools import lru_cache

from sentence_transformers import SentenceTransformer
import numpy as np


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a: str, b: str) -> float:
    embeddings = _model().encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run --with sentence-transformers pytest teaching_knowledge/ablation/test_analyze.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/analyze.py apps/evals/teaching_knowledge/ablation/test_analyze.py
git commit -m "feat(ablation): cosine similarity via sentence-transformers"
```

---

## Task 19: `analyze.decide_verdict` — pre-registered decision rule
**Group:** B (depends on Task 18)

**Behavior:** Returns `"true"` / `"false"` / `"equivocal"` per spec thresholds.

**Files:** modify `analyze.py` and `test_analyze.py`.

- [ ] **Step 1: Write the failing test** — append:

```python
from teaching_knowledge.ablation.analyze import decide_verdict


def test_verdict_true_when_all_thresholds_met():
    v = decide_verdict(deltas={"flip": 0.4, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.7)
    assert v == "true"


def test_verdict_false_when_flip_delta_low():
    v = decide_verdict(deltas={"flip": 0.1, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.7)
    assert v == "false"


def test_verdict_false_when_high_similarity():
    v = decide_verdict(deltas={"flip": 0.4, "shuffle": 0.2, "marginal": 0.18}, mean_sim_flip=0.95)
    assert v == "false"


def test_verdict_equivocal_in_gap():
    v = decide_verdict(deltas={"flip": 0.2, "shuffle": 0.1, "marginal": 0.1}, mean_sim_flip=0.88)
    assert v == "equivocal"
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run --with sentence-transformers pytest teaching_knowledge/ablation/test_analyze.py -v
```
Expected: FAIL — `ImportError: cannot import name 'decide_verdict'`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append to `analyze.py`:

```python
def decide_verdict(deltas: dict[str, float], mean_sim_flip: float) -> str:
    flip = deltas.get("flip", 0.0)
    shuffle = deltas.get("shuffle", 0.0)
    marginal = deltas.get("marginal", 0.0)
    if flip <= 0.15 or mean_sim_flip >= 0.92:
        return "false"
    if flip > 0.3 and shuffle > 0.15 and marginal > 0.15 and mean_sim_flip < 0.85:
        return "true"
    return "equivocal"
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run --with sentence-transformers pytest teaching_knowledge/ablation/test_analyze.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/ablation/analyze.py apps/evals/teaching_knowledge/ablation/test_analyze.py
git commit -m "feat(ablation): pre-registered verdict decision rule"
```

---

## Task 20: `judge_atomic.py` — happy-path response parsing
**Group:** B (depends on Task 5)

**Behavior:** Parses a well-formed judge JSON response into an `AtomicMatrixResult` with 8 entries.

**Interface under test:** `apps/evals/shared/judge_atomic.py::judge_atomic_matrix`.

**Files:**
- Create: `apps/evals/shared/judge_atomic.py`
- Create: `apps/evals/shared/test_judge_atomic.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/shared/test_judge_atomic.py
import pytest
from shared.judge_atomic import judge_atomic_matrix


class FakeJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_user: str | None = None

    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.last_user = user
        return self.response


HAPPY_RESPONSE = """{
  "moves": [
    {"move_id": "voicing_diagnosis", "attempted": true, "criteria": [true, true, false, true, false]},
    {"move_id": "pedal_triage", "attempted": false, "criteria": null},
    {"move_id": "rubato_coaching", "attempted": false, "criteria": null},
    {"move_id": "phrasing_arc_analysis", "attempted": true, "criteria": [false, false, true, true, false]},
    {"move_id": "tempo_stability_triage", "attempted": false, "criteria": null},
    {"move_id": "dynamic_range_audit", "attempted": false, "criteria": null},
    {"move_id": "articulation_clarity_check", "attempted": false, "criteria": null},
    {"move_id": "exercise_proposal", "attempted": true, "criteria": [true, true, true, false, true]}
  ]
}"""


def test_judge_atomic_parses_happy_response():
    judge = FakeJudge(HAPPY_RESPONSE)
    result = judge_atomic_matrix(
        synthesis_text="Make the LH below mp in bars 3-6. Practice LH alone.",
        context={"piece_name": "Prelude", "composer": "Bach"},
        client=judge,
    )
    assert len(result.moves) == 8
    voicing = next(m for m in result.moves if m.move_id == "voicing_diagnosis")
    assert voicing.attempted is True
    assert voicing.criteria == [True, True, False, True, False]
    pedal = next(m for m in result.moves if m.move_id == "pedal_triage")
    assert pedal.attempted is False
    assert pedal.criteria is None


def test_judge_atomic_includes_synthesis_in_user_msg():
    judge = FakeJudge(HAPPY_RESPONSE)
    judge_atomic_matrix(synthesis_text="UNIQUE_MARKER_TEXT", context={"piece_name": "X", "composer": "Y"}, client=judge)
    assert "UNIQUE_MARKER_TEXT" in judge.last_user
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_judge_atomic.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.judge_atomic'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/shared/judge_atomic.py
"""Single-judge atomic-skill matrix scoring (8 moves x 5 binary criteria)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

RUBRICS_PATH = Path(__file__).parent / "prompts" / "atomic_skill_rubrics.json"


class JudgeClient(Protocol):
    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class MoveResult:
    move_id: str
    attempted: bool
    criteria: list[bool] | None


@dataclass(frozen=True)
class AtomicMatrixResult:
    moves: list[MoveResult]


SYSTEM = (
    "You are a careful evaluator. Given a piano teacher's session synthesis, "
    "judge for each of 8 pedagogical moves whether the synthesis attempted that move "
    "(observable from the text), and if attempted, whether each of 5 binary criteria is satisfied. "
    "Output strict JSON with the documented schema. No prose."
)


def _build_user(synthesis_text: str, context: dict[str, Any]) -> str:
    rubrics = json.loads(RUBRICS_PATH.read_text())
    parts = [
        f"<context>{json.dumps(context)}</context>",
        "<synthesis>", synthesis_text, "</synthesis>",
        "<rubrics>", json.dumps(rubrics, indent=2), "</rubrics>",
        ('Output JSON: {"moves": [{"move_id": "...", "attempted": true|false, '
         '"criteria": [true|false, ...] | null}, ...]} with all 8 move_ids in order.'),
    ]
    return "\n".join(parts)


def judge_atomic_matrix(*, synthesis_text: str, context: dict, client: JudgeClient) -> AtomicMatrixResult:
    user = _build_user(synthesis_text, context)
    raw = client.complete(user=user, system=SYSTEM, max_tokens=2048)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    data = json.loads(text)
    moves_raw = data["moves"]
    moves = [
        MoveResult(
            move_id=m["move_id"],
            attempted=bool(m["attempted"]),
            criteria=[bool(c) for c in m["criteria"]] if m.get("criteria") is not None else None,
        )
        for m in moves_raw
    ]
    return AtomicMatrixResult(moves=moves)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_judge_atomic.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/judge_atomic.py apps/evals/shared/test_judge_atomic.py
git commit -m "feat(eval): atomic-matrix judge happy-path parser"
```

---

## Task 21: `judge_atomic` — malformed-response handling
**Group:** B (depends on Task 20)

**Behavior:** Raises a clear `ValueError` for invalid JSON or missing `moves` key. No silent fallback.

**Files:** modify `judge_atomic.py` and `test_judge_atomic.py`.

- [ ] **Step 1: Write the failing test** — append:

```python
class FakeJudge2:
    def __init__(self, response: str) -> None:
        self.response = response
    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        return self.response


def test_judge_atomic_raises_on_invalid_json():
    judge = FakeJudge2("this is not json")
    with pytest.raises(ValueError, match="atomic judge returned non-JSON"):
        judge_atomic_matrix(synthesis_text="x", context={}, client=judge)


def test_judge_atomic_raises_on_missing_moves_key():
    judge = FakeJudge2('{"foo": []}')
    with pytest.raises(ValueError, match="missing 'moves'"):
        judge_atomic_matrix(synthesis_text="x", context={}, client=judge)
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest shared/test_judge_atomic.py -v
```
Expected: FAIL — first test raises `json.JSONDecodeError` (not `ValueError` with our message).

- [ ] **Step 3: Implement the minimum to make the test pass** — in `judge_atomic.py`, replace the parse block at the end of `judge_atomic_matrix`:

```python
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"atomic judge returned non-JSON: {exc}; raw={raw[:200]!r}") from exc
    if not isinstance(data, dict) or "moves" not in data:
        raise ValueError(f"atomic judge response missing 'moves' key: raw={raw[:200]!r}")
    moves_raw = data["moves"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest shared/test_judge_atomic.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/shared/judge_atomic.py apps/evals/shared/test_judge_atomic.py
git commit -m "feat(eval): atomic-matrix judge raises on malformed response"
```
