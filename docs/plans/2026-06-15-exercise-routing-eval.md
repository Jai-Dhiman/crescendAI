# Exercise-Routing Eval Harness Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** An eval harness that scores `ExerciseRoutingDecision` correctness across five deterministic axes with a committed baseline, surfacing routing-prompt regressions before production.
**Spec:** docs/specs/2026-06-15-exercise-routing-eval-design.md
**Style:** Follow apps/api/TS_STYLE.md for any TS changes. Python: uv. No emojis. No backup files. Explicit exceptions over fallbacks.

---

## Task Groups

**Group A (parallel):**
- Task 1: Surface prescribed_exercise in eval_context (TS + Python)
- Task 2: score.py pure scoring module

**Group B (sequential, depends on A):**
- Task 3: shared/local_session.py deep driver

**Group C (sequential, depends on B):**
- Task 4: eval_routing.py orchestrator + baseline.json + justfile recipes

**Group D (sequential, depends on C):**
- Task 5: Smoke test for local_session.py + baseline parse test

**Cleanup (parallel with D — touches different files):**
- Task 6: Delete exercise_quality_judge_v1.txt + remove dead exercise_data placeholder

---

## Task 1: Surface prescribed_exercise in eval_context

**Group:** A (parallel with Task 2)

**Behavior being verified:** When `isEvalSession=true` and V6 synthesis runs, the `eval_context` payload on the synthesis WS message carries `prescribed_exercise` from the artifact.

**Interface under test:** `SynthesisResult.prescribed_exercise` field populated from `eval_context`; verified by reading `pipeline_client.py`'s `SynthesisResult` dataclass.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/evals/shared/pipeline_client.py`
- Test: `apps/evals/pipeline/exercise_routing/tests/test_prescribed_exercise_field.py`

---

- [ ] **Step 1: Write the failing test**

Create `apps/evals/pipeline/exercise_routing/__init__.py` (empty).
Create `apps/evals/pipeline/exercise_routing/tests/__init__.py` (empty).
Create `apps/evals/pipeline/exercise_routing/tests/test_prescribed_exercise_field.py`:

```python
"""Verify that SynthesisResult exposes the prescribed_exercise field."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

from shared.pipeline_client import SynthesisResult


def test_synthesis_result_has_prescribed_exercise_field():
    """SynthesisResult must carry prescribed_exercise parsed from eval_context."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={
            "prescribed_exercise": {
                "kind": "own_passage_loop",
                "target_dimension": "dynamics",
                "bar_range": [1, 4],
                "tempo_factor": 0.8,
            }
        },
    )
    assert result.prescribed_exercise is not None
    assert result.prescribed_exercise["kind"] == "own_passage_loop"
    assert result.prescribed_exercise["target_dimension"] == "dynamics"


def test_synthesis_result_prescribed_exercise_none_when_null():
    """prescribed_exercise is None when the artifact emitted null."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={"prescribed_exercise": None},
    )
    assert result.prescribed_exercise is None


def test_synthesis_result_prescribed_exercise_none_when_absent():
    """prescribed_exercise is None when eval_context has no key (legacy sessions)."""
    result = SynthesisResult(
        text="some synthesis",
        is_fallback=False,
        eval_context={},
    )
    assert result.prescribed_exercise is None
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_prescribed_exercise_field.py -v
```
Expected: FAIL — `SynthesisResult.__init__() got an unexpected keyword argument 'eval_context'` (field does not exist yet) OR AttributeError on `result.prescribed_exercise`.

- [ ] **Step 3: Implement**

**3a. Modify `apps/evals/shared/pipeline_client.py`:**

The `SynthesisResult` dataclass currently has three fields: `text`, `is_fallback`, `eval_context`. Add `prescribed_exercise`:

```python
@dataclass
class SynthesisResult:
    """Session synthesis output from the teacher LLM."""
    text: str
    is_fallback: bool
    eval_context: dict = field(default_factory=dict)
    prescribed_exercise: dict | None = None
```

In the `run_recording` function (around line 248 where `SynthesisResult` is constructed):

```python
synthesis_result = SynthesisResult(
    text=response.get("text", ""),
    is_fallback=response.get("is_fallback", False),
    eval_context=response.get("eval_context", {}),
    prescribed_exercise=response.get("eval_context", {}).get("prescribed_exercise"),
)
```

**3b. Modify `apps/api/src/do/session-brain.ts`:**

Find the block around line 1826 (inside the V6 path, after `buildV6WsPayload`):

```typescript
const wsPayloadWithEval =
    evalContext !== null
        ? { ...wsPayload, eval_context: evalContext }
        : wsPayload;
```

Change it to:

```typescript
const wsPayloadWithEval =
    evalContext !== null
        ? {
            ...wsPayload,
            eval_context: {
                ...evalContext,
                prescribed_exercise: artifact.prescribed_exercise,
                artifact,
            },
          }
        : wsPayload;
```

Note: `artifact` is in scope as the validated `SynthesisArtifact`. Adding the full `artifact` to eval_context gives downstream eval code access to `dominant_dimension` and other fields without additional look-up. `prescribed_exercise` is duplicated at top level for direct access.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_prescribed_exercise_field.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/evals/shared/pipeline_client.py apps/evals/pipeline/exercise_routing/__init__.py apps/evals/pipeline/exercise_routing/tests/__init__.py apps/evals/pipeline/exercise_routing/tests/test_prescribed_exercise_field.py && git commit -m "feat(#48): surface prescribed_exercise + artifact in eval_context"
```

---

## Task 2: score.py pure scoring module

**Group:** A (parallel with Task 1)

**Behavior being verified:** `score_session()` correctly evaluates all five axes for a single session's `SessionCapture`, and `aggregate()` produces correct axis rates.

**Interface under test:** `score_session(capture: SessionCapture) -> SessionScore` and `aggregate(scores: list[SessionScore]) -> AxisScores` — the pure public API of `score.py`.

**Files:**
- Create: `apps/evals/pipeline/exercise_routing/score.py`
- Test: `apps/evals/pipeline/exercise_routing/tests/test_score.py`

(This task creates `score.py` independently of Task 1; `SessionCapture` is defined here as a standalone dataclass with no import from `local_session.py`.)

---

- [ ] **Step 1: Write the failing test**

Create `apps/evals/pipeline/exercise_routing/tests/test_score.py`:

```python
"""Unit tests for score.py through its public interface."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

from pipeline.exercise_routing.score import (
    SessionCapture,
    SessionScore,
    AxisScores,
    score_session,
    aggregate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_capture(
    *,
    piece_slug: str = "fur_elise",
    piece_resolved: bool = True,
    dominant_dimension: str = "dynamics",
    prescribed_exercise: dict | None = None,
    top_moment_bar_range: list[int] | None = None,
) -> SessionCapture:
    teaching_moments = []
    if top_moment_bar_range is not None:
        teaching_moments = [{"dimension": dominant_dimension, "bar_range": top_moment_bar_range}]
    elif dominant_dimension:
        teaching_moments = [{"dimension": dominant_dimension}]
    return SessionCapture(
        session_id="test-session",
        recording=Path("dummy.wav"),
        piece_slug=piece_slug,
        teaching_moments=teaching_moments,
        baselines={},
        piece_identification={"pieceId": "fur-elise", "confidence": 0.95} if piece_resolved else None,
        piece_resolved=piece_resolved,
        dominant_dimension=dominant_dimension,
        prescribed_exercise=prescribed_exercise,
        synthesis_text="some synthesis",
    )


# ---------------------------------------------------------------------------
# invocation_rate
# ---------------------------------------------------------------------------

def test_no_prescription_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.invoked is False


def test_with_prescription_is_invoked():
    capture = make_capture(
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        }
    )
    score = score_session(capture)
    assert score.invoked is True


# ---------------------------------------------------------------------------
# kind_correctness
# ---------------------------------------------------------------------------

def test_kind_correct_own_passage_loop_when_piece_resolved():
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is True


def test_kind_incorrect_corpus_drill_when_piece_resolved():
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is False


def test_kind_correct_corpus_drill_when_not_resolved():
    capture = make_capture(
        piece_resolved=False,
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is True


def test_kind_incorrect_own_passage_loop_without_bar_range():
    """own_passage_loop with null bar_range is a kind violation."""
    capture = make_capture(
        piece_resolved=True,
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": None,
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.kind_correct is False


def test_kind_none_when_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.kind_correct is None


# ---------------------------------------------------------------------------
# dimension_match
# ---------------------------------------------------------------------------

def test_dimension_match_when_equal():
    capture = make_capture(
        dominant_dimension="dynamics",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.dimension_match is True


def test_dimension_mismatch():
    capture = make_capture(
        dominant_dimension="timing",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.dimension_match is False


def test_dimension_match_none_when_not_invoked():
    capture = make_capture(prescribed_exercise=None)
    score = score_session(capture)
    assert score.dimension_match is None


# ---------------------------------------------------------------------------
# bar_range_grounding
# ---------------------------------------------------------------------------

def test_bar_range_grounded_when_overlapping():
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=[2, 6],
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [4, 8],   # overlaps [2,6] at bars 4-6
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is True


def test_bar_range_not_grounded_when_disjoint():
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=[1, 4],
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [10, 14],   # no overlap with [1,4]
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is False


def test_bar_range_grounded_none_when_no_moment_bars():
    """If the top moment has no bar_range, grounding cannot be scored."""
    capture = make_capture(
        piece_resolved=True,
        top_moment_bar_range=None,   # no bar data
        prescribed_exercise={
            "kind": "own_passage_loop",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.8,
        },
    )
    score = score_session(capture)
    assert score.bar_range_grounded is None


# ---------------------------------------------------------------------------
# tempo_sanity
# ---------------------------------------------------------------------------

def test_tempo_in_bounds():
    capture = make_capture(
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 0.7,
        },
    )
    score = score_session(capture)
    assert score.tempo_in_bounds is True


def test_tempo_out_of_bounds():
    capture = make_capture(
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 1.5,   # above max 1.0
        },
    )
    score = score_session(capture)
    assert score.tempo_in_bounds is False


def test_tempo_weak_prior_flag_timing_at_1_0():
    """tempo_factor==1.0 on a timing-dominant session flags as weak prior."""
    capture = make_capture(
        dominant_dimension="timing",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "timing",
            "bar_range": [1, 4],
            "tempo_factor": 1.0,
        },
    )
    score = score_session(capture)
    assert score.tempo_weak_prior_flag is True


def test_tempo_no_weak_prior_flag_non_timing():
    """tempo_factor==1.0 on a non-timing session is not flagged."""
    capture = make_capture(
        dominant_dimension="dynamics",
        prescribed_exercise={
            "kind": "corpus_drill",
            "target_dimension": "dynamics",
            "bar_range": [1, 4],
            "tempo_factor": 1.0,
        },
    )
    score = score_session(capture)
    assert score.tempo_weak_prior_flag is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_invocation_rate():
    scores = [
        SessionScore(session_id="a", piece_slug="p", invoked=True, kind_correct=True,
                     dimension_match=True, bar_range_grounded=None, tempo_in_bounds=True,
                     tempo_weak_prior_flag=False, error=None),
        SessionScore(session_id="b", piece_slug="p", invoked=False, kind_correct=None,
                     dimension_match=None, bar_range_grounded=None, tempo_in_bounds=None,
                     tempo_weak_prior_flag=None, error=None),
        SessionScore(session_id="c", piece_slug="p", invoked=True, kind_correct=True,
                     dimension_match=False, bar_range_grounded=True, tempo_in_bounds=True,
                     tempo_weak_prior_flag=False, error=None),
    ]
    result = aggregate(scores)
    assert result.n_sessions == 3
    assert result.n_invoked == 2
    assert abs(result.invocation_rate - 2/3) < 0.001
    assert abs(result.kind_correctness_rate - 1.0) < 0.001   # 2/2 invoked
    assert abs(result.dimension_match_rate - 0.5) < 0.001    # 1/2 invoked
    assert abs(result.bar_range_grounding_rate - 1.0) < 0.001 # 1/1 with bar data
    assert result.bar_range_grounding_n == 1
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_score.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.exercise_routing.score'`

- [ ] **Step 3: Implement `apps/evals/pipeline/exercise_routing/score.py`**

```python
"""Pure scoring module for ExerciseRoutingDecision correctness.

No I/O. All metric definitions live here. Callable with synthetic SessionCapture
fixtures — the real test surface for the routing eval harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SessionCapture:
    """Structured output of driving one recording through the chunk_ready path."""
    session_id: str
    recording: Path
    piece_slug: str
    teaching_moments: list[dict]
    baselines: dict
    piece_identification: dict | None
    piece_resolved: bool
    dominant_dimension: str | None
    prescribed_exercise: dict | None
    synthesis_text: str


@dataclass
class SessionScore:
    """Per-session scoring result across all five axes."""
    session_id: str
    piece_slug: str
    invoked: bool
    kind_correct: bool | None
    dimension_match: bool | None
    bar_range_grounded: bool | None
    tempo_in_bounds: bool | None
    tempo_weak_prior_flag: bool | None
    error: str | None


@dataclass
class AxisScores:
    """Aggregate harness-level scores across all sessions."""
    invocation_rate: float
    kind_correctness_rate: float
    dimension_match_rate: float
    bar_range_grounding_rate: float
    bar_range_grounding_n: int
    tempo_sanity_rate: float
    tempo_weak_prior_flag_count: int
    n_sessions: int
    n_invoked: int
    n_errors: int


def score_session(capture: SessionCapture) -> SessionScore:
    """Score one session's capture across all five axes. Pure; no I/O."""
    ex = capture.prescribed_exercise
    invoked = ex is not None

    if not invoked:
        return SessionScore(
            session_id=capture.session_id,
            piece_slug=capture.piece_slug,
            invoked=False,
            kind_correct=None,
            dimension_match=None,
            bar_range_grounded=None,
            tempo_in_bounds=None,
            tempo_weak_prior_flag=None,
            error=None,
        )

    kind_correct = _score_kind(capture, ex)
    dimension_match = _score_dimension_match(capture, ex)
    bar_range_grounded = _score_bar_range_grounding(capture, ex)
    tempo_in_bounds, tempo_weak_prior_flag = _score_tempo(capture, ex)

    return SessionScore(
        session_id=capture.session_id,
        piece_slug=capture.piece_slug,
        invoked=True,
        kind_correct=kind_correct,
        dimension_match=dimension_match,
        bar_range_grounded=bar_range_grounded,
        tempo_in_bounds=tempo_in_bounds,
        tempo_weak_prior_flag=tempo_weak_prior_flag,
        error=None,
    )


def aggregate(scores: list[SessionScore]) -> AxisScores:
    """Aggregate per-session scores into harness-level AxisScores. Pure; no I/O."""
    n_sessions = len(scores)
    n_errors = sum(1 for s in scores if s.error is not None)
    invoked_scores = [s for s in scores if s.invoked and s.error is None]
    n_invoked = len(invoked_scores)

    invocation_rate = n_invoked / n_sessions if n_sessions > 0 else 0.0

    kind_judged = [s for s in invoked_scores if s.kind_correct is not None]
    kind_correctness_rate = (
        sum(1 for s in kind_judged if s.kind_correct) / len(kind_judged)
        if kind_judged else 0.0
    )

    dim_judged = [s for s in invoked_scores if s.dimension_match is not None]
    dimension_match_rate = (
        sum(1 for s in dim_judged if s.dimension_match) / len(dim_judged)
        if dim_judged else 0.0
    )

    bar_judged = [s for s in invoked_scores if s.bar_range_grounded is not None]
    bar_range_grounding_n = len(bar_judged)
    bar_range_grounding_rate = (
        sum(1 for s in bar_judged if s.bar_range_grounded) / bar_range_grounding_n
        if bar_range_grounding_n > 0 else 0.0
    )

    tempo_judged = [s for s in invoked_scores if s.tempo_in_bounds is not None]
    tempo_sanity_rate = (
        sum(1 for s in tempo_judged if s.tempo_in_bounds) / len(tempo_judged)
        if tempo_judged else 0.0
    )

    tempo_weak_prior_flag_count = sum(
        1 for s in invoked_scores if s.tempo_weak_prior_flag
    )

    return AxisScores(
        invocation_rate=invocation_rate,
        kind_correctness_rate=kind_correctness_rate,
        dimension_match_rate=dimension_match_rate,
        bar_range_grounding_rate=bar_range_grounding_rate,
        bar_range_grounding_n=bar_range_grounding_n,
        tempo_sanity_rate=tempo_sanity_rate,
        tempo_weak_prior_flag_count=tempo_weak_prior_flag_count,
        n_sessions=n_sessions,
        n_invoked=n_invoked,
        n_errors=n_errors,
    )


# ---------------------------------------------------------------------------
# Private axis scorers
# ---------------------------------------------------------------------------

def _score_kind(capture: SessionCapture, ex: dict) -> bool:
    """Correct kind = own_passage_loop iff piece_resolved; corpus_drill otherwise.

    Guard: own_passage_loop must carry a non-null bar_range (else it's a kind violation).
    """
    kind = ex.get("kind")
    if capture.piece_resolved:
        if kind != "own_passage_loop":
            return False
        bar_range = ex.get("bar_range")
        if not bar_range:
            return False  # own_passage_loop requires bar_range
        return True
    else:
        return kind == "corpus_drill"


def _score_dimension_match(capture: SessionCapture, ex: dict) -> bool | None:
    """prescription.target_dimension == capture.dominant_dimension."""
    if capture.dominant_dimension is None:
        return None
    return ex.get("target_dimension") == capture.dominant_dimension


def _score_bar_range_grounding(capture: SessionCapture, ex: dict) -> bool | None:
    """Prescription bar_range has non-empty intersection with top teaching moment's bar_range.

    Returns None if the top teaching moment has no bar_range (AMT did not provide bars).
    Only scored for own_passage_loop prescriptions (corpus_drill bar_range is a template hint).
    """
    if ex.get("kind") != "own_passage_loop":
        return None
    if not capture.teaching_moments:
        return None
    top_bar_range = capture.teaching_moments[0].get("bar_range")
    if not top_bar_range:
        return None
    prescription_bar = ex.get("bar_range")
    if not prescription_bar:
        return False
    # Non-empty intersection: max(starts) <= min(ends)
    p_start, p_end = prescription_bar[0], prescription_bar[1]
    m_start, m_end = top_bar_range[0], top_bar_range[1]
    return max(p_start, m_start) <= min(p_end, m_end)


def _score_tempo(capture: SessionCapture, ex: dict) -> tuple[bool, bool]:
    """Returns (tempo_in_bounds, tempo_weak_prior_flag).

    In-bounds: [0.25, 1.0] inclusive.
    Weak prior flag: tempo_factor==1.0 on a timing-dominant session (no tempo
    reduction prescribed for the dimension most likely to benefit from it).
    """
    tempo = ex.get("tempo_factor")
    if tempo is None:
        return (True, False)  # absent = schema default = 1.0, treated as in-bounds
    in_bounds = 0.25 <= tempo <= 1.0
    weak_prior = (tempo == 1.0) and (capture.dominant_dimension == "timing")
    return (in_bounds, weak_prior)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_score.py -v
```
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pipeline/exercise_routing/score.py apps/evals/pipeline/exercise_routing/tests/test_score.py && git commit -m "feat(#48): score.py pure routing eval module"
```

---

## Task 3: shared/local_session.py deep driver

**Group:** B (depends on Task 1 — needs `SynthesisResult.prescribed_exercise`)

**Behavior being verified:** `drive()` returns a populated `SessionCapture` in `--skip-inference` mode (the only mode testable without live services).

**Interface under test:** `drive()` and `SessionCapture` from `apps/evals/shared/local_session.py`.

**Files:**
- Create: `apps/evals/shared/local_session.py`
- Test: `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py` (written in Task 5)

The test for this module is in Task 5 (smoke test). This task only creates the module.

---

- [ ] **Step 1: Write the failing smoke test (will be completed in Task 5 — skip for now)**

The smoke test depends on the module existing. Create the module in Step 3 first; Task 5 adds the test.

- [ ] **Step 2: Implement `apps/evals/shared/local_session.py`**

```python
"""Deep driver: one WAV -> SessionCapture over the real chunk_ready path.

Hides WS connect, eval-identity headers, ffmpeg chunking, local R2 upload,
chunk_ready message loop, synthesis event parsing, and eval_context deserialization.
The caller hands over a WAV path and piece slug; the caller receives a SessionCapture.

Not importable during --skip-inference (it will import fine but drive() raises if
services are unavailable — the health check fires before any WS connection).
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import websockets

from shared.pipeline_client import _get_auth_session, _reset_auth_session

CHUNK_SECONDS = 15
R2_BUCKET = "crescendai-bucket"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_API_DIR = REPO_ROOT / "apps" / "api"
DEFAULT_DEV_VARS = DEFAULT_API_DIR / ".dev.vars"


@dataclass
class SessionCapture:
    """Structured output of driving one WAV through the chunk_ready path."""
    session_id: str
    recording: Path
    piece_slug: str
    teaching_moments: list[dict]
    baselines: dict
    piece_identification: dict | None
    piece_resolved: bool
    dominant_dimension: str | None
    prescribed_exercise: dict | None
    synthesis_text: str


def read_eval_secret(dev_vars: Path = DEFAULT_DEV_VARS) -> str:
    """Read EVAL_SHARED_SECRET from apps/api/.dev.vars. Raises if absent or empty."""
    if not dev_vars.exists():
        raise FileNotFoundError(
            f"apps/api/.dev.vars not found at {dev_vars}. "
            "Cannot read EVAL_SHARED_SECRET."
        )
    for line in dev_vars.read_text().splitlines():
        line = line.strip()
        if line.startswith("EVAL_SHARED_SECRET="):
            secret = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not secret:
                raise ValueError("EVAL_SHARED_SECRET is empty in .dev.vars")
            return secret
    raise KeyError("EVAL_SHARED_SECRET not present in .dev.vars")


def check_services(wrangler_url: str) -> None:
    """Raise RuntimeError if the API or MuQ/AMT services are not reachable.

    Called before any recording is processed so the operator gets a single clear
    error rather than 22 per-recording timeouts.
    """
    import requests
    try:
        resp = requests.get(f"{wrangler_url}/health", timeout=5)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"API not reachable at {wrangler_url}/health. "
            "Run `just dev` (or `just api`) before running the eval harness."
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"API health check failed: {resp.status_code}. "
            "Ensure `just dev` is running and `just seed-fingerprint` has been run."
        )


def _slice_to_webm_chunks(wav: Path, out_dir: Path, max_chunks: int) -> list[Path]:
    """ffmpeg-segment WAV into 15s Opus/WebM independently-decodable chunks."""
    pattern = str(out_dir / "chunk_%03d.webm")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(wav),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-c:a", "libopus", "-b:a", "96k", "-ac", "1",
            pattern,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {wav}:\n{result.stderr}")
    chunks = sorted(out_dir.glob("chunk_*.webm"))
    if not chunks:
        raise RuntimeError(f"ffmpeg produced no chunks from {wav}")
    return chunks[:max_chunks]


def _upload_chunk_to_r2(api_dir: Path, r2_key: str, file_path: Path) -> None:
    """Write one chunk into local R2 at the key the DO reads (chunk_ready path)."""
    result = subprocess.run(
        [
            "wrangler", "r2", "object", "put",
            f"{R2_BUCKET}/{r2_key}",
            f"--file={file_path}",
            "--local",
        ],
        cwd=api_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"wrangler r2 put failed for {r2_key}:\n{result.stderr}\n"
            "Is `just dev` (or `just api`) running?"
        )


async def _drive_async(
    recording: Path,
    piece_slug: str,
    session_id: str,
    r2_keys: list[str],
    wrangler_url: str,
    eval_secret: str,
    timeout_per_event: float,
) -> dict:
    """Internal async driver. Returns raw event dict with keys from synthesis event."""
    parsed = urlparse(wrangler_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = (
        f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"
        f"?eval=true&evalStudentId=eval-routing-harness"
    )

    auth = _get_auth_session(wrangler_url)
    headers = {"x-eval-secret": eval_secret}
    token = auth.headers.get("Authorization", "")
    if token:
        headers["Authorization"] = token
    cookie_str = "; ".join(f"{k}={v}" for k, v in auth.cookies.items())
    if cookie_str:
        headers["Cookie"] = cookie_str

    piece_identification: dict | None = None
    synthesis_event: dict | None = None

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "set_piece", "query": piece_slug}))

        for idx, r2_key in enumerate(r2_keys):
            await ws.send(json.dumps({"type": "chunk_ready", "index": idx, "r2Key": r2_key}))

        await ws.send(json.dumps({"type": "end_session"}))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_per_event)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout after {timeout_per_event}s waiting for synthesis "
                    f"for {recording}. Is MuQ:8000 warm?"
                )
            evt = json.loads(raw)
            etype = evt.get("type")
            if etype == "synthesis":
                synthesis_event = evt
                break
            elif etype == "piece_identified":
                piece_identification = evt
            elif etype == "error":
                raise RuntimeError(f"DO returned error: {evt.get('message')}")

    if synthesis_event is None:
        raise RuntimeError(f"No synthesis received for {recording}")

    return {"synthesis": synthesis_event, "piece_identification": piece_identification}


def drive(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    api_dir: Path = DEFAULT_API_DIR,
    eval_secret: str | None = None,
    timeout_per_event: float = 120.0,
    max_chunks: int = 6,
) -> SessionCapture:
    """Drive one WAV through the real chunk_ready path; return a SessionCapture.

    Raises RuntimeError if services are unreachable, ffmpeg fails, wrangler
    r2 put fails, or no synthesis is received within timeout.
    """
    if eval_secret is None:
        eval_secret = read_eval_secret()

    session_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="crescend-routing-eval-") as tmp:
        tmp_dir = Path(tmp)
        chunks = _slice_to_webm_chunks(recording, tmp_dir, max_chunks)

        r2_keys: list[str] = []
        for idx, chunk_path in enumerate(chunks):
            r2_key = f"sessions/{session_id}/chunks/{idx}.webm"
            _upload_chunk_to_r2(api_dir, r2_key, chunk_path)
            r2_keys.append(r2_key)

        result = asyncio.run(
            _drive_async(
                recording=recording,
                piece_slug=piece_slug,
                session_id=session_id,
                r2_keys=r2_keys,
                wrangler_url=wrangler_url,
                eval_secret=eval_secret,
                timeout_per_event=timeout_per_event,
            )
        )

    synth = result["synthesis"]
    piece_id_evt = result["piece_identification"]
    eval_ctx = synth.get("eval_context", {})
    artifact = eval_ctx.get("artifact", {})

    piece_identification = None
    if piece_id_evt:
        piece_identification = {
            "pieceId": piece_id_evt.get("pieceId", ""),
            "confidence": piece_id_evt.get("confidence", 0.0),
        }

    dominant_dimension = artifact.get("dominant_dimension")
    prescribed_exercise = eval_ctx.get("prescribed_exercise")
    teaching_moments = eval_ctx.get("teaching_moments", [])
    baselines = eval_ctx.get("baselines", {})

    return SessionCapture(
        session_id=session_id,
        recording=recording,
        piece_slug=piece_slug,
        teaching_moments=teaching_moments,
        baselines=baselines,
        piece_identification=piece_identification,
        piece_resolved=piece_identification is not None,
        dominant_dimension=dominant_dimension,
        prescribed_exercise=prescribed_exercise,
        synthesis_text=synth.get("text", ""),
    )
```

- [ ] **Step 3: Verify the import is clean**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -c "from shared.local_session import drive, SessionCapture, read_eval_secret; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add apps/evals/shared/local_session.py && git commit -m "feat(#48): shared/local_session.py deep driver for chunk_ready path"
```

---

## Task 4: eval_routing.py orchestrator + baseline.json + justfile recipes

**Group:** C (depends on Task 3)

**Behavior being verified:** `eval_routing.py --skip-inference` exits 0 and prints a summary without connecting to any service; the baseline.json has all five expected axes.

**Interface under test:** CLI entry point `python -m pipeline.exercise_routing.eval_routing --skip-inference`.

**Files:**
- Create: `apps/evals/pipeline/exercise_routing/eval_routing.py`
- Create: `apps/evals/results/exercise_routing/baseline.json`
- Modify: `justfile`

---

- [ ] **Step 1: Write the failing test**

Create `apps/evals/pipeline/exercise_routing/tests/test_baseline.py`:

```python
"""Verify the committed baseline.json parses and has all required axes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

BASELINE_PATH = (
    Path(__file__).parents[4]
    / "results"
    / "exercise_routing"
    / "baseline.json"
)

REQUIRED_AXES = {
    "invocation_rate_floor",
    "kind_correctness_floor",
    "dimension_match_floor",
    "bar_range_grounding_floor",
    "tempo_sanity_floor",
}


def test_baseline_exists():
    assert BASELINE_PATH.exists(), f"baseline.json not found at {BASELINE_PATH}"


def test_baseline_parses():
    data = json.loads(BASELINE_PATH.read_text())
    assert isinstance(data, dict)


def test_baseline_has_all_axes():
    data = json.loads(BASELINE_PATH.read_text())
    missing = REQUIRED_AXES - set(data.keys())
    assert not missing, f"baseline.json missing axes: {missing}"


def test_baseline_floors_are_floats_in_0_1():
    data = json.loads(BASELINE_PATH.read_text())
    for axis in REQUIRED_AXES:
        val = data[axis]
        assert isinstance(val, (int, float)), f"{axis} must be numeric, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{axis}={val} must be in [0.0, 1.0]"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_baseline.py -v
```
Expected: FAIL — `AssertionError: baseline.json not found at ...`

- [ ] **Step 3: Implement**

**3a. Create `apps/evals/results/exercise_routing/baseline.json`:**

```json
{
  "invocation_rate_floor": 0.0,
  "kind_correctness_floor": 0.50,
  "dimension_match_floor": 0.70,
  "bar_range_grounding_floor": 0.50,
  "tempo_sanity_floor": 0.90,
  "notes": "Bootstrap floors from first real run. Run `just exercise-routing-ratchet` after a deliberate improvement."
}
```

**3b. Create `apps/evals/pipeline/exercise_routing/eval_routing.py`:**

```python
"""Thin orchestrator for the exercise-routing eval harness.

Entry point: python -m pipeline.exercise_routing.eval_routing [--skip-inference]

Requires `just dev` (MuQ:8000 + AMT:8001 + API:8787) and `just seed-fingerprint`
unless --skip-inference is passed (CI-safe smoke: validates wiring only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[4]
PRACTICE_EVAL_ROOT = REPO_ROOT / "model" / "data" / "evals" / "practice_eval"
RESULTS_DIR = Path(__file__).parents[3] / "results" / "exercise_routing"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
LAST_RUN_PATH = RESULTS_DIR / "last_run.json"


def build_manifest() -> list[dict]:
    """Discover all practice_eval WAVs and their piece slugs from directory structure."""
    entries = []
    if not PRACTICE_EVAL_ROOT.exists():
        raise FileNotFoundError(
            f"practice_eval root not found: {PRACTICE_EVAL_ROOT}. "
            "Ensure model/data/evals/practice_eval/ is populated."
        )
    for piece_dir in sorted(PRACTICE_EVAL_ROOT.iterdir()):
        if not piece_dir.is_dir():
            continue
        audio_dir = piece_dir / "audio"
        if not audio_dir.exists():
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            entries.append({"recording": wav, "piece_slug": piece_dir.name})
    if not entries:
        raise FileNotFoundError(
            f"No WAV files found under {PRACTICE_EVAL_ROOT}. "
            "Run the audio-acquire recipe or provide recordings."
        )
    return entries


def _check_no_universal_piece_id_failure(scores: list) -> None:
    """Raise if ALL sessions returned piece_resolved=False (seed-fingerprint not run)."""
    from pipeline.exercise_routing.score import SessionScore
    invoked = [s for s in scores if s.invoked and s.error is None]
    if not invoked:
        return  # invocation itself is the issue; baseline gate catches it
    # We detect the seed-fingerprint gotcha by checking if kind_correct is
    # uniformly False with kind=corpus_drill (all sessions degraded to Tier-3).
    # If every invoked session has kind_correct=False and zero own_passage_loop,
    # it means piece-ID returned unknown for everything.
    own_passage_loop_count = sum(
        1 for s in invoked if s.kind_correct  # kind_correct=True only for own_passage_loop or correct corpus_drill
    )
    if own_passage_loop_count == 0 and len(invoked) >= 5:
        raise RuntimeError(
            "All invoked sessions produced corpus_drill with piece_resolved=False. "
            "This indicates `just seed-fingerprint` was not run before this eval. "
            "Run `just seed-fingerprint` and retry."
        )


def _check_baselines(axis_scores, baseline: dict) -> list[str]:
    """Return list of failure messages for axes below their floor thresholds."""
    failures = []
    checks = [
        ("invocation_rate", axis_scores.invocation_rate, baseline["invocation_rate_floor"]),
        ("kind_correctness_rate", axis_scores.kind_correctness_rate, baseline["kind_correctness_floor"]),
        ("dimension_match_rate", axis_scores.dimension_match_rate, baseline["dimension_match_floor"]),
        ("tempo_sanity_rate", axis_scores.tempo_sanity_rate, baseline["tempo_sanity_floor"]),
    ]
    for name, value, floor in checks:
        if value < floor:
            failures.append(f"  {name}: {value:.3f} < floor {floor:.3f}")

    # bar_range_grounding: only check if n > 0
    if axis_scores.bar_range_grounding_n > 0:
        floor = baseline["bar_range_grounding_floor"]
        if axis_scores.bar_range_grounding_rate < floor:
            failures.append(
                f"  bar_range_grounding_rate: {axis_scores.bar_range_grounding_rate:.3f} "
                f"< floor {floor:.3f} (n={axis_scores.bar_range_grounding_n})"
            )

    return failures


def run_skip_inference(baseline: dict) -> int:
    """Validate wiring without services: manifest parses, score.py imports, baseline has all axes."""
    from pipeline.exercise_routing.score import score_session, aggregate, SessionCapture, AxisScores

    print("[exercise-routing-eval] --skip-inference mode: validating wiring only")
    manifest = build_manifest()
    print(f"  manifest: {len(manifest)} recordings across practice_eval/")

    required = {"invocation_rate_floor", "kind_correctness_floor", "dimension_match_floor",
                "bar_range_grounding_floor", "tempo_sanity_floor"}
    missing = required - set(baseline.keys())
    if missing:
        raise RuntimeError(f"baseline.json is missing axes: {missing}")

    print("  score.py: OK")
    print("  baseline.json: OK")
    print("[exercise-routing-eval] smoke PASSED")
    return 0


def run_full(baseline: dict, wrangler_url: str) -> int:
    """Run full eval: drive all recordings, score, write last_run.json, diff baseline."""
    from shared.local_session import drive, check_services, SessionCapture
    from pipeline.exercise_routing.score import score_session, aggregate

    print(f"[exercise-routing-eval] checking services at {wrangler_url} ...")
    check_services(wrangler_url)
    print("  services: OK")

    manifest = build_manifest()
    print(f"  manifest: {len(manifest)} recordings")

    session_scores = []
    for entry in manifest:
        recording: Path = entry["recording"]
        piece_slug: str = entry["piece_slug"]
        print(f"  driving {recording.name} ({piece_slug}) ...", end=" ", flush=True)
        try:
            capture = drive(recording=recording, piece_slug=piece_slug, wrangler_url=wrangler_url)
            score = score_session(capture)
            session_scores.append(score)
            status = "invoked" if score.invoked else "null"
            print(f"{status} kind={score.kind_correct} dim={score.dimension_match}")
        except Exception as exc:
            from pipeline.exercise_routing.score import SessionScore
            err_score = SessionScore(
                session_id=str(recording),
                piece_slug=piece_slug,
                invoked=False,
                kind_correct=None,
                dimension_match=None,
                bar_range_grounded=None,
                tempo_in_bounds=None,
                tempo_weak_prior_flag=None,
                error=str(exc),
            )
            session_scores.append(err_score)
            print(f"ERROR: {exc}")

    _check_no_universal_piece_id_failure(session_scores)

    axis_scores = aggregate(session_scores)

    last_run = {
        "n_sessions": axis_scores.n_sessions,
        "n_invoked": axis_scores.n_invoked,
        "n_errors": axis_scores.n_errors,
        "invocation_rate": round(axis_scores.invocation_rate, 4),
        "kind_correctness_rate": round(axis_scores.kind_correctness_rate, 4),
        "dimension_match_rate": round(axis_scores.dimension_match_rate, 4),
        "bar_range_grounding_rate": round(axis_scores.bar_range_grounding_rate, 4),
        "bar_range_grounding_n": axis_scores.bar_range_grounding_n,
        "tempo_sanity_rate": round(axis_scores.tempo_sanity_rate, 4),
        "tempo_weak_prior_flag_count": axis_scores.tempo_weak_prior_flag_count,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(last_run, indent=2))
    print(f"\n[exercise-routing-eval] last_run.json written to {LAST_RUN_PATH}")

    print("\n--- Axis scores ---")
    for k, v in last_run.items():
        print(f"  {k}: {v}")

    failures = _check_baselines(axis_scores, baseline)
    if failures:
        print("\nFAIL: axes below baseline floors:")
        for f in failures:
            print(f)
        return 1

    print("\nPASS: all axes above baseline floors")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise-routing eval harness")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Smoke mode: validate wiring without live services (CI-safe)")
    parser.add_argument("--wrangler-url", default="http://localhost:8787",
                        help="API base URL (default: http://localhost:8787)")
    args = parser.parse_args()

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"baseline.json not found at {BASELINE_PATH}. "
            "Commit a baseline before running the harness."
        )
    baseline = json.loads(BASELINE_PATH.read_text())

    if args.skip_inference:
        return run_skip_inference(baseline)
    return run_full(baseline, args.wrangler_url)


if __name__ == "__main__":
    sys.exit(main())
```

**3c. Add justfile recipes** — append to `justfile`:

```just
# Exercise-routing eval: drive all practice_eval WAVs through real local inference.
# Requires `just dev` (MuQ:8000 + AMT:8001 + API:8787) and `just seed-fingerprint`.
exercise-routing-eval:
    cd apps/evals && uv run python -m pipeline.exercise_routing.eval_routing

# Exercise-routing eval smoke: validate wiring without live services (CI-safe).
exercise-routing-eval-smoke:
    cd apps/evals && uv run python -m pipeline.exercise_routing.eval_routing --skip-inference

# Promote last_run.json -> baseline.json after a deliberate routing improvement.
exercise-routing-ratchet:
    #!/usr/bin/env bash
    set -euo pipefail
    LAST="apps/evals/results/exercise_routing/last_run.json"
    BASELINE="apps/evals/results/exercise_routing/baseline.json"
    if [ ! -f "$LAST" ]; then
      echo "ERROR: $LAST not found -- run just exercise-routing-eval first"
      exit 1
    fi
    python3 - <<'EOF'
import json, sys
from pathlib import Path
last = json.loads(Path("apps/evals/results/exercise_routing/last_run.json").read_text())
baseline = json.loads(Path("apps/evals/results/exercise_routing/baseline.json").read_text())
out = {
    "invocation_rate_floor": last["invocation_rate"],
    "kind_correctness_floor": last["kind_correctness_rate"],
    "dimension_match_floor": last["dimension_match_rate"],
    "bar_range_grounding_floor": last["bar_range_grounding_rate"],
    "tempo_sanity_floor": last["tempo_sanity_rate"],
    "notes": baseline.get("notes", ""),
}
Path("apps/evals/results/exercise_routing/baseline.json").write_text(json.dumps(out, indent=2))
print("ratcheted baseline.json from last_run.json")
EOF
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_baseline.py -v
```
Expected: PASS (4 tests)

Also verify the smoke recipe works:
```bash
just exercise-routing-eval-smoke
```
Expected: `[exercise-routing-eval] smoke PASSED` (exits 0)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pipeline/exercise_routing/eval_routing.py apps/evals/results/exercise_routing/baseline.json apps/evals/pipeline/exercise_routing/tests/test_baseline.py justfile && git commit -m "feat(#48): eval_routing orchestrator + baseline.json + justfile recipes"
```

---

## Task 5: Smoke test for local_session.py

**Group:** D (depends on Task 3 — local_session.py must exist)

**Behavior being verified:** `local_session.py` module imports cleanly and `read_eval_secret()` raises with a clear error message when the secrets file is absent.

**Interface under test:** `read_eval_secret()` from `apps/evals/shared/local_session.py`.

**Files:**
- Create: `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py`

---

- [ ] **Step 1: Write the failing test**

Create `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py`:

```python
"""Smoke tests for shared/local_session.py -- no live services required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))


def test_local_session_imports():
    """Module must be importable without side effects."""
    from shared.local_session import drive, SessionCapture, read_eval_secret
    assert callable(drive)
    assert callable(read_eval_secret)


def test_read_eval_secret_raises_on_missing_file(tmp_path):
    """read_eval_secret raises FileNotFoundError when .dev.vars is absent."""
    from shared.local_session import read_eval_secret
    missing = tmp_path / "no_such_file.vars"
    with pytest.raises(FileNotFoundError, match="Cannot read EVAL_SHARED_SECRET"):
        read_eval_secret(missing)


def test_read_eval_secret_raises_on_empty_secret(tmp_path):
    """read_eval_secret raises ValueError when the secret is an empty string."""
    from shared.local_session import read_eval_secret
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text('EVAL_SHARED_SECRET=""\n')
    with pytest.raises(ValueError, match="EVAL_SHARED_SECRET is empty"):
        read_eval_secret(dev_vars)


def test_read_eval_secret_raises_on_missing_key(tmp_path):
    """read_eval_secret raises KeyError when .dev.vars has no EVAL_SHARED_SECRET line."""
    from shared.local_session import read_eval_secret
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text("OTHER_VAR=value\n")
    with pytest.raises(KeyError, match="EVAL_SHARED_SECRET not present"):
        read_eval_secret(dev_vars)


def test_read_eval_secret_returns_value(tmp_path):
    """read_eval_secret returns the secret string when correctly configured."""
    from shared.local_session import read_eval_secret
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text('EVAL_SHARED_SECRET="my-test-secret"\n')
    assert read_eval_secret(dev_vars) == "my-test-secret"


def test_session_capture_fields():
    """SessionCapture has all required fields as documented in the spec."""
    from shared.local_session import SessionCapture
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(SessionCapture)}
    required = {
        "session_id", "recording", "piece_slug", "teaching_moments",
        "baselines", "piece_identification", "piece_resolved",
        "dominant_dimension", "prescribed_exercise", "synthesis_text",
    }
    missing = required - field_names
    assert not missing, f"SessionCapture missing fields: {missing}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_local_session_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.local_session'` (file does not exist yet — but wait, Task 3 created it). If Task 3 is already merged, the tests will run and potentially fail for other reasons. Verify against the actual error.

- [ ] **Step 3: Run tests — verify PASS** (no implementation needed, module exists from Task 3)

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/test_local_session_smoke.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 4: Commit**

```bash
git add apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py && git commit -m "test(#48): smoke tests for shared/local_session.py"
```

---

## Task 6: Delete dead scaffolding

**Group:** D-parallel (touches different files from Task 5 — can run concurrently)

**Behavior being verified:** The deleted file and placeholder are gone; no tests reference them.

**Files:**
- Delete: `apps/evals/shared/prompts/exercise_quality_judge_v1.txt`
- Modify: `apps/evals/pipeline/practice_eval/eval_practice.py` (remove dead `exercise_data` placeholder)

---

- [ ] **Step 1: Verify nothing imports exercise_quality_judge_v1.txt**

```bash
grep -r "exercise_quality_judge_v1" /Users/jdhiman/Documents/crescendai/apps/evals --include="*.py"
```
Expected: no output (no Python file imports or reads this prompt).

- [ ] **Step 2: Delete the orphaned prompt file**

```bash
rm /Users/jdhiman/Documents/crescendai/apps/evals/shared/prompts/exercise_quality_judge_v1.txt
```

- [ ] **Step 3: Remove dead exercise_data placeholder from eval_practice.py**

In `apps/evals/pipeline/practice_eval/eval_practice.py`, find and remove:
- Line 244: `exercise_data: list[dict] = []` (the accumulator declaration)
- The checkpoint `"exercise_data": exercise_data,` entry (in the checkpoint save dict around line 476)
- The details `"exercise_data": exercise_data,` entry (in the details dict around line 582)

Exact old_string at line 244 (in the accumulator block):
```python
    exercise_data: list[dict] = []                     # per-exercise judge scores
```
Replace with: (delete line entirely — no replacement)

Exact old_string in the checkpoint save dict (around line 476):
```python
                    "exercise_data": exercise_data,
```
Replace with: (delete line entirely)

Exact old_string in the details dict (around line 582):
```python
        "exercise_data": exercise_data,
```
Replace with: (delete line entirely)

- [ ] **Step 4: Verify eval_practice.py still passes its existing tests**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_analyze_e2e.py -v 2>/dev/null || echo "no test_analyze_e2e"
```
If no directly related tests exist, at minimum verify the file parses:
```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -c "import pipeline.practice_eval.eval_practice; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add -u apps/evals/shared/prompts/exercise_quality_judge_v1.txt apps/evals/pipeline/practice_eval/eval_practice.py && git commit -m "chore(#48): delete orphaned exercise judge prompt + dead exercise_data placeholder"
```

---

## Full test suite verification

After all tasks are complete, run the full new test suite:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest pipeline/exercise_routing/tests/ -v
```
Expected: all tests pass (test_score.py, test_prescribed_exercise_field.py, test_local_session_smoke.py, test_baseline.py).

Run the API TS tests to confirm the session-brain.ts change did not regress:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test --run src/do/session-brain.test.ts
```
Expected: all tests pass.

Run the smoke justfile recipe:

```bash
cd /Users/jdhiman/Documents/crescendai && just exercise-routing-eval-smoke
```
Expected: `[exercise-routing-eval] smoke PASSED`
