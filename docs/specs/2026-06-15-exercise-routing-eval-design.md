# Exercise-Routing Eval Harness Design

**Goal:** An eval harness that scores per-session `ExerciseRoutingDecision` correctness across five deterministic axes, with a committed baseline, surfacing regressions from routing-prompt changes before they reach production.

**Not in scope:**
- LLM judge for exercise prose quality (the `ExerciseRoutingDecision` schema has no prose field)
- Remote/CI-driven real inference (this is a local-first harness; CI only runs the --skip-inference smoke check)
- Scoring historical `eval_chunk`-path holdout sessions (they have `piece_resolved=False` by construction — structurally degenerate for `kind` and `bar_range` axes)
- Any `corpus_drill` primitive_id resolution or exercise retrieval validation
- Multi-recording ensemble statistical confidence beyond directional floor thresholds

---

## Problem

The `ExerciseRoutingDecision` (kind, target_dimension, bar_range, tempo_factor) shipped in #29, but there is zero measurement of routing correctness. The only way a prompt regression is caught today is by manually reading session output. With N≈22 recordings across 16 pieces, this harness is deliberately a **directional regression detector** — if a routing-prompt change tanks `dimension_match` below 70%, it fires. It is not a precision instrument (inference + LLM variance → wide CIs at N=22).

---

## Solution (from the user's perspective)

After `just dev` + `just seed-fingerprint`, the developer runs `just exercise-routing-eval`. The harness drives each practice_eval WAV through real MuQ+AMT+teacher inference, captures the `prescribed_exercise` from the synthesis artifact, evaluates five axes, writes `last_run.json`, and diffs it against a committed `baseline.json`. If any axis falls below its floor threshold, the harness exits non-zero and prints which axis failed and by how much.

After a deliberate routing improvement the developer runs `just exercise-routing-ratchet` to promote `last_run.json` → `baseline.json`.

In CI (no services), `just exercise-routing-eval --skip-inference` validates wiring (manifest parses, score.py loads, baseline.json has all axes) without touching any network.

---

## Design

### Why the chunk_ready path, not eval_chunk

The existing `#22` DO-path holdout drives `eval_chunk`, which bypasses real MuQ+AMT. That means:
- `piece_resolved=False` for all 98 sessions (no WASM piece-ID ran; no v2 fingerprint was seeded)
- `bar_range=null` on all teaching moments (AMT never ran; no alignment data)

With those two fields null, `kind_correctness` and `bar_range_grounding` — two of the five axes — are structurally unscoreable. Confirmed by reading `apps/evals/results/baseline_v2_do.jsonl`.

This harness drives `chunk_ready` with real audio uploaded to local R2 (exactly what `scripts/drive_local_session.py` does). The eval-identity override (`?eval=true&evalStudentId=`) is used to attach `prescribed_exercise` to the `eval_context` payload — it does NOT redirect inference away from `chunk_ready`.

### The cross-package change: surface prescribed_exercise in eval_context

The `eval_context` object (attached to the `synthesis` WS message when `isEvalSession=true`) currently carries `scored_chunks`, `teaching_moments`, `baselines`, `mode_transitions`, `drilling_records`, `timeline`. It does NOT carry the `prescribed_exercise` from the `SynthesisArtifact`.

One TS line is added inside the `if (state.isEvalSession)` block in `apps/api/src/do/session-brain.ts` at line 1559:

```ts
prescribed_exercise: artifact.prescribed_exercise,
```

`artifact` is in scope at that point (it is the validated `SynthesisArtifact`). The `SynthesisResult` dataclass in `apps/evals/shared/pipeline_client.py` is extended with a `prescribed_exercise: dict | None` field parsed from `eval_context`.

### Why stats-only v1 (no LLM judge)

The orphaned `apps/evals/shared/prompts/exercise_quality_judge_v1.txt` evaluates *prose instructions* (Actionability, Specificity, Progression). The `ExerciseRoutingDecision` has no prose — only `kind`, `target_dimension`, `bar_range`, `tempo_factor`. The judge criteria map 1-for-1 to statistical axes that can be checked deterministically and without LLM cost or rate-limit friction. Deleting the prompt file removes dead scaffolding; it is not wired into any current test.

### The "golden set" is the manifest, not frozen labels

Per-axis "expected" values are derived live from each session's own accumulated state (the `eval_context` fields). For example, `kind_correctness` compares `prescription.kind` against `piece_resolved` from `eval_context.teaching_moments` — not against a hand-authored expected label. This makes the manifest stable across routing-prompt changes.

### Noise handling

Two noise sources per run: inference stochasticity (MuQ float outputs) and teacher LLM stochasticity (routing decision). Floor thresholds in `baseline.json` absorb this. The ratchet recipe promotes new floors only when the developer deliberately verifies an improvement.

### Failure behavior (explicit exceptions over silent fallbacks)

- Services down: FAIL LOUDLY with a clear error message naming which service and its expected URL.
- Seed-fingerprint not run: FAIL LOUDLY. Piece-ID returning `unknown` silently collapses every session to `corpus_drill` — a known gotcha from MEMORY. The harness detects this by checking if ALL sessions return `piece_resolved=False` (no piece identification happened) and raises `RuntimeError`.
- Per-recording failure: error row in `last_run.json`, never swallowed. Harness continues remaining recordings.
- `prescribed_exercise=null`: valid row, counted in `invocation_rate` denominator.
- `bar_range_grounding` coverage below 10% (too few sessions with AMT bar data): logged explicitly; axis is reported with `n=<coverage_count>` warning; harness does not crash.

---

## Modules

### Module 1: `apps/evals/shared/local_session.py`

**Interface:**
```python
@dataclass
class SessionCapture:
    session_id: str
    recording: Path
    piece_slug: str
    teaching_moments: list[dict]    # from eval_context
    baselines: dict                 # from eval_context
    piece_identification: dict | None  # {"pieceId": str, "confidence": float} | None
    piece_resolved: bool            # True iff piece_identification is not None
    dominant_dimension: str | None  # from artifact inside eval_context
    prescribed_exercise: dict | None  # ExerciseRoutingDecision | null
    synthesis_text: str

def drive(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    api_dir: Path = ...,  # apps/api, for wrangler r2 put
    eval_secret: str = ...,  # read from .dev.vars
    timeout_per_event: float = 120.0,
) -> SessionCapture:
    """Drive one WAV through the real chunk_ready path; return a SessionCapture."""
```

**Hides:** WS connect, eval-identity header composition, ffmpeg WAV→WebM chunking, `wrangler r2 object put --local`, `chunk_ready` message loop, synthesis event parsing, `eval_context` deserialization, per-event dispatch, re-auth on 401.

**Depth verdict:** DEEP — the caller needs to hand over a WAV and get back structured data; it never sees WS frames.

### Module 2: `apps/evals/pipeline/exercise_routing/score.py`

**Interface:**
```python
@dataclass
class AxisScores:
    invocation_rate: float              # count(prescription != null) / N
    kind_correctness_rate: float        # correct kind decisions / N
    dimension_match_rate: float         # target_dim == dominant_dim / invoked
    bar_range_grounding_rate: float     # overlap / sessions_with_bar_data
    bar_range_grounding_n: int          # how many sessions had bar data
    tempo_sanity_rate: float            # in-bounds / invoked
    tempo_weak_prior_flag_count: int    # tempo_factor==1.0 on timing-flagged sessions
    n_sessions: int
    n_invoked: int
    n_errors: int

@dataclass  
class SessionScore:
    session_id: str
    piece_slug: str
    invoked: bool
    kind_correct: bool | None          # None if not invoked
    dimension_match: bool | None       # None if not invoked
    bar_range_grounded: bool | None    # None if no bar data on top moment
    tempo_in_bounds: bool | None       # None if not invoked
    tempo_weak_prior_flag: bool | None # None if not invoked
    error: str | None

def score_session(capture: SessionCapture) -> SessionScore:
    """Score one session's capture. Pure; no I/O."""

def aggregate(scores: list[SessionScore]) -> AxisScores:
    """Aggregate per-session scores into harness-level AxisScores. Pure; no I/O."""
```

**Hides:** All metric-definition logic. The caller never implements overlap checks, bar_range guard logic, tempo_factor bound checks, or the weak-prior flag heuristic.

**Depth verdict:** DEEP — simple 2-function interface over all scoring rules.

### Module 3: `apps/evals/pipeline/exercise_routing/eval_routing.py`

**Interface:** Entry point only — `python -m pipeline.exercise_routing.eval_routing [--skip-inference] [--wrangler-url URL]`

**Hides:** Manifest discovery, service health checks, per-recording `drive()` loop, parallel/sequential dispatch, `last_run.json` write, baseline diff and threshold gate.

**Depth verdict:** SHALLOW by design — thin orchestrator. Acceptable; it is a script, not a reusable library.

---

## Scoring Axis Definitions

All definitions live in `score.py`. These are contractual; changing them requires a ratchet.

**invocation_rate** — `count(prescription != null) / N_total`. Reference range 20-40%. Floor threshold: >0% (harness fails if 0/22 sessions get a prescription; indicates DO routing is broken).

**kind_correctness** — For each session: if `piece_resolved=True`, expected kind is `own_passage_loop`; if `piece_resolved=False`, expected kind is `corpus_drill`. PLUS guard: `own_passage_loop` must carry a non-null `bar_range`. A prescription with `kind=own_passage_loop` but `bar_range=null` counts as incorrect. Floor threshold: >50%.

**dimension_match** — Among invoked sessions: `prescription.target_dimension == dominant_dimension` from `eval_context`. `dominant_dimension` is read from `eval_context.artifact.dominant_dimension` (the top-level field of `SynthesisArtifact`). Floor threshold: >70%.

**bar_range_grounding** — Among sessions where `eval_context.teaching_moments[0].bar_range` is non-null AND the prescription is `own_passage_loop`: the prescription's `bar_range` must have non-empty intersection with the top teaching moment's `bar_range`. Coverage count is logged. Floor threshold: >50% of sessions with bar data. If coverage is 0 (AMT returned no bars for any recording), the axis is reported with `n=0` and NOT checked against the floor — it would indicate AMT is down, which the service health check catches first.

**tempo_sanity** — Schema-guaranteed [0.25, 1.0]; harness verifies this is not violated. Additionally, if `dominant_dimension == "timing"` (timing-flagged session) and `tempo_factor == 1.0`, flag as weak prior (no tempo reduction prescribed for a timing-weak session). Reported as a count, not a floor.

---

## Verification Architecture

**Canonical success state:** `last_run.json` with all five axes above their floor thresholds, written by driving the 22 practice_eval WAVs through real inference.

**Automated check — fast (CI):**
```bash
just exercise-routing-eval --skip-inference
```
Validates: manifest parses, `score.py` imports, `baseline.json` contains all five axes. No network calls. Completes in <5 seconds.

**Automated check — full:**
```bash
just exercise-routing-eval
```
Requires `just dev` + `just seed-fingerprint`. Drives all recordings; writes `last_run.json`; exits non-zero if any axis falls below its floor.

**Harness buildable before feature?** No — the harness requires the `prescribed_exercise` field in `eval_context`, which is the TS change in Task 1. Task 1 is therefore Group A (prerequisite). The unit tests for `score.py` (Task 2) can be written against synthetic `SessionCapture` fixtures and are independent of the TS change.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/do/session-brain.ts` | Add `prescribed_exercise: artifact.prescribed_exercise` to `evalContext` object (line ~1565, inside `if (state.isEvalSession)` block) | Modify |
| `apps/evals/shared/pipeline_client.py` | Add `prescribed_exercise: dict \| None` field to `SynthesisResult`; parse it from `eval_context` | Modify |
| `apps/evals/shared/local_session.py` | New deep module: `drive()` → `SessionCapture`. Extracts logic from `scripts/drive_local_session.py` | New |
| `apps/evals/pipeline/exercise_routing/__init__.py` | Empty init | New |
| `apps/evals/pipeline/exercise_routing/score.py` | Pure scoring module: `score_session()`, `aggregate()`, dataclasses | New |
| `apps/evals/pipeline/exercise_routing/eval_routing.py` | Thin orchestrator; CLI entry point | New |
| `apps/evals/pipeline/exercise_routing/tests/__init__.py` | Empty init | New |
| `apps/evals/pipeline/exercise_routing/tests/test_score.py` | Unit tests for `score.py` via public interface | New |
| `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py` | Smoke test for local_session.py in --skip-inference mode | New |
| `apps/evals/pipeline/exercise_routing/tests/test_baseline.py` | Baseline JSON parses + has all axes | New |
| `apps/evals/results/exercise_routing/baseline.json` | Committed baseline (floor thresholds per axis) | New |
| `apps/evals/shared/prompts/exercise_quality_judge_v1.txt` | Delete (mismatched to schema; no callers) | Delete |
| `apps/evals/pipeline/practice_eval/eval_practice.py` | Remove dead `exercise_data = []` placeholder (line 244) and its single checkpoint/serialization reference | Modify |
| `justfile` | Add `exercise-routing-eval` and `exercise-routing-ratchet` recipes | Modify |

---

## Open Questions

- Q: Should `scripts/drive_local_session.py` be refactored to a thin CLI wrapper over `shared/local_session.py`?
  Default: Yes, but only the eval-relevant parts are extracted in this plan. The script's existing CLI surface is preserved unchanged. No behaviour is removed.

- Q: What floor thresholds to commit in the initial `baseline.json`?
  Default: Bootstrapped conservatively — `invocation_rate > 0.0`, `kind_correctness > 0.50`, `dimension_match > 0.70`, `bar_range_grounding > 0.50` (or n=0 skip), `tempo_sanity > 0.90`. First real run will ratchet upward.
