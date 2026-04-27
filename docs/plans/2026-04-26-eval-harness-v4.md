# Eval Harness v4 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task within a group, sequentially across groups).
> Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Add three eval-harness capabilities — top-2 cluster-based teacher-voice injection from a single source of truth, a 4-condition signal ablation that tests whether MuQ + AMT signals are load-bearing in synthesis, and a binary 8x5 atomic-skill rubric matrix that decomposes synthesis failures.

**Spec:** `docs/specs/2026-04-26-eval-harness-v4-design.md`

**Style:** Follow project standards (`CLAUDE.md`, `apps/api/TS_STYLE.md`). Use `uv` for Python, `bun` for JS, no emojis, explicit exception handling.

## Task Groups

```
Group A (parallel): Task 1, Task 2, Task 3, Task 4, Task 5
Group B (depends on Group A):
  Track Py (sequential): Task 6 -> 7 -> 8 -> 9 -> 10  (all touch teacher_style.py)
  Track Ts (sequential): Task 11 -> 12 -> [Task 16 must complete] -> 13 -> 14 -> 15  (all touch teacher_style.ts; Task 13 requires playbook.json from Task 16)
  Track Compile: Task 16 (depends on Task 1)
  Track Ablation: Task 17, 18, 19 (depend on Tasks 2-4)
  Track Atomic: Task 20 -> 21 (depends on Task 5)
Group C (depends on Group B):
  Task 22 (prompts.ts; depends on Task 15)
  Task 23 (run_eval.py cluster injection; depends on Task 10)
  Task 24 (run_eval.py atomic gate; depends on Task 23 + Task 20)
  Task 25 (justfile recipes; depends on Task 16)
```

The full per-task body (test code, exact failing-error expectations, exact implementation code, commit messages) is committed in five appendix files to keep this index readable; each appendix is self-contained for its task range so the build agent dispatches one subagent per task with the corresponding appendix file as context.

- `docs/plans/2026-04-26-eval-harness-v4/tasks-01-05.md` (Group A, 5 tasks)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-06-10.md` (Python teacher_style track)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-11-15.md` (TS teacher_style track)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-16.md` (compile script)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-17-21.md` (ablation + atomic judge)
- `docs/plans/2026-04-26-eval-harness-v4/tasks-22-25.md` (integrations + just recipes)

Each appendix follows the strict task shape:

```
### Task N: <name>
Group: <A|B|C>
Behavior being verified: <one sentence>
Interface under test: <public API>
Files: Create / Modify / Test
Step 1: failing test (exact code)
Step 2: run command + expected exact failure error
Step 3: minimum implementation (exact code)
Step 4: run command + expected pass
Step 5: commit (exact files + exact message)
```

## Plan self-review notes

- Spec coverage: Tasks 1 + 6-15 cover sub-goal 1; Tasks 2-4 + 17-19 cover sub-goal 2; Tasks 5 + 20-21 + 24 cover sub-goal 3; Tasks 16 + 25 cover infra; Tasks 22-23 cover integration.
- Vertical-slice check: every task has exactly one test + one implementation + one commit.
- Behavior-test check: all tests assert on public function returns or file outputs, not internal state.
- No mocking of internal collaborators: Tasks 17 and 20 use Protocol-typed fakes for the LLM client, which is a system boundary.
- Group correctness: tasks sharing a file are sequential within their track; cross-track parallelism preserved.
- No placeholders: every step in every appendix has exact code or commands.

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

**Right problem?** Yes. The spec identifies three concrete gaps with real production consequences:
1. Cluster-blind teacher voice means `playbook.yaml`'s richer content is invisible to both prod synthesis and evals — editing the YAML has zero effect.
2. Unvalidated signal utility is a strategic risk: MuQ/AMT infrastructure cost is hard to justify without evidence signals change outputs.
3. 7-dim composite gives a single number on failure — locating which pedagogical move failed requires inspecting the raw text manually.

**Direct path?** Yes. Each sub-goal directly addresses one gap with the minimal machinery required.

**Existing coverage?** `apps/api/src/services/prompts.ts:buildSynthesisFraming` already injects `<style_guidance>` (era-level weights from `style-rules.json`). The plan correctly extends this existing pattern rather than replacing it. `apps/evals/teaching_knowledge/run_eval.py:build_synthesis_user_msg` mirrors it. Both are exact insertion points.

#### 2. Scope Check

The compile-script + committed JSON artifact pattern adds a build step (infra). It's necessary: the TS side cannot YAML-parse at runtime in Workers. This scope is load-bearing.

The hand-written DSL evaluator in both Python and TypeScript is the highest-complexity addition. The spec justifies it (no dynamic code execution, no security surface, TS parity). Acceptable, but adds ~200 LOC of parser to each language.

**File count:** 27 files touched/created. Exceeds the 8-file complexity smell threshold, but the count includes 12 test files (unavoidable) and 5 JSON/YAML content files (no logic). Net logic files: ~10. Acceptable.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                          THIS PLAN                           12-MONTH IDEAL
─────────────────────────────────────  ──────────────────────────────────  ────────────────────────────────
playbook.yaml unreferenced in prod     single source wired to both prod    finetuned Qwen teacher replaces
                                       and eval via compile step           current synthesis; playbook still
                                                                           drives cluster selection

signals passed through but not         ablation produces load-bearing      model v2 training decision gated
tested for utility                     verdict before more investment      on load-bearing confirmation

7-dim composite: single scalar         atomic matrix: surgical failure     training rubrics derived from
failure diagnosis                      mode localization                   validated atomic matrix
```

This plan moves directly toward the 12-month ideal. No tech debt created.

#### 4. Alternatives Check

The spec documents the reasoning for: (a) triggers in YAML alongside prose (vs separate file), (b) top-2 cluster cardinality, (c) diagnostic-only atomic matrix, (d) battery corruption, (e) embedding + judge score instead of pairwise judging. All key decisions are justified.

---

### Engineering Pass

#### 5. Architecture

**Sub-goal 1 data flow:**
```
shared/teacher-style/playbook.yaml
        │
        ├── compile_playbook.py ──→ apps/api/src/lib/playbook.json
        │                                    │
        │                          teacher_style.ts::selectClusters
        │                                    │
        │                          prompts.ts::buildSynthesisFraming → LLM
        │
        └── teacher_style.py::select_clusters
                    │
           run_eval.py::build_synthesis_user_msg → LLM
```

**Sub-goal 2 data flow:**
```
holdout sessions (20)
        │
        └── run_ablation.py (4 conditions × 20 = 80 rows)
                │
                ├── synthesis_client.complete() → synthesis_text  ✓
                ├── [MISSING: judge_synthesis_v2 call]            ← BLOCKER
                │
                └── analyze.py → Δ_score → decide_verdict()
                    (no judge scores to compute from)
```

**Security:** No new user input flows to SQL, shell, or LLM without transformation. The DSL evaluator explicitly rejects unknown signal names via allowlist. No dynamic code execution used. Clean.

#### 6. Module Depth Audit

| Module | Interface size | Implementation size | Verdict |
|--------|---------------|---------------------|---------|
| `teacher_style.py` | 3 public fns + 2 dataclasses | ~250 LOC DSL evaluator + cluster logic | DEEP |
| `teacher_style.ts` | 4 exported fns + 2 types | ~250 LOC DSL evaluator + cluster logic | DEEP |
| `corrupt_signals.py` | 1 public fn (`corrupt`) | 3 branches, seeded RNG, distributional sampling | DEEP |
| `run_ablation.py` | 1 public fn (`run_ablation`) | orchestration, resume-safety, condition loop | DEEP |
| `analyze.py` | 2 public fns | model loading, embedding, decision rule | DEEP |
| `judge_atomic.py` | 1 public fn | prompt construction, provider call, parse, validate | DEEP |
| `compile_playbook.py` | CLI (1 mode flag) | YAML-to-JSON serialization, drift check | DEEP |

All modules are deep. No shallow modules found.

#### 7. Code Quality

**No catch-all exception handling** in new code. `judge_atomic.py` wraps `json.loads` in a specific `json.JSONDecodeError` catch (Task 21). Explicit.

**Edge cases:**
- `corrupt(..., mode="shuffle", all_top_moments=[single_session])` → raises `ValueError("cannot shuffle")`. Tested.
- `decide_verdict` with missing keys → `.get("flip", 0.0)` default. Safe.
- `_maybe_atomic_judge` with no numeric scores → returns `None`. Safe.
- `_first_exemplar` with no `good_examples` → returns `""` (empty exemplar silently omitted). Safe.

**Potential issue (OBS):** `_marginal` in `corrupt_signals.py` computes `deviation_from_mean` as `new_score - 0.5` (hardcoded baseline), ignoring the per-dimension `SCALER_MEAN` (which varies 0.459–0.545). Marginal-corrupted moments will have inaccurate deviation annotations. Not a correctness blocker for the ablation goal but worth noting.

#### 8. Test Philosophy Audit

All tests exercise public function returns or file outputs. Protocol-typed fakes (Tasks 17, 20) stand in for LLM system boundaries only. No internal collaborators mocked. No shape-only tests. No private method calls. Philosophy adherent throughout.

#### 9. Vertical Slice Audit

Every task: one failing test → one implementation → one commit. No horizontal slicing. Tasks 2/3/4 each add one branch of `corrupt()` sequentially — correct.

**Minor exception:** Task 24 adds `_maybe_atomic_judge` helper AND wires it into `run()` in the same commit. Two behaviors in one commit — [RISK].

#### 10. Test Coverage Gaps

```
[+] teacher_style.py::select_clusters()
    ├── [TESTED]  negative deviation → Technical — Task 8 ★★
    ├── [TESTED]  positive deviation → Praise — Task 8 ★★
    ├── [TESTED]  two distinct clusters returned — Task 8 ★★
    └── [GAP]     fallback fixture values produce wrong clusters (see BLOCKER 3)

[+] run_ablation.py
    ├── [TESTED]  emits 4 rows with correct condition labels — Task 17 ★★
    ├── [TESTED]  resume-safety — Task 17 ★★
    └── [GAP]     judge invocation path (skip_judge=False) never tested;
                  no task adds it — delta computation is untestable (BLOCKER 2)

[+] judge_atomic.py
    ├── [TESTED]  happy-path parse — Task 20 ★★
    ├── [TESTED]  synthesis text in user msg — Task 20 ★★
    ├── [TESTED]  non-JSON raises ValueError — Task 21 ★★★
    └── [TESTED]  missing moves key raises ValueError — Task 21 ★★★
```

#### 11. Failure Modes

- **`run_ablation.py` mid-run crash:** JSONL append + resume-safety reads completed `(recording_id, condition)` pairs. No corrupt state.
- **`judge_atomic.py` malformed response:** raises `ValueError`, propagates through `run_eval.py`'s `except Exception` handler, written as error row. Session not lost. No silent failure.
- **`sentence-transformers` model download unavailable:** `cosine_similarity` raises on first call; `decide_verdict` is blocked. Verdict computation fails loudly, not silently.
- **Compiled `playbook.json` stale at deploy:** `just check-playbook-sync` in CI prevents merging with drift. Acceptable.

#### 12. Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `parents[2]` from `apps/evals/shared/teacher_style.py` reaches repo root | **RISKY** | `parents[0]`=`apps/evals/shared`, `parents[1]`=`apps/evals`, `parents[2]`=`apps`. Must be `parents[3]`. Verified by path tracing. |
| Fixture `drilling_improvement`: secondary = "Technical" | **RISKY** | By formula: Artifact=0.5 > Technical=-0.125. Secondary is "Artifact". Computed mathematically. |
| Fixture `all_signals_low_fallback`: fallback fires | **RISKY** | Positive-encouragement scores 0.5 (> CONFIDENCE_FLOOR 0.3) because `max_neg_dev < 0.1` is True. Fallback does not fire. Actual primary=Positive, secondary=Motivational. |
| `run_ablation.py` produces judge scores for `analyze.py` | **RISKY** | Task 17 only calls synthesis; JSONL rows have no `judge_dimensions`. No task adds judge invocation. `Δ_score` is uncomputable. |
| Task 13 can run in parallel with Task 16 | **VALIDATE** | Task 13 imports `playbook.json` which only exists after Task 16 completes. Must be sequenced, not parallelized. |
| `playbook.yaml` move doesn't break other importers | **SAFE** | `grep` found only `synthesize_playbook.py` (CLI output default, not import) and `derive_rubrics.py` (CLI arg, not hardcoded path). Move is safe. |
| `uv run --with pyyaml --with pytest` works from `shared/teacher-style/` | **VALIDATE** | `shared/` has no `pyproject.toml`. Verify `uv run` works from a directory without a project file or add a minimal one. |
| Dual-judge ablation (Gemma-4 + GPT-5.4-mini) is implemented in tasks 17-21 | **RISKY** | `run_ablation.py` accepts one `synthesis_client`; no dual-judge orchestration is implemented. Verdict rule uses `min(Δ across judges)` but only one judge would run. |

---

### Summary

[BLOCKER] count: 3
[RISK]    count: 3
[QUESTION] count: 1

**[BLOCKER] (confidence: 9/10)** — `parents[2]` off-by-one in Task 8's `teacher_style.py`. Path resolves to `apps/shared/teacher-style/playbook.yaml` (non-existent). Fix: use `parents[3]`. Every downstream Python task (8, 9, 10, 23) will fail with `FileNotFoundError` at call time, not the expected ImportError. Task 8's Step 2 "Expected: FAIL — ImportError" is also incorrect as a result.

**[BLOCKER] (confidence: 9/10)** — `run_ablation.py` (Task 17) never invokes a judge. JSONL rows contain `synthesis_text` but no `judge_dimensions`. `analyze.py` has no source for `Δ_score` deltas. No task in 17-21 adds judge invocation or a `compute_deltas(jsonl_path)` function. Ablation cannot produce its verdict. Fix: add judge call inside `run_ablation.py`'s condition loop when `skip_judge=False`, persisting judge scores into the JSONL row; add a `compute_deltas(jsonl_path) -> dict` function in `analyze.py`; add a task between 17 and 18 or extend Task 17.

**[BLOCKER] (confidence: 9/10)** — Two fixture bugs in `test_fixtures.json` (Task 10). (a) `drilling_improvement`: `expected_secondary_substring` must be `"Artifact"` (Artifact=0.5 > Technical=-0.125). (b) `all_signals_low_fallback`: fallback does not fire (Positive scores 0.5 ≥ 0.3); expected primary/secondary should be `"Positive"` / `"Motivational"`, or the signal values must be changed so all clusters score < 0.3. Fix before building so Step 4 of Task 10 and Task 13's parity tests pass.

**[RISK] (confidence: 8/10)** — Task 13 dispatched in parallel with Task 16 will fail. `playbook.json` doesn't exist until Task 16 completes. Fix: add explicit sequencing in the plan index (Task 13 depends on Task 16, not just Task 12).

**[RISK] (confidence: 7/10)** — `shared/teacher-style/` has no `pyproject.toml`. Task 1's test invocation runs `uv run` from that directory. Verify this works, or add a minimal `pyproject.toml`.

**[RISK] (confidence: 7/10)** — Task 24 bundles two behaviors (new helper + wiring into `run()`) in one commit. Low severity but makes bisecting harder.

**[QUESTION]** — The spec says ablation uses dual-judge (Gemma-4 + GPT-5.4-mini) and verdict uses `min(Δ across judges)`. No dual-judge orchestration appears in any task. Is GPT-5.4-mini access configured in the eval environment, and is dual-judge intended for a follow-up task (post-build), or must it be part of this plan?

VERDICT: NEEDS_REWORK — resolve the three blockers above before dispatching the build agent.
