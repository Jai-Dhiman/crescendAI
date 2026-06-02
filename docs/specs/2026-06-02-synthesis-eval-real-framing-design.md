# Synthesis Eval Real-Framing Re-Baseline Design

**Goal:** Re-measure the teacher-synthesis ASCF baseline through the real production framing (`buildSynthesisFraming`, driven by the `SessionBrain` Durable Object) instead of a thin Python strawman, and delete the divergent Python framing ports so a single source of truth remains.
**Not in scope:** longitudinal/trends digest + multi-session fixtures; bar_analysis enrichment (slice 1); score-conditioning (model epic); agentic tools / multi-turn loop; changing any production code in `apps/api/`.

## Problem

The locked Sonnet ASCF baseline (`Audible-Specific Corrective Feedback`, `mean_outcome` = 1.387, held in `apps/evals/teacher_model/stage0/tests/test_aggregator.py:17`) was measured against `build_synthesis_user_msg` in `apps/evals/teaching_knowledge/run_eval.py:120-214` — a hand-ported Python reconstruction of the prompt. Production synthesis is built by `buildSynthesisFraming` in `apps/api/src/services/prompts.ts:109-164`, called from `synthesize` in `apps/api/src/services/teacher.ts:639-721`, which the DO invokes in `apps/api/src/do/session-brain.ts:1675`. The two have drifted:

- The Python port injects a `bar_analysis` block onto top moments (`run_eval.py:152-165`) that production cold-start sessions never produce.
- The Python port assembles `top_moments` from global `SCALER_MEAN` deviations; production assembles them from the live accumulator (`session-brain.ts:1410-1421`) with per-student baselines.
- The Python port omits the `<student_memory>` block that production renders (`prompts.ts:151-156`).

The comment at `session-brain.ts:1521` claiming the DO legacy path produced the locked baseline is misleading: the locked number came from the Python strawman, not the DO. The baseline is therefore not an honest measurement of what production ships.

Three more framing duplicates compound the drift and must be deleted:
- `apps/evals/teaching_knowledge/bar_analysis_local.py` (re-implements `analyzeTier1/2`).
- `apps/evals/teacher_model/stage0/run_synthesis.py` (a second holdout runner built on `build_synthesis_user_msg`).
- `apps/evals/teaching_knowledge/piece_score_map.py` usage inside `build_synthesis_user_msg` (the bar-analysis score lookup).

## Solution (from the user's perspective)

A maintainer runs the baseline runner with `wrangler dev` up. The runner drives each holdout recording through the real DO (`?eval=true` -> `set_piece` -> `eval_chunk` xN -> `end_session` -> capture `synthesis`), judges the captured synthesis text exactly as before, and writes `apps/evals/results/baseline_v2_do.jsonl`. Aggregating that file yields an honest per-dimension `mean_outcome`, and `_SONNET_BASELINE` in the aggregator test is re-locked to it. The thin-framing duplicates and their tests are gone; the only synthesis framing left is production's `buildSynthesisFraming`.

Success is **an honest ASCF measured through the DO path and locked as the new baseline + the drift source deleted** — NOT "ASCF went up." The number may land below 1.387.

## Design

**Approach B (gut-and-repoint).** Reuse the existing DO-replay driver `run_recording` in `apps/evals/shared/pipeline_client.py:105-270` (already proven by `apps/evals/pipeline/practice_eval/eval_practice.py:307`). Add a DO-path baseline runner inside `run_eval.py` that:

1. Loads holdout rows (`recording_id`, `composer`, `title`, `skill_bucket`, `piece_slug`, `briefing_path`) from `apps/evals/teacher_model/stage0/data/stage0_holdout.jsonl`.
2. Loads the inference cache (`briefing_path` JSON: `recording_id` + `chunks`) for each row.
3. Calls `run_recording(wrangler_url, recording_cache, student_id, piece_query)` to get a `SessionResult` whose `synthesis.text` came from the real `buildSynthesisFraming`.
4. Judges `synthesis.text` with the existing `judge_synthesis_v2` wiring, stamps provenance (`make_run_provenance`), and writes a JSONL row in the schema the aggregator reads (`judge_dimensions[].{criterion,outcome,...}`, `error`).
5. Aggregates with the existing `scripts/aggregate.py` to produce `baseline_v2_do_aggregate.json`, then re-locks `_SONNET_BASELINE`.

**Cold-start fidelity (B1, no fabrication).** The eval student has no prior history, so the DO's longitudinal queries (`session-brain.ts:1457-1501`) return `[]` and `<student_memory>` renders empty. This is a real first-session production state; the runner fabricates nothing.

**Judge context comes from the holdout row, not `eval_context`.** The synthesis WS `eval_context` payload (`session-brain.ts:1386-1396`) does not include `piece_context`; the judge's `piece_name`/`composer`/`skill_level` are read from the holdout manifest row, matching the old runner (`run_eval.py:407-411`).

**`piece_query` resolution.** Each holdout row's `piece_slug` is passed as the `piece_query` to `set_piece`. If the DO cannot resolve it (no `piece_identified`), the session proceeds as Tier-2/3 with `bar_range: null` — exactly what production does for an unrecognized piece. The runner records a provenance flag (`piece_resolved: bool`) and never pretends Tier-1.

**Why not patch `build_synthesis_user_msg` to match production?** That keeps two implementations of the same prompt in sync forever — the exact drift we are deleting. Gut-and-repoint removes the second implementation entirely.

**Trade-off accepted.** The DO path requires a live `wrangler dev` + local DB, so the end-to-end runner cannot be unit-tested without a real DO. We mitigate by extracting a pure, fixture-testable row-builder (`build_do_row`) that the integration runner calls, and unit-testing that through its public interface. Only the thin orchestration shell touches the live DO; it is verified by a documented manual smoke step.

**`cli.py synthesis` repoint.** `apps/evals/teacher_model/stage0/cli.py:134-147` invokes the deleted `run_synthesis`. Its `synthesis` subcommand is repointed to the new DO-path runner so the stage0 dossier flow keeps working on the real framing.

## Modules

### `build_do_row` (new, in `run_eval.py`) — DEEP
- **Interface:** `build_do_row(session_result, holdout_meta, judge_fn, provenance, *, dry_run=False) -> dict`. Given a `SessionResult` (or its synthesis text + errors), the holdout meta, and a judge callable, returns one aggregator-schema JSONL row.
- **Hides:** error mapping (WS/DO failure -> `error` field, never a thin-framing fallback), judge invocation + dimension flattening, provenance stamping, the `piece_resolved` flag, and the empty-synthesis guard.
- **Tested through:** public interface with a fake `SessionResult` and a fake judge — no live DO, no internal-state assertions.

### `run_do_baseline` (new, in `run_eval.py`) — DEEP
- **Interface:** `run_do_baseline(holdout_path, out_path, wrangler_url, judge_fn, *, limit=None, dry_run=False) -> None`. Drives the full holdout through the DO and writes JSONL.
- **Hides:** holdout iteration, inference-cache loading from `briefing_path`, resume-by-`recording_id`, `asyncio.run(run_recording(...))` orchestration, and per-row `build_do_row` calls.
- **Tested through:** documented manual smoke (`--limit 1` against live `wrangler dev`). Its pure logic lives in `build_do_row`, which is unit-tested.

### `run_recording` (reused, unchanged) — DEEP
- Already the single DO-replay driver. No change.

## Verification Architecture

- **Canonical success state:** `apps/evals/results/baseline_v2_do.jsonl` contains one row per processed holdout recording, each with non-empty `synthesis_text` and populated `judge_dimensions` (or a non-empty `error`); `baseline_v2_do_aggregate.json` reports a per-dimension `mean_outcome`; `_SONNET_BASELINE` in `test_aggregator.py` equals that aggregate.
- **Automated check (unit, no DO):** `uv run pytest apps/evals/teaching_knowledge/tests/test_do_row.py` — `build_do_row` produces a valid aggregator row from a fixture `SessionResult`, maps a WS error to `error`, and never emits thin-framing fields. Plus `uv run pytest apps/evals/teacher_model/stage0/tests/test_aggregator.py` passing against the re-locked baseline.
- **Manual check (integration, needs DO):** with `just api` running, `uv run python -m teaching_knowledge.run_eval --do-path --limit 1` prints `synthesis (N chars)` with N > 0 and writes a row whose `synthesis_latency_ms` > 500 (a real Anthropic call, not a short-circuit).
- **Harness:** No reference-implementation harness is buildable (the "reference" IS the live DO). Task Group A builds the fixture-testable `build_do_row` first; the live-DO measurement is a manual run documented in `EVAL_CHECKLIST.md`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/teaching_knowledge/run_eval.py` | Delete `build_synthesis_user_msg` body + its `bar_analysis_local`/`piece_score_map` imports; add `build_do_row` + `run_do_baseline` + `--do-path` CLI flag | Modify |
| `apps/evals/teaching_knowledge/tests/test_do_row.py` | Unit tests for `build_do_row` | New |
| `apps/evals/teaching_knowledge/bar_analysis_local.py` | Delete | Delete |
| `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py` | Delete | Delete |
| `apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py` | Delete | Delete |
| `apps/evals/teaching_knowledge/test_run_eval_blocks.py` | Delete | Delete |
| `apps/evals/tests/test_run_eval_style_injection.py` | Delete | Delete |
| `apps/evals/teacher_model/stage0/run_synthesis.py` | Delete | Delete |
| `apps/evals/teacher_model/stage0/tests/test_run_synthesis.py` | Delete | Delete |
| `apps/evals/teacher_model/stage0/cli.py` | Repoint `synthesis` subcommand to `run_do_baseline` | Modify |
| `apps/evals/teacher_model/stage0/tests/test_aggregator.py` | Re-lock `_SONNET_BASELINE` to the DO-path aggregate | Modify |
| `apps/evals/results/baseline_v2_do.jsonl` | DO-path holdout run output | New (generated) |
| `apps/evals/results/baseline_v2_do_aggregate.json` | Aggregate of the above | New (generated) |
| `apps/evals/EVAL_CHECKLIST.md` | Document the DO-path baseline runbook; remove thin-framing entrypoint | Modify |

## Open Questions

- Q: Should `cli.py`'s `synthesis` subcommand be repointed to the DO-path runner, or removed entirely?
  Default (autopilot): **repoint** it to `run_do_baseline` so the stage0 dossier flow keeps producing `synthesis_runs.jsonl` on real framing. Keeps the dossier pipeline intact with the smallest blast radius.
- Q: The legacy synthesis WS payload uses `isFallback` (camelCase) but `pipeline_client.py` reads `is_fallback` (snake_case), so `is_fallback` always reads `False` on the legacy path.
  Default: **out of scope** — for cold-start holdout runs a real synthesis is always expected, so a false `is_fallback=False` does not corrupt the baseline. Note it in the plan; do not fix in this issue.
- Q: Does the live-DO holdout run land above or below the locked 1.387?
  Default: **unknown and not a gate.** Whatever it measures is locked verbatim; success is the honest measurement + deletion, not a direction.
- Q: After `build_synthesis_user_msg` is deleted, is there any non-DO synthesis path left in `run_eval.py`?
  Default: **no.** The thin-framing path is removed entirely; `run_do_baseline` is the sole synthesis source. The `--do-path` flag exists only so the new runner is reachable from the existing `main()` without changing the legacy judge-only/dry-run argument surface; it is the default and only behavior for producing synthesis.
