# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Build environment

- `/build` skill had no Task subagent dispatcher available, so the controller-as-implementer pattern was used: a single agent performed each task sequentially with strict TDD discipline. All commits follow the plan's "test fails → implement → test passes → commit" cycle.
- Worktree at `.worktrees/feat/bar-analysis-synthesis-context` was missing untracked artifacts (`model/data/{eval,midi,raw,embeddings,checkpoints,references,weights}`, `apps/api/.dev.vars`, `apps/api/src/wasm/{score-analysis,piece-identify}/pkg`, and the gitignored score JSONs in `model/data/scores/`). These were symlinked from the main worktree to enable the eval smoke + tests; no source files were modified.

## Task 0 — Baseline gate

- Smoke ran 10 recordings; ASCF outcome mean = 1.400 (band [1.237, 1.537]) — gate passes. Output at `apps/evals/results/baseline_smoke.jsonl` (not committed).
- Eval row `judge_dimensions` shape is a list of dicts (not the dict-of-dicts shape the plan suggested); extraction path uses `criterion == "Audible-Specific Corrective Feedback"` filter.

## Task 1c — Watch-it-fail observed

- Commented out filter/sort/slice in `buildBarAnalysisFacts`; 3 tests failed (cap, threshold, all-zero correlated). Restored, all 4 pass.

## Task 5 — piece_score_map mapping

ls-verified mappings (4 entries):
- `bach_prelude_c_wtc1` → `bach.prelude.bwv_846.json`
- `chopin_ballade_1` → `chopin.ballades.1.json`
- `pathetique_mvt2` → `beethoven.piano_sonatas.8-2.json`
- `chopin_etude_op10no4` → `chopin.etudes_op_10.4.json`

Intentionally unmapped (file absent or ambiguous): `moonlight_sonata_mvt1` (only 14-3 exists, not 14-1), `fantaisie_impromptu`, `fur_elise`, `rachmaninoff_prelude_csm` (4 preludes available — ambiguous), `chopin_waltz_csm`, `liszt_liebestraum_3`, `nocturne_op9no2`, `debussy_arabesque_1`, `bach_invention_1`, `clair_de_lune`, `mozart_k545_mvt1` (Sonata 16 absent from scores/), `schumann_traumerei`, `ensemble_4fold` (meta).

Tier-1 (mapped) coverage in the eval cache is therefore ~4/17 piece slugs.

## Task 9 — Articulation-test sensitivity

- The plan-specified test `assert "ratio" in text or "score" in text` matches the substring "ratio" against the Tier-2 articulation string "Mean note duration X.XXs." — and "duration" contains "ratio". So this test passes even when Tier-1 articulation enrichment is disabled. Watch-it-fail performed (disabled enrichment, observed all tier1 tests still passing as result — a known plan brittleness). Implementation kept correct per plan; test is preserved verbatim per plan.

## Task 10 — Plan test fixture inconsistency

- `test_correlated_includes_dimensions_above_threshold_cap_2` originally had `dynamics: 0.80, timing: 0.20` (both dev=0.30), expecting `correlated == ["dynamics", "pedaling"]`. But `select_worst_chunk` uses strict `>` comparison and dict iteration order; dynamics(0.30) gets selected first, so timing(0.30) doesn't replace it, making the selected dim `dynamics`, not `timing`. With `dynamics` selected, correlated becomes `["timing", "pedaling"]` — contradicting the test's expected output.
- Resolved by deviation: bumped timing in the fixture from 0.20 to 0.19 (dev=0.31, strictly worst), preserving plan intent (timing selected). Test now passes with plan's expected `["dynamics", "pedaling"]`. Documented inline in the test fixture.

## Task 3 — session-brain wiring

- Plan said "4 sites in handleProcessChunk, 4 sites in handleEvalChunk" for analyzeTier calls; actual count is 3 per handler (1 Tier1 + 2 Tier2 fallbacks). Loop-1 challenge review flagged this; I located all sites by grep and assigned `chunkAnalysis = analysis` immediately after every `wasm.analyzeTier1/2(...)` call.
- Added `ChunkAnalysis` to the named-type import block (already pulling other types from wasm-bridge); avoided a duplicate inline `import type` to keep imports tidy.
- Both `accMoment` build sites use `momentDim` and `baselines` from local scope (already available) — no scope hoisting needed beyond `chunkAnalysis`.
- Contract test in session-brain.unit.test.ts pins the shape, as plan instructs. The session-brain edit itself is covered by typecheck + Task 12 (eval) — no DO unit test that avoids WASM mocking is feasible per plan.

## Task 11 — Eval uses SCALER_MEAN baselines (documented divergence)

- Per plan, the eval mirror uses `SCALER_MEAN` (MuQ global mean) as the baselines passed to `build_bar_analysis`. Production uses per-student baselines computed in session-brain.ts. This is a known divergence; the loop-1 review flagged it as a "ride through" risk. Inline comment added in `build_synthesis_user_msg` calling this out.

## Task 12 — DEFERRED

Per scope override from the operator, Task 12 (the multi-hour full 513-recording eval lift verification) was NOT executed. It is a measurement gate, not an implementation task, and will be handled separately after /review by the operator. Branch is ready for /review independent of Task 12.
