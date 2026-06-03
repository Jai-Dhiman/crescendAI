# Cold-Start (No-Baseline) Synthesis Design

**Goal:** A student with zero prior observations gets a synthesis grounded in what actually happened this session (real teaching moments + a real duration), instead of the teacher saying "I don't see any practice data."
**Not in scope:**
- The live mid-session observation path (`session-brain.ts` ~line 1267 gate). Live cold-start observations are a separate follow-up issue.
- Population/cohort reference. Reference is strictly within-session mean (option A) until a player cohort exists.
- Any change to the returning-student (`baselines !== null`) path. That path must be provably unchanged.
- The v6 harness synthesis path. v6 is gated off for eval and is not the path the cold-start bug hits in production first-sessions; this work targets the inputs (`topMoments`, `sessionDurationMs`, framing) that both the legacy and v6 paths read, but the only behavioral wiring change is in `runSynthesisAndPersist` ahead of both branches.

## Problem

`loadBaselinesFromDb` (`apps/api/src/services/synthesis.ts:84`) returns `null` when a student has zero observations in the last 30 days. In `runSynthesisAndPersist` (`apps/api/src/do/session-brain.ts:1362`), `acc.topMoments()` is empty for a first session because the live teaching-moment gate (`shouldAttemptMoment && state.baselines !== null`, line 1267) never fires when baselines are null, so no moments were ever accumulated. `buildSynthesisFraming` (`apps/api/src/services/prompts.ts:109`) therefore surfaces an empty `top_moments` array, and the teacher LLM produces "I don't see any practice data." This hits real production first-sessions, not just the eval harness.

Separately, `sessionDurationMs` (`session-brain.ts:1372`) is derived from `acc.timeline` `Date.now()` stamps (first vs last event). In eval replay all timeline events are stamped within the same millisecond, so `duration_minutes` rounds to 0 in `buildSynthesisFraming` (`prompts.ts:121`), and the teacher is told the session was 0 minutes long.

## Solution (from the user's perspective)

A first-time student finishes a practice session. The teacher's synthesis names what happened *within this session* -- e.g. "your pedaling was the weakest part of this run compared with the rest of the session, while your phrasing held up well" -- and references a real session length (e.g. "in this ~2 minute session"). The teacher never claims improvement over past sessions or references history the student does not have.

## Design

Four vertical slices, each one test -> one impl -> one commit.

1. **WASM additive function.** Add a pure Rust `select_session_moments(chunks, reference, max) -> Vec<TeachingMoment>` to `teaching_moments.rs`, reusing the existing worst-dimension / magnitude-ranking / positive-fallback logic but ranked against a caller-supplied within-session `reference` (the per-dim session mean) instead of a stored baseline. Reasoning strings are within-session phrased ("weakest relative to the rest of this session"), never longitudinal. Returns up to `max` moments spanning distinct dimensions. Returns empty when `< 2` chunks. The existing `select_teaching_moment` (live path) is untouched. Export via `wasm-bindgen` in `lib.rs`, wrap in `wasm-bridge.ts` as `selectSessionMoments`, rebuild `pkg/`.

   *Why a new function, not a parameter on the existing one:* the live path's contract (top-1, dedup against recent observations, baseline-relative) differs from synthesis cold-start (up-to-max, distinct dimensions, session-mean-relative). Overloading `select_teaching_moment` would couple two unrelated callers and risk regressing the live path. Additive is the surgical choice.

2. **Duration fix.** Replace the timeline-stamp duration calc in `runSynthesisAndPersist` with `scoredChunkCount * 15000` (MuQ chunk = 15s). Extract the arithmetic into an exported pure function `computeSessionDurationMs(scoredChunkCount)` so it is testable through a public interface (mirrors the existing `nextSynthesisAlarmDelayMs` extraction pattern). Applies to both prod and eval, removes a `Date.now()` dependency.

3. **DO wiring.** In `runSynthesisAndPersist`, when `state.baselines === null`, compute the per-dim session mean across `state.scoredChunks`, call the bridge, and push the resulting moments into `acc.teachingMoments` so the existing `acc.topMoments()` picks them up. Set a `referenceMode` value of `"within_session"` to thread downstream. The mean computation + mapping is extracted into an exported pure function `buildColdStartMoments(scoredChunks, max)` returning `AccumulatedMoment[]`, testable with real WASM through its public interface. When `baselines !== null`, this branch is never entered (regression-lock).

4. **Fabrication guardrail.** Add an optional `referenceMode` param to `buildSynthesisFraming`, emit a `reference_mode` field in the `session_data` block, and when `referenceMode === "within_session"` append the instruction: "This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time." Thread `referenceMode` through `SynthesisInput` and `synthesize` (`teacher.ts`).

### Error handling (explicit, no silent fallbacks)

- WASM unavailable at synthesis -> `buildColdStartMoments` throws; the DO call site catches, logs structured JSON, and leaves `acc.teachingMoments` empty. Synthesis still runs on duration + practice_pattern (degraded, not crash). This matches the existing live-path `try { ... } catch {}` convention around `wasm.selectTeachingMoment`.
- `scoredChunks.length < 2` -> `select_session_moments` returns empty; `buildColdStartMoments` returns `[]`. No moments, honest.
- `baselines !== null` -> cold-start branch never entered; returning-student path byte-for-byte unchanged.

## Modules

**`select_session_moments` (Rust, in `teaching_moments.rs`)**
- Interface: `select_session_moments(chunks: &[ScoredChunk], reference: &[f64; 6], max: usize) -> Vec<TeachingMoment>`
- Hides: worst-dimension detection vs the within-session reference, magnitude ranking, distinct-dimension selection up to `max`, positive-fallback when all chunks sit at the mean, the `< 2` chunk guard, and within-session reasoning string construction.
- Tested through: `cargo test` unit tests calling the pure function (no `JsValue`), asserting ranking, distinct dimensions, positive fallback, and empty-on-`<2`.
- Depth: DEEP -- one call hides all session-relative selection logic.

**`selectSessionMoments` (TS bridge, in `wasm-bridge.ts`)**
- Interface: `selectSessionMoments(chunks: ScoredChunk[], reference: StudentBaselines, max: number): TeachingMoment[]`
- Hides: the `serde_wasm_bindgen` boundary and pkg import.
- Tested through: real-WASM workerd test asserting it returns real computed moments.
- Depth: SHALLOW by design -- it is a forwarding wrapper, consistent with every other function in `wasm-bridge.ts`. Justified: the bridge's single responsibility is "be the only importer of `pkg/`"; depth lives in the Rust crate behind it.

**`computeSessionDurationMs` (TS, exported from `session-brain.ts`)**
- Interface: `computeSessionDurationMs(scoredChunkCount: number): number`
- Hides: the MuQ chunk-length constant (15000ms) and the choice to derive duration from chunk count rather than wall-clock timeline stamps.
- Tested through: direct call asserting `> 0` for a 10-chunk session.
- Depth: SHALLOW but warranted -- it exists to make the duration contract testable and to centralize the 15s constant; mirrors the existing `nextSynthesisAlarmDelayMs` seam.

**`buildColdStartMoments` (TS, exported from `session-brain.ts`)**
- Interface: `buildColdStartMoments(scoredChunks: { chunkIndex: number; scores: number[] }[], max: number): AccumulatedMoment[]`
- Hides: per-dim session-mean computation, the `ScoredChunk` shaping for the bridge, the bridge call, and mapping `TeachingMoment[]` -> `AccumulatedMoment[]` (filling `baseline` with the session mean, `analysisTier: 3`, `barRange: null`, `llmAnalysis: null`).
- Tested through: real-WASM workerd test asserting non-empty distinct-dimension moments for a multi-chunk session and `[]` for a single-chunk session.
- Depth: DEEP -- one call hides mean computation + WASM marshalling + accumulator mapping.

**`buildSynthesisFraming` (TS, modified, in `prompts.ts`)**
- Interface: gains a trailing optional `referenceMode?: "within_session" | null` param (8th positional, default `null` -> existing callers unchanged).
- Hides: the conditional `reference_mode` field and the no-history instruction.
- Tested through: existing prompts.test.ts pattern -- assert the field + instruction appear iff `within_session`, omitted otherwise.
- Depth: DEEP (already a deep prompt-builder; the addition is conditional output behind the same interface).

## Verification Architecture

- **Canonical success state:**
  - `select_session_moments` cargo tests pass (ranking, distinct dims, positive fallback, empty-on-`<2`).
  - `computeSessionDurationMs(10) === 150000` (and `buildSynthesisFraming` with that duration emits `duration_minutes: 3`, i.e. `> 0`).
  - `buildColdStartMoments` returns a non-empty `AccumulatedMoment[]` with distinct dimensions for a 6-chunk session and `[]` for a 1-chunk session, under real WASM.
  - `buildSynthesisFraming(..., "within_session")` contains `"reference_mode"` and the no-history instruction; with `null`/omitted it contains neither.
- **Automated check:**
  - Rust: `cd apps/api/src/wasm/score-analysis && cargo test select_session_moments`
  - TS: `cd apps/api && bun run test -- <file>` (vitest, `@cloudflare/vitest-pool-workers`)
- **Harness:** No separate Task Group 0 harness is buildable beyond the per-slice tests -- each slice's test *is* the verification, and slices 1->3 form a dependency chain (bridge before DO wiring). The end-to-end "first session yields non-empty top_moments" claim is verified at the `buildColdStartMoments` public interface (Slice 3) rather than by booting a full DO, because the DO requires a workers binding + storage fixtures that would test the harness, not the behavior. The DO call-site wiring (null-branch only) is verified by reading the diff in `/review`, not by an automated DO boot test.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/wasm/score-analysis/src/teaching_moments.rs` | Add pure `select_session_moments` + cargo unit tests | Modify |
| `apps/api/src/wasm/score-analysis/src/lib.rs` | Add `#[wasm_bindgen]` export forwarding to `select_session_moments` | Modify |
| `apps/api/src/wasm/score-analysis/pkg/*` | Rebuilt by `wasm-pack` (build artifact) | Modify |
| `apps/api/src/services/wasm-bridge.ts` | Add `selectSessionMoments` wrapper | Modify |
| `apps/api/src/services/wasm-bridge.workerd.test.ts` | Real-WASM test for `selectSessionMoments` | Modify |
| `apps/api/src/do/session-brain.ts` | Export `computeSessionDurationMs`, `buildColdStartMoments`; wire null-baseline branch + duration fix in `runSynthesisAndPersist`; thread `referenceMode` into `SynthesisInput` | Modify |
| `apps/api/src/do/session-brain.unit.test.ts` | Unit tests for `computeSessionDurationMs` and `buildColdStartMoments` | Modify |
| `apps/api/src/services/prompts.ts` | Add `referenceMode` param + conditional `reference_mode` field + no-history instruction | Modify |
| `apps/api/src/services/prompts.test.ts` | Tests for conditional reference_mode output | Modify |
| `apps/api/src/services/teacher.ts` | Add `referenceMode` to `SynthesisInput`; pass to `buildSynthesisFraming` | Modify |

## Open Questions

- Q: Should `max` cold-start moments default to the same cap (8) as `acc.topMoments()`?  Default: pass `max = 6` (one per dimension) from the DO; `topMoments()` still caps the merged set at 8, so 6 is safe and keeps distinct dimensions.
- Q: For `AccumulatedMoment.baseline` on cold-start moments, what value?  Default: the per-dim session mean (the reference the deviation was computed against), so downstream `deviation = score - baseline` stays consistent.
