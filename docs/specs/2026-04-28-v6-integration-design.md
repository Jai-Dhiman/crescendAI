# V6 Integration Design

**Goal:** Wire all 15 atoms into the OnSessionEnd compound binding, populate HookContext.digest with per-chunk signal data from DO storage, persist Phase 1 diagnosis artifacts to the database, and validate the full pipeline with an E2E integration test.

**Not in scope:**
- Molecules (Plan 3 — when it ships, it adds ALL_MOLECULES to the registry and extends the E2E test)
- Diagnosis artifact querying UI (frontend surface for past diagnoses)
- iOS path (web-only synthesis pipeline)
- Cohort table values derived from live data (static approximations from training distributions are used)

---

## Problem

Without this plan:

1. `compound-registry.ts` has `tools: []` — Phase 1 never dispatches any atom or molecule. The LLM sees an empty tool list and immediately ends its turn. No diagnoses are collected. Phase 2 synthesizes from topMoments/drillingRecords alone.
2. `HookContext.digest` is populated with `sessionDurationMs`, `practicePattern`, `topMoments`, `drillingRecords`, and `pieceMetadata` — but atoms that need raw per-chunk signal data (`extract-bar-range-signals`, `compute-pedal-overlap-ratio`, etc.) get nothing to work with. The digest lacks `chunks`, `baselines`, `session_history`, `past_diagnoses`, and `cohort_tables`.
3. Per-chunk `midi_notes`, `pedal_cc`, and `alignment` from AMT/WASM are discarded after `handleChunkReady` — there is no storage path for this data to reach synthesis time.
4. No `diagnosis_artifacts` DB table exists, so Phase 1 diagnoses are never persisted and atoms like `fetch-similar-past-observation` always receive empty history.
5. The only integration test (`integration.test.ts`) validates the skill markdown catalog structure — it does not exercise the harness loop end-to-end.

---

## Solution (from the user's perspective)

When `HARNESS_V6_ENABLED=true`, a session ending triggers the V6 synthesis path. Phase 1 receives the full digest — all 15 atoms are available as tools. The LLM dispatches atoms against per-chunk signal data in the digest. Phase 1 results validate as `DiagnosisArtifact` objects and are persisted to the database. Phase 2 synthesizes from real diagnoses into a `SynthesisArtifact` with non-trivial `focus_areas`. Subsequent sessions can reference past diagnoses for longitudinal context.

After this plan ships, `HARNESS_V6_ENABLED=true` is safe to set in production.

---

## Design

### Per-chunk enriched data storage

The DO's `handleChunkReady` discards `perfNotes`, `perfPedal`, and WASM alignment after each chunk. Storing all per-chunk raw data in the main `"state"` key risks the 128KB Cloudflare DO per-key limit (20+ chunks × ~5KB each = 100KB+). The chosen approach: store each chunk's enriched data in a separate DO storage key (`"chunk_enriched:${chunkIndex}"`), bulk-read at synthesis time via `ctx.storage.get([...keys])`, and delete at finalization.

Tradeoff accepted: N additional `ctx.storage.put()` calls per session (one per chunk, ~3–8KB each). These are writes, not reads on the hot path, and are non-fatal if they fail.

### Fat SynthesisInput, pure synthesizeV6

`runSynthesisAndPersist` assembles all data (enriched chunks, baselines from state, session history from DB, past diagnoses from DB) and passes it via `SynthesisInput` before calling `synthesizeV6`. `synthesizeV6` remains a pure reshaper — no I/O, no async calls — mapping `SynthesisInput` fields to `HookContext.digest` keys. This preserves testability of `synthesizeV6` without DB mocks.

### Cohort tables as inline constants

Cohort percentile tables for `fetch-reference-percentile` are static model artifacts derived from MuQ training distribution parameters (`SCALER_MEAN`, `SCALER_STD`). They change only when the model changes, which requires a redeploy anyway. Stored as a module-level `COHORT_TABLES` constant in `teacher.ts`.

### Diagnosis persistence via safeParse scan

After the event loop over `synthesizeV6`, `runSynthesisAndPersist` calls `persistDiagnosisArtifacts(db, phase1Results, sessionId, studentId, pieceId)`. This function iterates all Phase 1 `{ tool, output }` pairs, attempts `DiagnosisArtifactSchema.safeParse(output)` on each, and bulk-inserts valid ones into `diagnosis_artifacts`. No new `HookEvent` type is required. If the insert fails, it is logged as `level: "error"` and not rethrown — the synthesis artifact was already delivered.

### E2E test: real atom dispatch, mocked LLM

The integration test stubs `fetch` (via `vi.stubGlobal`) to return a canned Phase 1 LLM response that calls `extract-bar-range-signals` with fixture args drawn from the fixture digest. The atom's real `invoke` code runs. Phase 2 stub returns a `SynthesisArtifact` with one `focus_areas` entry. This proves ALL_ATOMS are wired into the registry, the digest shape matches what atoms expect, and the artifact validates.

When Plan 3 ships molecules, the E2E test should be extended to also dispatch one molecule and assert its `DiagnosisArtifact` output validates.

---

## Modules

### `toEnrichedChunk` (in `session-brain.ts`)
- **Interface:** `toEnrichedChunk(chunkIndex, muqScores, perfNotes, perfPedal, barMapAlignments, barCoverage): EnrichedChunk`
- **Hides:** Unit conversion (seconds → ms for onset, offset, pedal time), WASM `NoteAlignment[]` → atom `Alignment[]` reshaping, null-safe handling of missing alignment/bar coverage
- **Tested through:** Public function call with known inputs; assert output field values including `onset_ms` conversion

### `persistDiagnosisArtifacts` (in `services/synthesis.ts`)
- **Interface:** `persistDiagnosisArtifacts(db, phase1Results: Array<{ tool: string; output: unknown }>, sessionId, studentId, pieceId): Promise<void>`
- **Hides:** `DiagnosisArtifactSchema.safeParse` loop, bulk Drizzle insert into `diagnosis_artifacts`, mapping `bar_range` tuple to `bar_range_start`/`bar_range_end` columns
- **Tested through:** PGlite integration test — call with mixed valid/invalid results, assert rows inserted

### `synthesizeV6` (in `services/teacher.ts`)
- **Interface:** unchanged — `(ctx, input: SynthesisInput, sessionId, waitUntil?) => AsyncGenerator<HookEvent<SynthesisArtifact>>`
- **Hides:** Mapping of `SynthesisInput` fields to `digest` keys, inline `COHORT_TABLES` constant, `past_diagnoses: []` placeholder, full `HookContext` construction
- **Tested through:** Call with fully-populated `SynthesisInput`; assert `digest` shape from the first `runHook` argument (captured via test spy or by reading events)

### `compound-registry.ts`
- **Interface:** `getCompoundBinding(hook: HookKind): CompoundBinding | undefined` — unchanged
- **Hides:** `ALL_ATOMS` import, tool list assembly, future `ALL_MOLECULES` addition point
- **Tested through:** `getCompoundBinding("OnSessionEnd").tools.length` and all tool names unique

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/do/session-brain.ts` | Add `toEnrichedChunk()` pure function; add `barMapAlignments` capture in `handleChunkReady`; add enriched chunk `ctx.storage.put()` after WASM block; add bulk read + DB queries + `persistDiagnosisArtifacts` call in `runSynthesisAndPersist`; add key cleanup in `finalizeSession` | Modify |
| `apps/api/src/services/teacher.ts` | Add `EnrichedChunk`, `SessionHistoryRecord`, `PastDiagnosisRecord` types; expand `SynthesisInput`; add `COHORT_TABLES` constant; update `synthesizeV6` digest population | Modify |
| `apps/api/src/services/synthesis.ts` | Add `persistDiagnosisArtifacts()` function | Modify |
| `apps/api/src/db/schema/diagnosis-artifacts.ts` | New Drizzle table definition | New |
| `apps/api/src/db/schema/index.ts` | Add `export * from "./diagnosis-artifacts"` | Modify |
| `apps/api/src/harness/loop/compound-registry.ts` | Import `ALL_ATOMS`; replace `tools: []` with `tools: [...ALL_ATOMS]` | Modify |
| `apps/api/src/harness/loop/compound-registry.test.ts` | Replace `tools.toEqual([])` with `tools.length` and uniqueness assertions | Modify |
| `apps/api/src/harness/skills/__catalog__/integration.test.ts` | Add E2E test: fixture digest → fetch stub → real `extract-bar-range-signals` invocation → artifact with `focus_areas` | Modify |
| `docs/apps/00-status.md` | Note V6 integration COMPLETE, `HARNESS_V6_ENABLED=true` safe to flip | Modify |

---

## Open Questions

- Q: Should `score_index` in the alignment entries use actual WASM match position or just array index?
  Default: use array index (`i`) — `NoteAlignment` does not expose the score note index, and atoms that use alignment primarily care about `bar` and `expected_onset_ms`.

- Q: Should the session history query join `messages` for synthesis text?
  Default: yes — join `messages` on `session_id` and `message_type = 'synthesis'` to populate `SessionHistoryRecord.synthesis`. If no synthesis message exists for a session, `synthesis` is `null`.
