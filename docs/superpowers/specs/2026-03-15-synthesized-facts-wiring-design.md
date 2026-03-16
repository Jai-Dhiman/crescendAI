# Synthesized Facts Wiring -- Design Spec

Wire the existing synthesis pipeline (`run_synthesis()`, `should_synthesize()`) into two trigger points, add response observability, and create an integration test.

## Context

The synthesis pipeline converts raw practice observations into temporal facts about a student. Nearly all implementation exists in `memory.rs` and `prompts.rs` -- the missing piece is the trigger that actually calls it.

### What already exists

- `run_synthesis(env, student_id)` -- full pipeline: fetch active facts + new observations + teaching approaches + baselines, call Groq (Llama 70B) with `SYNTHESIS_SYSTEM` prompt, parse JSON, apply invalidations, insert new facts, update `student_memory_meta`
- `should_synthesize(env, student_id)` -- trigger check with three branches: (a) never synthesized before: true when `total_observations >= 3` from `student_memory_meta`; (b) previously synthesized, >= 3 new observations since `last_synthesis_at` (date-windowed COUNT query); (c) previously synthesized, > 7 days since last synthesis with any new observations
- `increment_observation_count(env, student_id)` -- called from `handle_ask()` after each observation is stored. **Bug:** NOT called from DO-originated observations (WebSocket practice sessions), so the `student_memory_meta.total_observations` counter is only accurate for HTTP `/api/ask` observations
- `SYNTHESIS_SYSTEM` prompt + `build_synthesis_prompt()` -- complete prompt with JSON schema for new/invalidated/unchanged facts
- `synthesized_facts` table + `student_memory_meta` table -- in `0001_init.sql`
- Active facts already wired into subagent context via `build_memory_context()` -> `format_memory_context()` in `handle_ask_inner()`

### What is missing

1. Nothing calls `should_synthesize()` or `run_synthesis()`
2. `run_synthesis()` returns `Result<(), String>` -- no observability into what changed
3. No HTTP endpoint for manual/eval triggering
4. No integration test
5. DO-originated observations (WebSocket sessions) never call `increment_observation_count()`, so the meta counter is inaccurate for the most common user flow

## Design

### 1. SynthesisResult return type (memory.rs)

Refactor `run_synthesis()` from `Result<(), String>` to `Result<SynthesisResult, String>`:

```rust
pub struct SynthesisResult {
    pub new_facts: usize,
    pub invalidated: usize,
    pub unchanged: usize,
    pub observations_processed: usize,
}
```

Count `new_facts` and `invalidated` from the parsed Groq response arrays. Derive `unchanged` arithmetically as `active_facts.len() - invalidated` (do not trust the LLM's `unchanged_facts` array -- it may omit or hallucinate IDs). Return `observations_processed` from `new_observations.len()`. Early return when no new observations returns a zeroed result. The JSON parsing uses the existing `extract_synthesis_json()` helper (memory.rs line 1625).

### 2. DO session finalization trigger (session.rs)

Insert between `persist_observations()` and the WS summary send in `finalize_session()`.

**Fix observation counting bug:** DO-originated observations (WebSocket sessions) currently never call `increment_observation_count()`. Add a loop in `finalize_session()` after `persist_observations()` to call `increment_observation_count()` for each persisted observation. This ensures the `student_memory_meta.total_observations` counter is accurate for both HTTP and WebSocket flows.

**Avoid stale-read issue:** `should_synthesize()` uses the meta counter (not a date-windowed count query) for never-synthesized students. Since `increment_observation_count()` is called before `should_synthesize()`, the counter reflects the just-persisted observations. For previously-synthesized students, `should_synthesize()` uses a `COUNT(*)` query against `observations WHERE created_at > last_synthesis_at`. The observations were just written by `persist_observations()` in the same DO invocation -- D1 read-after-write consistency within a single Worker invocation is guaranteed (D1 uses a single HTTP connection per binding).

```rust
// Update observation count for synthesis tracking
for _ in &observations {
    if let Err(e) = crate::services::memory::increment_observation_count(
        &self.env, &student_id
    ).await {
        console_error!("Failed to increment observation count: {}", e);
    }
}

// Run synthesis if enough observations have accumulated
match crate::services::memory::should_synthesize(&self.env, &student_id).await {
    Ok(true) => {
        match crate::services::memory::run_synthesis(&self.env, &student_id).await {
            Ok(result) => {
                console_log!(
                    "Synthesis for {}: {} new, {} invalidated, {} unchanged",
                    student_id, result.new_facts, result.invalidated, result.unchanged
                );
            }
            Err(e) => {
                console_error!("Synthesis failed for {}: {}", student_id, e);
            }
        }
    }
    Ok(false) => {}
    Err(e) => {
        console_error!("Synthesis check failed: {}", e);
    }
}
```

Awaited (not fire-and-forget). The ~0.3s Groq latency is acceptable at session end -- the student has stopped playing. Synthesis failure is non-fatal: session summary and WS close still proceed.

### 3. HTTP endpoint (memory.rs + server.rs)

`POST /api/memory/synthesize` -- manual/eval trigger.

Request body: `{"student_id": "..."}` (explicit, consistent with `/api/memory/store-facts` and `/api/memory/search`).

Response:
- If synthesis ran: `{"new_facts": N, "invalidated": N, "unchanged": N, "observations_processed": N}`
- If threshold not met: `{"skipped": true, "reason": "Not enough new observations"}`

Handler follows `handle_store_facts` pattern: auth check, parse body, delegate to `should_synthesize()` + `run_synthesis()`.

Route added in `server.rs` alongside existing `/api/memory/*` routes.

### 4. Error handling

All errors are non-fatal. Synthesis failure never blocks the user experience.

| Error | Location | Handling |
|---|---|---|
| `should_synthesize()` D1 query fails | DO finalization | `console_error!`, skip synthesis, proceed to WS summary |
| `run_synthesis()` Groq call fails | DO finalization | `console_error!`, proceed to WS summary. Meta NOT updated, so next session retries |
| Groq returns unparseable JSON | `run_synthesis()` | Returns `Err`. Observation count preserved for next attempt |
| Groq hallucinates a fact_id for invalidation | `run_synthesis()` | UPDATE with `WHERE id = ? AND student_id = ?` matches zero rows. No damage |
| D1 write fails mid-synthesis | `run_synthesis()` | Partial facts written, meta NOT updated (meta update is the last step). Next synthesis re-processes the same observations. Duplicate prevention: the synthesis prompt receives current active facts, so the LLM sees already-created facts and places them in `unchanged_facts` rather than re-creating them |
| HTTP endpoint with invalid student_id | `handle_synthesize()` | `should_synthesize()` returns `Ok(false)`, response is `{"skipped": true}` |

### 5. Integration test (apps/api/evals/memory/src/test_synthesis.py)

Python test using existing eval infrastructure. Hits production API endpoints against dev environment.

**Test sequence:**

1. **Setup** -- auth JWT for test student, clear stale test data via `POST /api/memory/clear-benchmark`
2. **Seed observations** -- 5 observations across 3 dimensions via direct D1 seeding endpoint:
   - dynamics: score 0.3 (below baseline 0.5) x2
   - pedaling: score 0.7 (above baseline 0.5) x2
   - timing: score 0.4 (below baseline 0.5) x1
   - Each with reasoning_trace and framing
3. **First synthesis** -- `POST /api/memory/synthesize {student_id}`
   - Assert: `new_facts >= 1`, `invalidated == 0`, `observations_processed == 5`
4. **Verify facts** -- `POST /api/memory/search {student_id, query: "dynamics"}`
   - Assert: at least 1 fact mentioning dynamics
   - Verify bi-temporal: `valid_at` set, `invalid_at` null
5. **Seed contradicting observations** -- 3 dynamics observations with score 0.8 (improvement)
6. **Second synthesis** -- `POST /api/memory/synthesize {student_id}`
   - Assert: `new_facts >= 1`, `invalidated >= 1`
7. **Verify invalidation** -- old dynamics weakness fact has `invalid_at` set, new improvement fact exists
8. **Cleanup** -- clear test data

9. **Threshold not met** -- seed 1 observation only, call `POST /api/memory/synthesize`, assert response is `{"skipped": true, ...}`

**Non-determinism management:**
- Temperature 0.1 (matching production config)
- Assert structural properties (counts > 0) not exact text
- Mark `@pytest.mark.integration` for CI exclusion

### 6. Observation seeding endpoint

The test needs to insert observations directly into D1 without going through the full HF inference + STOP classifier pipeline. Add `POST /api/memory/seed-observations` (test/eval only):

Request: `{"student_id": "...", "observations": [{"dimension": "dynamics", "observation_text": "...", "framing": "correction", "dimension_score": 0.3, "student_baseline": 0.5, "reasoning_trace": "..."}]}`

Response: `{"seeded": N}`

This follows the `store-facts` pattern. Gated behind `AUTH_DEBUG_ENABLED` (same guard as `/api/auth/debug`) to prevent production users from injecting arbitrary observations. Returns 404 in production.

The seed endpoint must also call `increment_observation_count()` for each seeded observation, so that `should_synthesize()` returns correct results when the test subsequently calls `/api/memory/synthesize`.

## Files changed

| File | Change |
|---|---|
| `apps/api/src/services/memory.rs` | Add `SynthesisResult` struct, refactor `run_synthesis()` return type, add `handle_synthesize()`, add `handle_seed_observations()` |
| `apps/api/src/practice/session.rs` | Add synthesis trigger in `finalize_session()` |
| `apps/api/src/server.rs` | Add `POST /api/memory/synthesize` and `POST /api/memory/seed-observations` routes |
| `apps/api/evals/memory/src/test_synthesis.py` | Integration test |

## What is NOT changing

- `SYNTHESIS_SYSTEM` prompt (already complete)
- `build_synthesis_prompt()` (already complete)
- `synthesized_facts` D1 schema (already in 0001_init.sql)
- `build_memory_context()` / `format_memory_context()` (already wires facts into subagent)
- `query_active_facts()` (already works)
