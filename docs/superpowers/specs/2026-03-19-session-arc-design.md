# Session Arc: Memory-Informed Practice Sessions

**Date:** 2026-03-19
**Status:** Approved
**Addresses:** Beta P0 -- session opening context + session closing summary (see `docs/apps/03-memory-system.md`)

---

## Problem

The memory system stores observations and synthesized facts per student, but nothing connects memory to the practice session lifecycle. The session DO generates observations without knowing what the student worked on last time, and the session summary is a raw bullet list built client-side with no teacher voice or cross-session context.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Welcome back message | No proactive message | Memory context loads silently; the teacher's first observation is naturally informed by prior sessions |
| Session open memory | No changes needed | `handle_ask_inner` already calls `build_memory_context()` per observation -- the DO path already gets memory |
| Session summary | LLM-generated in `finalize_session` | Student is already waiting; one wait instead of two sequential waits |
| Summary LLM | Anthropic (Sonnet) | Same voice as chat teacher; tonal consistency; ~150-200 token generation is cheap |
| Summary fallback | Client-built from raw observations | Existing behavior preserved when LLM call fails |

## Design

### Session Open: No Changes Required

`handle_ask_inner` (in `services/ask.rs`, line 100) already calls `build_memory_context()` on every invocation, using the student ID and piece context from the request. This means every observation the DO generates through `generate_observation` -> `handle_ask_inner` already gets fresh memory context (active synthesized facts, recent observations, piece-specific facts). No caching in `SessionState` is needed -- and caching would actually be worse, since memory goes stale as observations accumulate during a session.

### Session Close: LLM-Generated Summary

After the existing finalization steps, new steps load memory and call Anthropic to generate a narrative summary.

**Sequence in `finalize_session`:**
1. Persist observations to D1 (existing)
2. Increment observation count (existing)
3. **NEW:** Start memory context load (`build_memory_context` D1 query) and synthesis check concurrently
4. Run synthesis if threshold met (existing) -- runs concurrently with step 5
5. **NEW:** Call Anthropic with session observations + memory + baselines + piece context -- runs concurrently with step 4
6. Send `session_summary` with `summary` field populated (currently always `""`)
7. Close WebSocket (existing)

**Concurrency note:** Steps 3-5 should overlap where possible. In CF Workers WASM, we cannot use `tokio::join!`, but we can structure the code to initiate the memory load first, then while synthesis runs via Groq, fire the Anthropic summary call. The wall-clock time is max(synthesis, summary) rather than sum. If structuring true concurrency is impractical in the DO's `RefCell` borrowing model, sequential is acceptable -- add a 5s timeout on the Anthropic call so it does not block finalization indefinitely.

**Summary prompt receives:**
- This session's observations (dimension, text, framing, score, baseline)
- Active synthesized facts from `build_memory_context` (cross-session context)
- Student baselines (6 dimensions)
- Piece context if available (composer, title)

**Summary output:** 2-4 sentences in the teacher's voice. References what improved, what to focus on next, and connects to prior sessions if memory provides context. Conversational prose, not bullet lists.

### Client Changes

In `usePracticeSession.ts`, the `session_summary` handler checks if `data.summary` is non-empty. If so, it uses `data.summary` directly as the value passed to `setSummary()`, skipping the entire `builtSummary` construction block. If `data.summary` is empty or missing, the existing client-built logic runs as-is (including the "analysis service still warming up" path for full inference failure).

Concretely: `setSummary(data.summary || builtSummary)`.

## Files Changed

### API (`apps/api/src/`)

| File | Change |
|---|---|
| `practice/session.rs` | Add LLM summary call + memory load in `finalize_session`. No changes to `SessionState` or observation generation. |
| `services/prompts.rs` | Add `build_session_summary_prompt()` -- takes observations, memory context, baselines, piece context. Returns system + user prompt for Anthropic. |

### Web (`apps/web/src/`)

| File | Change |
|---|---|
| `hooks/usePracticeSession.ts` | In `session_summary` handler: use `data.summary` when non-empty, fall back to client-built `builtSummary`. |

## Edge Cases

| Scenario | Handling |
|---|---|
| No observations in session | Skip LLM summary. Client-side fallback handles this. |
| Synthesis runs but LLM summary fails | Send `session_summary` with empty `summary`. Client falls back. Log error. |
| WebSocket disconnected before summary | Synthesis still runs. LLM summary skipped (no receiver). |
| Empty memory (new student) | Summary focuses on this session only, no cross-session references. |
| DO alarm fires (30-min timeout) | Same finalization path. Summary generated if observations exist. |
| Eval session (`is_eval_session = true`) | Skip LLM summary. Eval pipelines should not incur LLM cost/latency. Send empty `summary`. |
| Anthropic call exceeds 5s timeout | Abort, log error, send empty `summary`. Client falls back. |

## What This Does NOT Change

- No new API endpoints
- No changes to `AskInnerRequest` or `handle_ask_inner` (already loads memory per call)
- No changes to the chat path (already has memory context)
- No changes to the synthesis pipeline logic
- No proactive "welcome back" messages
- No changes to ListeningMode UI (summary is consumed in the chat view)
- No changes to `SessionState` struct
