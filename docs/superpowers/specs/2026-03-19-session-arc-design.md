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
| Where memory loads | In the DO, on first `chunk_ready` | Same pattern as baselines loading; avoids blocking WebSocket upgrade |
| Session summary | LLM-generated in `finalize_session` | Student is already waiting; one wait instead of two sequential waits |
| Summary LLM | Anthropic (Sonnet) | Same voice as chat teacher; tonal consistency; ~150-200 token generation is cheap |
| Summary fallback | Client-built from raw observations | Existing behavior preserved when LLM call fails |

## Design

### Session Open: Memory-Informed Context

When the DO receives its first `chunk_ready`, alongside the existing baselines load, it loads the student's memory context via `build_memory_context`. The formatted text is stored in `SessionState` as `memory_context: Option<String>`.

This context is passed into every `generate_observation` call via `AskInnerRequest.memory_context`. The subagent prompt receives memory data through the same mechanism that the `/api/ask` HTTP path already uses -- this just wires the DO path to the same data.

**SessionState addition:**
```rust
memory_context: Option<String>,  // formatted text from format_memory_context()
```

### Session Close: LLM-Generated Summary

After the existing finalization steps, a new step calls Anthropic with the session's observations + memory context to generate a narrative summary.

**Sequence in `finalize_session`:**
1. Persist observations to D1 (existing)
2. Increment observation count (existing)
3. Run synthesis if threshold met (existing)
4. **NEW:** Load memory context (active facts + recent observations)
5. **NEW:** Call Anthropic with session observations + memory + baselines + piece context
6. Send `session_summary` with `summary` field populated (currently always `""`)
7. Close WebSocket (existing)

**Summary prompt receives:**
- This session's observations (dimension, text, framing, score, baseline)
- Active synthesized facts (what the system knows about this student)
- Student baselines (6 dimensions)
- Piece context if available (composer, title)

**Summary output:** 2-4 sentences in the teacher's voice. References what improved, what to focus on next, and connects to prior sessions if memory provides context. Conversational prose, not bullet lists.

### Client Changes

`usePracticeSession.ts` checks if `session_summary.summary` is non-empty. If so, uses it directly instead of building the client-side summary. The client-built summary becomes the fallback path only.

## Files Changed

### API (`apps/api/src/`)

| File | Change |
|---|---|
| `practice/session.rs` | Add `memory_context: Option<String>` to `SessionState`. Load alongside baselines on first chunk. Pass into `generate_observation`. Add LLM summary call in `finalize_session`. |
| `services/ask.rs` | Add `memory_context: Option<String>` to `AskInnerRequest`. Inject into subagent prompt when present. |
| `services/prompts.rs` | Add `build_session_summary_prompt()` -- takes observations, memory context, baselines, piece context. Returns system + user prompt for Anthropic. |

### Web (`apps/web/src/`)

| File | Change |
|---|---|
| `hooks/usePracticeSession.ts` | In `session_summary` handler: if `data.summary` is non-empty, use it directly. Keep client-built as fallback. |

## Edge Cases

| Scenario | Handling |
|---|---|
| No observations in session | Skip LLM summary. Client-side fallback handles this. |
| Synthesis runs but LLM summary fails | Send `session_summary` with empty `summary`. Client falls back. Log error. |
| WebSocket disconnected before summary | Synthesis still runs. LLM summary skipped (no receiver). |
| Empty memory (new student) | Summary focuses on this session only, no cross-session references. |
| DO alarm fires (30-min timeout) | Same finalization path. Summary generated if observations exist. |

## What This Does NOT Change

- No new API endpoints
- No changes to the chat path (already has memory context)
- No changes to the synthesis pipeline logic
- No proactive "welcome back" messages
- No changes to ListeningMode UI (summary is consumed in the chat view)
