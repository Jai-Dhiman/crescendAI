# Session Arc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire memory retrieval into session finalization so the practice session ends with an LLM-generated, memory-informed summary instead of a raw observation list.

**Architecture:** Add a session summary prompt to `prompts.rs`, call Anthropic (non-streaming) during `finalize_session` in the DO with the session's observations + memory context, send the result in the existing `session_summary` WebSocket event's `summary` field. Client uses the LLM summary when present, falls back to client-built.

**Tech Stack:** Rust (CF Workers WASM), Anthropic Sonnet via `call_anthropic`, D1 (memory queries), TypeScript/React (client fallback)

**Spec:** `docs/superpowers/specs/2026-03-19-session-arc-design.md`

---

## File Structure

| File | Responsibility | Change Type |
|---|---|---|
| `apps/api/src/services/prompts.rs` | Session summary prompt (system + user builder) | Add function |
| `apps/api/src/practice/session.rs` | Call LLM summary in `finalize_session` | Modify lines 798-874 |
| `apps/web/src/hooks/usePracticeSession.ts` | Use LLM summary when present, fallback to client-built | Modify lines 236-253 |

---

### Task 1: Add Session Summary Prompt

**Files:**
- Modify: `apps/api/src/services/prompts.rs`

- [ ] **Step 1: Add the session summary system prompt constant**

Add after the existing `CHAT_SYSTEM` constant (around line 310):

```rust
pub const SESSION_SUMMARY_SYSTEM: &str = r#"You are a piano teacher summarizing a practice session for your student. You know this student from prior sessions.

Write 2-4 sentences in a warm, direct voice. Reference what you noticed during this session. If you have context from prior sessions, connect the dots (e.g., "Your pedaling is getting cleaner since we started working on it"). If not, focus on this session alone.

Do NOT use bullet points or formatting. Do NOT list every observation -- synthesize. Do NOT start with "Great session" or similar generic openers. Be specific about what you heard.

End with one concrete suggestion for next time."#;
```

- [ ] **Step 2: Add the user prompt builder function**

Add after `SESSION_SUMMARY_SYSTEM`:

```rust
/// Build the user prompt for session summary generation.
pub fn build_session_summary_prompt(
    observations: &[(String, String, String, f64, f64)],  // (dimension, text, framing, score, baseline)
    memory_context: &str,
    piece_context: Option<&str>,
) -> String {
    let mut prompt = String::with_capacity(2000);

    if !memory_context.is_empty() {
        prompt.push_str("<student_history>\n");
        prompt.push_str(memory_context);
        prompt.push_str("</student_history>\n\n");
    }

    if let Some(piece) = piece_context {
        prompt.push_str(&format!("<piece>{}</piece>\n\n", piece));
    }

    prompt.push_str("<session_observations>\n");
    for (dim, text, framing, score, baseline) in observations {
        prompt.push_str(&format!(
            "- [{}] {} (framing: {}, score: {:.2}, baseline: {:.2})\n",
            dim, text, framing, score, baseline
        ));
    }
    prompt.push_str("</session_observations>\n\n");

    prompt.push_str("<task>\nSummarize this practice session in 2-4 sentences. Be specific about what you heard. If student history is available, reference progress or patterns. End with one suggestion for next time.\n</task>");

    prompt
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: no errors related to prompts.rs

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/prompts.rs
git commit -m "feat: add session summary prompt for LLM-generated practice summaries"
```

---

### Task 2: Wire LLM Summary Into finalize_session

**Files:**
- Modify: `apps/api/src/practice/session.rs:798-874`

- [ ] **Step 1: Add a `generate_session_summary` helper method to `PracticeSession`**

Add before `finalize_session` (around line 796):

```rust
/// Generate an LLM session summary using Anthropic.
/// Returns the summary text, or None if generation fails or should be skipped.
async fn generate_session_summary(
    &self,
    observations: &[ObservationRecord],
    student_id: &str,
) -> Option<String> {
    // Skip for eval sessions
    if self.inner.borrow().is_eval_session {
        return None;
    }

    // Skip if no observations
    if observations.is_empty() {
        return None;
    }

    // 1. Load memory context (D1 queries for cross-session facts)
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];
    let memory_ctx = crate::services::memory::build_memory_context(
        &self.env, student_id, None, today, None,
    ).await;
    let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

    // 2. Build piece context string
    let piece_context = {
        let s = self.inner.borrow();
        s.score_context.as_ref().map(|ctx| {
            format!("{} - {}", ctx.composer, ctx.title)
        })
    };

    // 3. Build observation tuples for the prompt
    let obs_tuples: Vec<(String, String, String, f64, f64)> = observations
        .iter()
        .map(|o| (
            o.dimension.clone(),
            o.text.clone(),
            o.framing.clone(),
            o.score,
            o.baseline,
        ))
        .collect();

    // 4. Build prompt
    let user_prompt = crate::services::prompts::build_session_summary_prompt(
        &obs_tuples,
        &memory_text,
        piece_context.as_deref(),
    );

    // 5. Call Anthropic (non-streaming, 300 max tokens)
    // No explicit timeout needed: CF Workers enforces a 30s subrequest limit,
    // and the 300 max_tokens cap keeps generation fast (~1-2s typical).
    match crate::services::llm::call_anthropic(
        &self.env,
        crate::services::prompts::SESSION_SUMMARY_SYSTEM,
        &user_prompt,
        300,
    ).await {
        Ok(text) if !text.is_empty() => {
            console_log!("Session summary generated for {}", student_id);
            Some(text)
        }
        Ok(_) => None,
        Err(e) => {
            console_error!("Session summary generation failed: {}", e);
            None
        }
    }
}
```

**Note on timeout:** CF Workers enforces a 30s limit on subrequests (fetch calls to external APIs). The `call_anthropic` function uses `fetch` internally, so it will be killed after 30s automatically. The 300 max_tokens cap means Anthropic typically responds in 1-2s. No Promise.race or custom timeout is needed.

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check 2>&1 | tail -10`
Expected: no errors (the method is not called yet, but should compile)

- [ ] **Step 3: Modify `finalize_session` to call the summary generator**

Replace the existing section 3 ("Send session_summary via WebSocket") at lines 849-868 with:

```rust
        // 3. Generate LLM summary + send session_summary via WebSocket
        if let Some(ws) = ws {
            let obs_json: Vec<serde_json::Value> = observations
                .iter()
                .map(|o| serde_json::json!({
                    "text": o.text,
                    "dimension": o.dimension,
                    "framing": o.framing,
                }))
                .collect();

            // Generate LLM summary (skips for eval sessions, empty sessions, or on failure)
            let summary_text = self
                .generate_session_summary(&observations, &student_id)
                .await
                .unwrap_or_default();

            let summary = serde_json::json!({
                "type": "session_summary",
                "observations": obs_json,
                "summary": summary_text,
                "inference_failures": inference_failures,
                "total_chunks": total_chunks,
            });
            let _ = ws.send_with_str(&summary.to_string());
        }
```

- [ ] **Step 4: Verify it compiles**

Run: `cd apps/api && cargo check 2>&1 | tail -10`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: generate LLM session summary in finalize_session

Calls Anthropic Sonnet with session observations + memory context to
produce a teacher-voiced 2-4 sentence summary. Falls back to empty
string on failure (client builds raw summary). Skips for eval sessions."
```

---

### Task 3: Client Fallback Logic

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts:236-253`

- [ ] **Step 1: Modify the `session_summary` handler to prefer LLM summary**

In `usePracticeSession.ts`, replace lines 242-253 (the `builtSummary` block through `setSummary(builtSummary)`) with:

```typescript
					let builtSummary: string;
					if (inferenceFailures && totalChunks && inferenceFailures === totalChunks) {
						// All inference failed (e.g., HF endpoint cold/down)
						builtSummary = "I wasn't able to analyze your playing this time -- the analysis service is still warming up. Try again in about a minute.";
					} else if (inferenceFailures && inferenceFailures > 0 && obsLines) {
						builtSummary = `I listened to ${chunksCount} sections of your playing (${inferenceFailures} couldn't be analyzed).\n\nDuring the session, I noticed:\n${obsLines}\n\nWant to hear more about any of these?`;
					} else if (obsLines) {
						builtSummary = `I listened to ${chunksCount} sections of your playing.\n\nDuring the session, I noticed:\n${obsLines}\n\nWant to hear more about any of these?`;
					} else {
						builtSummary = `I listened to ${chunksCount} sections of your playing.\n\nI didn't notice anything specific to flag this time. Want to talk about how it felt?`;
					}
					setSummary(data.summary || builtSummary);
```

The only change is the last line: `setSummary(builtSummary)` becomes `setSummary(data.summary || builtSummary)`.

When the server sends a non-empty `summary` field (LLM-generated), it takes precedence. When empty (fallback, eval, or failure), the client-built summary is used. The "analysis service still warming up" path is preserved because the LLM summary call is skipped when there are no observations.

- [ ] **Step 2: Verify types**

Check that `data.summary` is typed as `string` in the `PracticeWsEvent` type. Read `apps/web/src/lib/practice-api.ts` and verify the `session_summary` event type includes `summary: string`.

- [ ] **Step 3: Verify it builds**

Run: `cd apps/web && bun run build 2>&1 | tail -5`
Expected: no type errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts
git commit -m "feat: prefer LLM session summary over client-built fallback"
```

---

### Task 4: E2E Verification

- [ ] **Step 1: Check the `PracticeWsEvent` type includes `summary`**

Read `apps/web/src/lib/practice-api.ts` and verify the `session_summary` variant has a `summary` field. If missing, add it:

```typescript
summary: string;
```

- [ ] **Step 2: Deploy to dev and test manually**

Run: `cd apps/api && npx wrangler deploy --env development`

Test flow:
1. Open crescend.ai dev environment
2. Start a practice session, play briefly (2-3 chunks)
3. Stop the session
4. Verify the summary shown in chat is LLM-generated prose (not a bullet list)
5. Check worker logs for the `console_log` from synthesis and any errors from summary generation

- [ ] **Step 3: Test fallback path**

Temporarily break the Anthropic call (e.g., invalid API key in dev) and verify the client-built summary still appears.

- [ ] **Step 4: Test eval session path**

Use the dev eval_chunk WebSocket message path. Verify no LLM summary call is made (check worker logs).

- [ ] **Step 5: Commit any fixes from testing**

```bash
git add -u
git commit -m "fix: address issues found during session arc E2E testing"
```

---

### Task 5: Update Documentation

**Files:**
- Modify: `docs/apps/03-memory-system.md`

- [ ] **Step 1: Update the Phase 3 status in memory system docs**

In `docs/apps/03-memory-system.md`, update the status header (line 5) to reflect that session arc is implemented:

```
> **Status (2026-03-19):** Observations table COMPLETE. Synthesized facts COMPLETE. Session arc COMPLETE (LLM-generated session summary with memory context, client fallback). Memory retrieval E2E validated. Phase 3 retrieval optimization remains.
```

- [ ] **Step 2: Add a note about session summary in the Implementation Sequence**

After the Phase 2 section (line 249), add:

```
### Phase 2.5: Session Arc -- COMPLETE

```
[x] Memory context loaded in finalize_session for summary generation
[x] LLM-generated session summary (Anthropic Sonnet, 2-4 sentences)
[x] Client-built fallback on LLM failure
[x] Eval session bypass
[x] CF Workers 30s subrequest limit as timeout backstop
[x] handle_ask_inner already loads memory per observation (no DO changes needed)
```
```

- [ ] **Step 3: Commit**

```bash
git add docs/apps/03-memory-system.md
git commit -m "docs: mark session arc as complete in memory system docs"
```
