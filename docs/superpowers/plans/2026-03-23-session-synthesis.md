# Session Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace real-time per-observation LLM delivery with accumulated-signal post-session synthesis, so the teacher gives one cohesive response when the student stops playing.

**Architecture:** The DO silently accumulates structured signals (teaching moments, mode transitions, drilling records) during listening mode -- no LLM calls, no WS pushes. When the user exits listening mode (`end_session`), a single Anthropic teacher call synthesizes the accumulated data into a cohesive teaching response. Eval sessions preserve the old per-chunk path.

**Tech Stack:** Rust (Cloudflare Workers WASM), D1 (SQLite), serde, Anthropic API

**Spec:** `docs/superpowers/specs/2026-03-23-session-synthesis-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `apps/api/src/practice/accumulator.rs` | Create | SessionAccumulator struct, accumulation methods, prompt builder, top_moments selection |
| `apps/api/src/practice/synthesis.rs` | Create | synthesize_session() core logic (prompt build, Anthropic call, D1 persist). Used by both DO and Worker route. |
| `apps/api/src/practice/session.rs` | Modify | Replace observation buffer with accumulator, wire accumulation into chunk pipeline, update end_session/finalize |
| `apps/api/src/practice/practice_mode.rs` | Modify | Add DrillingRecord final_scores snapshot on mode exit, rename observation_policy() to mode_context() |
| `apps/api/src/practice/mod.rs` | Modify | Add `pub mod accumulator; pub mod synthesis;` |
| `apps/api/src/server.rs` | Modify | Add `POST /api/practice/synthesize` route |
| `apps/api/migrations/0008_session_synthesis.sql` | Create | ALTER TABLE sessions ADD COLUMN accumulator_json, needs_synthesis |
| `apps/api/src/services/prompts.rs` | Modify | Add synthesis system prompt |
| `apps/web/src/hooks/usePracticeSession.ts` | Modify | Remove observation handling during recording, add synthesis event handling |
| `apps/web/src/components/ListeningMode.tsx` | Modify | Remove observation toast display during listening |
| `apps/web/src/lib/practice-api.ts` | Modify | Add SynthesisEvent type, add deferred synthesis check |

---

### Task 1: D1 Migration

**Files:**
- Create: `apps/api/migrations/0008_session_synthesis.sql`

- [ ] **Step 1: Write the migration SQL**

```sql
-- 0008_session_synthesis.sql
-- Add columns for deferred synthesis recovery.
-- accumulator_json stores the serialized SessionAccumulator when the DO
-- cannot synthesize (disconnect, alarm timeout).
-- needs_synthesis flags sessions that need deferred synthesis on next load.

ALTER TABLE sessions ADD COLUMN accumulator_json TEXT;
ALTER TABLE sessions ADD COLUMN needs_synthesis INTEGER DEFAULT 0;
```

- [ ] **Step 2: Apply migration locally**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --local`
Expected: Migration 0008 applied successfully

- [ ] **Step 3: Verify columns exist**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --command "PRAGMA table_info(sessions);" | grep -E "accumulator_json|needs_synthesis"`
Expected: Both columns listed

- [ ] **Step 4: Commit**

```bash
git add apps/api/migrations/0008_session_synthesis.sql
git commit -m "feat: add D1 migration for session synthesis recovery columns"
```

---

### Task 2: SessionAccumulator Struct + Unit Tests

**Files:**
- Create: `apps/api/src/practice/accumulator.rs`
- Modify: `apps/api/src/practice/mod.rs` -- add `pub mod accumulator;`

- [ ] **Step 1: Write the accumulator struct with tests**

```rust
// apps/api/src/practice/accumulator.rs

use crate::practice::practice_mode::PracticeMode;

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionAccumulator {
    pub teaching_moments: Vec<AccumulatedMoment>,
    pub mode_transitions: Vec<ModeTransitionRecord>,
    pub drilling_records: Vec<DrillingRecord>,
    pub timeline: Vec<TimelineEvent>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct AccumulatedMoment {
    pub chunk_index: usize,
    pub dimension: String,
    pub score: f64,
    pub baseline: f64,
    pub deviation: f64,
    pub is_positive: bool,
    pub reasoning: String,
    pub bar_range: Option<(u32, u32)>,
    pub analysis_tier: u8,
    pub timestamp_ms: u64,
    pub llm_analysis: Option<String>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ModeTransitionRecord {
    pub from: PracticeMode,
    pub to: PracticeMode,
    pub chunk_index: usize,
    pub timestamp_ms: u64,
    pub dwell_ms: u64,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DrillingRecord {
    pub bar_range: Option<(u32, u32)>,
    pub repetition_count: usize,
    pub first_scores: [f64; 6],
    pub final_scores: [f64; 6],
    pub started_at_chunk: usize,
    pub ended_at_chunk: usize,
}

/// Gap detection only -- chunk-to-mode mapping derived from mode_transitions.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TimelineEvent {
    pub chunk_index: usize,
    pub timestamp_ms: u64,
    pub has_audio: bool,
}

impl SessionAccumulator {
    pub fn accumulate_moment(&mut self, moment: AccumulatedMoment) {
        self.teaching_moments.push(moment);
    }

    pub fn accumulate_mode_transition(&mut self, record: ModeTransitionRecord) {
        self.mode_transitions.push(record);
    }

    pub fn accumulate_drilling_record(&mut self, record: DrillingRecord) {
        self.drilling_records.push(record);
    }

    pub fn accumulate_timeline_event(&mut self, event: TimelineEvent) {
        self.timeline.push(event);
    }

    pub fn has_teaching_content(&self) -> bool {
        !self.teaching_moments.is_empty() || !self.drilling_records.is_empty()
    }

    /// Select top moments for the synthesis prompt.
    /// Algorithm: top-1 per dimension by |deviation|, plus top-1 positive per
    /// dimension if available. Cap at 8 total. Sort by chunk_index.
    pub fn top_moments(&self) -> Vec<&AccumulatedMoment> {
        use std::collections::HashMap;

        let mut by_dim: HashMap<&str, Vec<&AccumulatedMoment>> = HashMap::new();
        for m in &self.teaching_moments {
            by_dim.entry(&m.dimension).or_default().push(m);
        }

        let mut selected: Vec<&AccumulatedMoment> = Vec::new();

        for (_dim, moments) in &by_dim {
            // Top-1 by |deviation|
            if let Some(top) = moments.iter().max_by(|a, b| {
                a.deviation.abs().partial_cmp(&b.deviation.abs()).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                selected.push(top);
            }
            // Top-1 positive (if different from above)
            if let Some(top_pos) = moments.iter()
                .filter(|m| m.is_positive)
                .max_by(|a, b| a.deviation.partial_cmp(&b.deviation).unwrap_or(std::cmp::Ordering::Equal))
            {
                if !selected.iter().any(|s| std::ptr::eq(*s, *top_pos)) {
                    selected.push(top_pos);
                }
            }
        }

        // Cap at 8, sort by chunk_index for chronological narrative
        selected.sort_by_key(|m| m.chunk_index);
        selected.truncate(8);
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_moment(chunk: usize, dim: &str, deviation: f64, positive: bool) -> AccumulatedMoment {
        AccumulatedMoment {
            chunk_index: chunk,
            dimension: dim.to_string(),
            score: 0.5 + deviation,
            baseline: 0.5,
            deviation,
            is_positive: positive,
            reasoning: format!("test moment {}", chunk),
            bar_range: None,
            analysis_tier: 1,
            timestamp_ms: chunk as u64 * 15000,
            llm_analysis: None,
        }
    }

    #[test]
    fn test_accumulate_moment() {
        let mut acc = SessionAccumulator::default();
        assert!(!acc.has_teaching_content());

        acc.accumulate_moment(make_moment(5, "dynamics", 0.3, true));
        assert!(acc.has_teaching_content());
        assert_eq!(acc.teaching_moments.len(), 1);
        assert_eq!(acc.teaching_moments[0].dimension, "dynamics");
    }

    #[test]
    fn test_top_moments_dedup_by_dimension() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(make_moment(1, "dynamics", 0.1, true));
        acc.accumulate_moment(make_moment(3, "dynamics", 0.4, true));
        acc.accumulate_moment(make_moment(5, "dynamics", 0.2, false));

        let top = acc.top_moments();
        // Should have: top-1 by |deviation| (0.4) + top-1 negative (-0.2, but is_positive=false with dev=0.2)
        // Actually: 0.4 is top by |deviation| and is positive. 0.2 is not positive.
        // So: top-1 = chunk 3 (dev 0.4), no separate positive needed (top is already positive)
        // But chunk 5 is negative -> no top-1 positive different from top-1
        assert!(top.len() <= 2);
        assert_eq!(top[0].chunk_index, 3); // highest |deviation|
    }

    #[test]
    fn test_top_moments_cap_at_8() {
        let mut acc = SessionAccumulator::default();
        let dims = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"];
        for (i, dim) in dims.iter().enumerate() {
            acc.accumulate_moment(make_moment(i, dim, 0.3, true));
            acc.accumulate_moment(make_moment(i + 10, dim, -0.2, false));
        }
        let top = acc.top_moments();
        assert!(top.len() <= 8);
    }

    #[test]
    fn test_top_moments_sorted_chronologically() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(make_moment(10, "dynamics", 0.5, true));
        acc.accumulate_moment(make_moment(2, "timing", 0.3, false));
        let top = acc.top_moments();
        assert_eq!(top[0].chunk_index, 2);
        assert_eq!(top[1].chunk_index, 10);
    }

    #[test]
    fn test_accumulate_mode_transition() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Running,
            chunk_index: 4,
            timestamp_ms: 60000,
            dwell_ms: 60000,
        });
        assert_eq!(acc.mode_transitions.len(), 1);
        assert_eq!(acc.mode_transitions[0].dwell_ms, 60000);
    }

    #[test]
    fn test_accumulate_drilling_record() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_drilling_record(DrillingRecord {
            bar_range: Some((12, 16)),
            repetition_count: 4,
            first_scores: [0.4, 0.5, 0.3, 0.6, 0.5, 0.4],
            final_scores: [0.7, 0.6, 0.5, 0.7, 0.6, 0.5],
            started_at_chunk: 8,
            ended_at_chunk: 20,
        });
        assert_eq!(acc.drilling_records.len(), 1);
        assert_eq!(acc.drilling_records[0].repetition_count, 4);
    }

    #[test]
    fn test_serde_round_trip() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(make_moment(1, "dynamics", 0.3, true));
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Running,
            chunk_index: 4,
            timestamp_ms: 60000,
            dwell_ms: 60000,
        });

        let json = serde_json::to_string(&acc).unwrap();
        let restored: SessionAccumulator = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.teaching_moments.len(), 1);
        assert_eq!(restored.mode_transitions.len(), 1);
    }
}
```

- [ ] **Step 2: Add module to mod.rs**

Add `pub mod accumulator;` to `apps/api/src/practice/mod.rs`.

- [ ] **Step 3: Run tests**

Run: `cd apps/api && cargo test accumulator -- --nocapture`
Expected: All 6 tests pass

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/accumulator.rs apps/api/src/practice/mod.rs
git commit -m "feat: add SessionAccumulator struct with accumulation + top_moments selection"
```

---

### Task 3: Synthesis Module (Prompt Builder + LLM Call)

**Files:**
- Create: `apps/api/src/practice/synthesis.rs`
- Modify: `apps/api/src/practice/mod.rs` -- add `pub mod synthesis;`
- Modify: `apps/api/src/services/prompts.rs` -- add synthesis system prompt

- [ ] **Step 1: Add synthesis system prompt to prompts.rs**

Append to `apps/api/src/services/prompts.rs`:

```rust
/// Synthesis system prompt -- single call after session ends.
/// The structured JSON context IS the analysis; the teacher narrates.
pub const SYNTHESIS_SYSTEM: &str = r#"You are a warm, perceptive piano teacher reviewing a practice session. You watched the entire session and now give your student one cohesive, encouraging response.

## What you receive

A JSON object with the full session context: duration, practice pattern (modes and transitions), top teaching moments (dimensions with scores and deviations from baseline), drilling progress, and student memory.

## How to respond

1. Start with what went well -- acknowledge effort and specific improvements.
2. Identify the 1-2 most important things to work on, grounded in the session data.
3. If drilling occurred, comment on the progression (first vs final scores).
4. Frame suggestions as actionable practice strategies, not abstract criticism.
5. Keep it conversational -- 3-6 sentences. You are talking TO the student.
6. Reference specific musical details (bars, sections, dimensions) when the data supports it.
7. Do NOT mention scores, numbers, or model outputs directly. Translate them into musical language.
8. Do NOT list all dimensions. Focus on what matters most for THIS session.

## Calibration

The MuQ audio model has R2~0.5 and 80% pairwise accuracy. Scores are directional signals, not precise measurements. A deviation of 0.1 is noise; 0.2+ is meaningful. Use deviations to identify patterns, not to make absolute claims."#;
```

- [ ] **Step 2: Write synthesis.rs**

```rust
// apps/api/src/practice/synthesis.rs
//! Post-session synthesis: builds prompt from accumulator, calls Anthropic teacher,
//! persists result to D1. Used by both the DO (normal path) and the Worker route
//! (deferred recovery).

use wasm_bindgen::JsValue;
use worker::Env;

use crate::practice::accumulator::{SessionAccumulator, AccumulatedMoment};
use crate::practice::practice_mode::PracticeMode;
use crate::services::teaching_moments::StudentBaselines;

/// Context needed for synthesis beyond the accumulator itself.
pub struct SynthesisContext {
    pub session_id: String,
    pub student_id: String,
    pub conversation_id: String,
    pub baselines: Option<StudentBaselines>,
    pub piece_context: Option<serde_json::Value>,
    pub student_memory: Option<String>,
    pub total_chunks: usize,
    pub session_duration_ms: u64,
}

pub struct SynthesisResult {
    pub text: String,
    pub is_fallback: bool,
}

/// Build the structured JSON prompt context from the accumulator.
pub fn build_synthesis_prompt(
    acc: &SessionAccumulator,
    ctx: &SynthesisContext,
) -> serde_json::Value {
    let duration_min = ctx.session_duration_ms as f64 / 60_000.0;

    // Build practice_pattern from mode transitions
    let practice_pattern = build_practice_pattern(acc, ctx);

    // Select top moments
    let top_moments: Vec<serde_json::Value> = acc.top_moments()
        .into_iter()
        .map(|m| moment_to_json(m))
        .collect();

    // Build baselines JSON
    let baselines = ctx.baselines.as_ref().map(|b| {
        serde_json::json!({
            "dynamics": b.dynamics,
            "timing": b.timing,
            "pedaling": b.pedaling,
            "articulation": b.articulation,
            "phrasing": b.phrasing,
            "interpretation": b.interpretation,
        })
    }).unwrap_or(serde_json::json!(null));

    let mut prompt = serde_json::json!({
        "session_duration_minutes": (duration_min * 10.0).round() / 10.0,
        "chunks_processed": ctx.total_chunks,
        "practice_pattern": practice_pattern,
        "top_moments": top_moments,
        "baselines": baselines,
    });

    if let Some(ref piece) = ctx.piece_context {
        prompt["piece"] = piece.clone();
    }
    if let Some(ref memory) = ctx.student_memory {
        prompt["student_memory"] = serde_json::json!(memory);
    }

    // Add drilling progress if any
    if !acc.drilling_records.is_empty() {
        let drilling: Vec<serde_json::Value> = acc.drilling_records.iter().map(|d| {
            serde_json::json!({
                "bar_range": d.bar_range,
                "repetitions": d.repetition_count,
                "first_scores": {
                    "dynamics": d.first_scores[0],
                    "timing": d.first_scores[1],
                    "pedaling": d.first_scores[2],
                    "articulation": d.first_scores[3],
                    "phrasing": d.first_scores[4],
                    "interpretation": d.first_scores[5],
                },
                "final_scores": {
                    "dynamics": d.final_scores[0],
                    "timing": d.final_scores[1],
                    "pedaling": d.final_scores[2],
                    "articulation": d.final_scores[3],
                    "phrasing": d.final_scores[4],
                    "interpretation": d.final_scores[5],
                },
            })
        }).collect();
        prompt["drilling_progress"] = serde_json::json!(drilling);
    }

    prompt
}

fn moment_to_json(m: &AccumulatedMoment) -> serde_json::Value {
    let mut j = serde_json::json!({
        "dimension": m.dimension,
        "deviation": m.deviation,
        "is_positive": m.is_positive,
        "reasoning": m.reasoning,
    });
    if let Some(ref br) = m.bar_range {
        j["bar_range"] = serde_json::json!(br);
    }
    j
}

fn build_practice_pattern(
    acc: &SessionAccumulator,
    ctx: &SynthesisContext,
) -> Vec<serde_json::Value> {
    if acc.mode_transitions.is_empty() {
        // No transitions -- single mode for entire session
        return vec![serde_json::json!({
            "mode": "warming",
            "duration_min": ctx.session_duration_ms as f64 / 60_000.0,
            "chunks": ctx.total_chunks,
        })];
    }

    let mut pattern = Vec::new();
    let mut transitions = acc.mode_transitions.iter().peekable();

    while let Some(t) = transitions.next() {
        let duration_ms = if let Some(next) = transitions.peek() {
            next.timestamp_ms.saturating_sub(t.timestamp_ms)
        } else {
            // Last mode -- runs until session end
            ctx.session_duration_ms.saturating_sub(t.timestamp_ms)
        };

        let mut entry = serde_json::json!({
            "mode": format!("{:?}", t.to).to_lowercase(),
            "duration_min": (duration_ms as f64 / 60_000.0 * 10.0).round() / 10.0,
        });

        // If drilling, find matching drilling record
        if t.to == PracticeMode::Drilling {
            if let Some(dr) = acc.drilling_records.iter().find(|d| d.started_at_chunk == t.chunk_index) {
                entry["bar_range"] = serde_json::json!(dr.bar_range);
                entry["repetitions"] = serde_json::json!(dr.repetition_count);
            }
        }

        pattern.push(entry);
    }

    pattern
}

/// Call the Anthropic teacher with the synthesis prompt.
/// Returns the synthesis text or a fallback on error.
pub async fn call_synthesis_llm(
    env: &Env,
    prompt_context: &serde_json::Value,
) -> SynthesisResult {
    let system_prompt = crate::services::prompts::SYNTHESIS_SYSTEM;
    let user_message = serde_json::to_string_pretty(prompt_context).unwrap_or_default();

    let api_key = match env.secret("ANTHROPIC_API_KEY") {
        Ok(k) => k.to_string(),
        Err(e) => {
            console_error!("ANTHROPIC_API_KEY not found: {:?}", e);
            return SynthesisResult {
                text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
                is_fallback: true,
            };
        }
    };

    let model = env.var("ANTHROPIC_MODEL")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

    let body = serde_json::json!({
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    });

    let start_ms = js_sys::Date::now() as u64;

    let mut headers = worker::Headers::new();
    let _ = headers.set("Content-Type", "application/json");
    let _ = headers.set("x-api-key", &api_key);
    let _ = headers.set("anthropic-version", "2023-06-01");

    let mut init = worker::RequestInit::new();
    init.with_method(worker::Method::Post)
        .with_headers(headers)
        .with_body(Some(JsValue::from_str(&body.to_string())));

    let request = match worker::Request::new_with_init(
        "https://api.anthropic.com/v1/messages",
        &init,
    ) {
        Ok(r) => r,
        Err(e) => {
            console_error!("Failed to build synthesis request: {:?}", e);
            return SynthesisResult {
                text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
                is_fallback: true,
            };
        }
    };

    let response = match worker::Fetch::Request(request).send().await {
        Ok(mut r) => {
            let latency = js_sys::Date::now() as u64 - start_ms;
            console_log!("Synthesis LLM call: status={}, latency={}ms", r.status_code(), latency);
            match r.text().await {
                Ok(text) => text,
                Err(e) => {
                    console_error!("Failed to read synthesis response body: {:?}", e);
                    return SynthesisResult {
                        text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
                        is_fallback: true,
                    };
                }
            }
        }
        Err(e) => {
            console_error!("Synthesis LLM call failed: {:?}", e);
            return SynthesisResult {
                text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
                is_fallback: true,
            };
        }
    };

    // Parse Anthropic response
    let parsed: serde_json::Value = match serde_json::from_str(&response) {
        Ok(v) => v,
        Err(e) => {
            console_error!("Failed to parse synthesis response: {:?}. Raw: {}", e, &response[..response.len().min(500)]);
            return SynthesisResult {
                text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
                is_fallback: true,
            };
        }
    };

    let text = parsed.get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|block| block.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    if text.is_empty() {
        console_error!("Synthesis returned empty text. Response: {}", &response[..response.len().min(500)]);
        return SynthesisResult {
            text: "I had trouble preparing your feedback. Try playing again and I'll have more to say next time.".to_string(),
            is_fallback: true,
        };
    }

    SynthesisResult { text, is_fallback: false }
}

/// Persist the synthesis result to D1 as a message in the conversation.
pub async fn persist_synthesis_message(
    env: &Env,
    conversation_id: &str,
    session_id: &str,
    synthesis_text: &str,
) -> Result<String, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding: {:?}", e))?;
    let msg_id = crate::services::ask::generate_uuid();
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    let stmt = db.prepare(
        "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
         VALUES (?1, ?2, 'assistant', ?3, 'synthesis', ?4, ?5)"
    )
    .bind(&[
        JsValue::from_str(&msg_id),
        JsValue::from_str(conversation_id),
        JsValue::from_str(synthesis_text),
        JsValue::from_str(session_id),
        JsValue::from_str(&now),
    ])
    .map_err(|e| format!("Bind failed: {:?}", e))?;

    stmt.run().await.map_err(|e| format!("Run failed: {:?}", e))?;
    Ok(msg_id)
}

/// Persist accumulated moments to the observations table (for analytics/memory).
pub async fn persist_accumulated_moments(
    env: &Env,
    student_id: &str,
    session_id: &str,
    moments: &[AccumulatedMoment],
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding: {:?}", e))?;
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    for m in moments {
        let id = crate::services::ask::generate_uuid();
        let stmt = db.prepare(
            "INSERT OR IGNORE INTO observations (id, student_id, session_id, chunk_index, \
             dimension, observation_text, reasoning_trace, framing, dimension_score, \
             student_baseline, piece_context, is_fallback, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
        )
        .bind(&[
            JsValue::from_str(&id),
            JsValue::from_str(student_id),
            JsValue::from_str(session_id),
            JsValue::from_f64(m.chunk_index as f64),
            JsValue::from_str(&m.dimension),
            JsValue::from_str(&m.reasoning),  // observation_text repurposed
            JsValue::from_str(""),             // reasoning_trace (not applicable)
            JsValue::from_str(if m.is_positive { "recognition" } else { "correction" }),
            JsValue::from_f64(m.score),
            JsValue::from_f64(m.baseline),
            JsValue::NULL,
            JsValue::from_bool(false),
            JsValue::from_str(&now),
        ]);

        if let Ok(s) = stmt {
            if let Err(e) = s.run().await {
                console_error!("Failed to insert accumulated moment {}: {:?}", id, e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::practice::accumulator::*;

    #[test]
    fn test_build_synthesis_prompt_minimal() {
        let acc = SessionAccumulator::default();
        let ctx = SynthesisContext {
            session_id: "s1".into(),
            student_id: "u1".into(),
            conversation_id: "c1".into(),
            baselines: None,
            piece_context: None,
            student_memory: None,
            total_chunks: 0,
            session_duration_ms: 0,
        };
        let prompt = build_synthesis_prompt(&acc, &ctx);
        assert_eq!(prompt["chunks_processed"], 0);
        assert!(prompt["top_moments"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_build_synthesis_prompt_with_data() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(AccumulatedMoment {
            chunk_index: 5,
            dimension: "dynamics".into(),
            score: 0.72,
            baseline: 0.55,
            deviation: 0.17,
            is_positive: true,
            reasoning: "improved dynamics".into(),
            bar_range: Some((12, 16)),
            analysis_tier: 1,
            timestamp_ms: 75000,
            llm_analysis: None,
        });
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Running,
            chunk_index: 4,
            timestamp_ms: 60000,
            dwell_ms: 60000,
        });

        let ctx = SynthesisContext {
            session_id: "s1".into(),
            student_id: "u1".into(),
            conversation_id: "c1".into(),
            baselines: None,
            piece_context: Some(serde_json::json!({"composer": "Bach", "title": "Prelude in C"})),
            student_memory: Some("Working on dynamics for 3 sessions".into()),
            total_chunks: 20,
            session_duration_ms: 300_000,
        };

        let prompt = build_synthesis_prompt(&acc, &ctx);
        assert_eq!(prompt["chunks_processed"], 20);
        assert_eq!(prompt["piece"]["composer"], "Bach");
        assert!(!prompt["top_moments"].as_array().unwrap().is_empty());
        assert!(prompt["student_memory"].is_string());
    }
}
```

- [ ] **Step 3: Add module to mod.rs**

Add `pub mod synthesis;` to `apps/api/src/practice/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cd apps/api && cargo test synthesis -- --nocapture`
Expected: Both tests pass (build_synthesis_prompt_minimal, build_synthesis_prompt_with_data)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/synthesis.rs apps/api/src/practice/mod.rs apps/api/src/services/prompts.rs
git commit -m "feat: add synthesis module (prompt builder, LLM call, D1 persistence)"
```

---

### Task 4: Update practice_mode.rs (DrillingRecord + mode_context)

**Files:**
- Modify: `apps/api/src/practice/practice_mode.rs`

- [ ] **Step 1: Add final_scores snapshot to ModeDetector**

In `practice_mode.rs`, the `DrillingPassage` struct (line 31-35) only has `first_scores`. We need to capture `final_scores` on drilling exit. Add a method to `ModeDetector` that returns a `DrillingRecord` when exiting drilling mode.

Add after the existing `observation_policy()` method (line 255):

```rust
/// Return mode metadata for synthesis context (replaces observation_policy for non-throttle uses).
pub fn mode_context(&self) -> ModeContext {
    ModeContext {
        mode: self.mode,
        comparative: matches!(self.mode, PracticeMode::Drilling),
        entered_at_ms: self.entered_at_ms,
        chunk_count: self.chunk_count,
    }
}

/// If we just exited drilling mode, return the completed DrillingRecord
/// with final_scores captured from the current chunk's scores.
pub fn take_completed_drilling(&mut self, current_scores: [f64; 6], current_chunk: usize) -> Option<crate::practice::accumulator::DrillingRecord> {
    self.drilling_passage.take().map(|dp| {
        crate::practice::accumulator::DrillingRecord {
            bar_range: dp.bar_range,
            repetition_count: dp.repetition_count,
            first_scores: dp.first_scores,
            final_scores: current_scores,
            started_at_chunk: current_chunk.saturating_sub(dp.repetition_count * 15), // approximate
            ended_at_chunk: current_chunk,
        }
    })
}
```

Add the `ModeContext` struct near `ObservationPolicy`:

```rust
pub struct ModeContext {
    pub mode: PracticeMode,
    pub comparative: bool,
    pub entered_at_ms: u64,
    pub chunk_count: usize,
}
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd apps/api && cargo test practice_mode -- --nocapture`
Expected: All existing mode detector tests pass

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/practice/practice_mode.rs
git commit -m "feat: add DrillingRecord final_scores snapshot and mode_context() to ModeDetector"
```

---

### Task 5: Wire Accumulation into session.rs (Replace Observation Pipeline)

**Files:**
- Modify: `apps/api/src/practice/session.rs`

This is the largest task. It replaces the observation buffer with the accumulator, wires accumulation into the chunk pipeline, updates `ensure_session_identity()` to reload the accumulator, and removes the per-chunk LLM path for production sessions.

- [ ] **Step 1: Update SessionState struct (lines 87-121)**

Replace:
- `observations: Vec<ObservationRecord>` -> `accumulator: SessionAccumulator`
- Remove `last_observation_at: Option<u64>`
- Update `Default` impl (lines 123-151)

Add import at top: `use crate::practice::accumulator::{SessionAccumulator, AccumulatedMoment, ModeTransitionRecord, TimelineEvent};`

- [ ] **Step 2: Expand ensure_session_identity() to ensure_session_state() (lines 369-396)**

After reloading identity fields, also reload the accumulator:

```rust
if let Ok(Some(acc_json)) = storage.get::<String>("accumulator").await {
    if let Ok(acc) = serde_json::from_str::<SessionAccumulator>(&acc_json) {
        s.accumulator = acc;
        console_log!("DO accumulator reloaded: {} moments, {} transitions",
            s.accumulator.teaching_moments.len(),
            s.accumulator.mode_transitions.len());
    }
}
```

Rename the function from `ensure_session_identity` to `ensure_session_state`. Update all call sites (lines 203, 223, 859, 1054 approximately).

- [ ] **Step 3: Wire accumulation into handle_chunk_ready()**

After `process_muq_result()` and `process_amt_result()` (where STOP classification, mode detection, and teaching moment selection already happen), replace the `try_generate_observation()` call with accumulation logic:

```rust
// After mode detector update and teaching moment selection:

// Guard: eval sessions use the old per-chunk observation path
if self.inner.borrow().is_eval_session {
    self.try_generate_observation(ws, chunk_analysis.as_ref(), &scores_array).await;
} else {
    // Accumulate timeline event
    let has_audio = !perf_notes.is_empty(); // AMT detected notes
    let timestamp_ms = js_sys::Date::now() as u64;
    {
        let mut s = self.inner.borrow_mut();
        s.accumulator.accumulate_timeline_event(TimelineEvent {
            chunk_index: index,
            timestamp_ms,
            has_audio,
        });
    }

    // Accumulate mode transitions.
    // mode_detector.update() is already called earlier in handle_chunk_ready (line 533),
    // which returns Vec<ModeTransition>. Capture that return value and accumulate here.
    // The existing code at line 533: `let mode_transitions = self.inner.borrow_mut().mode_detector.update(&chunk_signal);`
    // After the existing WS broadcast loop (lines 536-544), add:
    for transition in &mode_transitions {
        // ModeTransition.mode is the NEW (target) mode. Derive `from` by looking at
        // the last recorded transition's `to`, or default to Warming if none yet.
        let from_mode = {
            let s = self.inner.borrow();
            s.accumulator.mode_transitions.last()
                .map(|t| t.to)
                .unwrap_or(PracticeMode::Warming)
        };
        let dwell_ms = {
            let s = self.inner.borrow();
            timestamp_ms.saturating_sub(s.mode_detector.entered_at_ms)
        };
        self.inner.borrow_mut().accumulator.accumulate_mode_transition(ModeTransitionRecord {
            from: from_mode,
            to: transition.mode,
            chunk_index: transition.chunk_index,
            timestamp_ms,
            dwell_ms,
        });
    }

    // Accumulate teaching moment if STOP triggered.
    // IMPORTANT: Use the stop_result already computed by try_generate_observation's call
    // site (or from process_muq_result). Do NOT call stop::classify() again.
    // The existing code calls stop::classify in try_generate_observation (line 827).
    // Since we're replacing that call, use the same pattern:
    let stop_result = stop::classify(&scores_array); // single call, replaces the one in try_generate_observation
    if stop_result.triggered {
        if let Some(baselines) = self.inner.borrow().baselines.clone() {
            let recent_obs: Vec<RecentObservation> = self.inner.borrow()
                .accumulator.teaching_moments.iter().rev().take(3)
                .map(|m| RecentObservation { dimension: m.dimension.clone() })
                .collect();
            let scored_chunks = self.inner.borrow().scored_chunks.clone();

            if let Some(moment) = crate::services::teaching_moments::select_teaching_moment(
                &scored_chunks, &baselines, &recent_obs,
            ) {
                let bar_range = chunk_analysis.as_ref().and_then(|ca| ca.bar_range.clone());
                let tier = chunk_analysis.as_ref().map(|ca| ca.tier).unwrap_or(3);

                self.inner.borrow_mut().accumulator.accumulate_moment(AccumulatedMoment {
                    chunk_index: moment.chunk_index,
                    dimension: moment.dimension.clone(),
                    score: moment.score,
                    baseline: moment.baseline,
                    deviation: moment.deviation,
                    is_positive: moment.is_positive,
                    reasoning: moment.reasoning.clone(),
                    bar_range,
                    analysis_tier: tier,
                    timestamp_ms,
                    llm_analysis: None,
                });
            }
        }
    }

    // Accumulate mode transition if one occurred
    // (mode_detector.update() already called above -- check if transitions were emitted)
    // This depends on how transitions are tracked -- wire into the existing mode transition WS push code.

    // Accumulate drilling record if exiting drilling
    {
        let mut s = self.inner.borrow_mut();
        if let Some(dr) = s.mode_detector.take_completed_drilling(scores_array, index) {
            s.accumulator.accumulate_drilling_record(dr);
        }
    }

    // Persist accumulator to DO storage
    let acc_json = serde_json::to_string(&self.inner.borrow().accumulator).unwrap_or_default();
    let _ = self.state.storage().put("accumulator", &acc_json).await;
}
```

- [ ] **Step 4: Update end_session handler to call synthesize before finalize**

Modify the `end_session` handler (line 270-281) to call synthesis between waiting for in-flight chunks and finalizing:

```rust
"end_session" => {
    let in_flight = {
        let mut s = self.inner.borrow_mut();
        s.session_ending = true;
        s.chunks_in_flight
    };
    if in_flight == 0 {
        // Synthesize before finalizing (keeps WS open for synthesis push)
        if !self.inner.borrow().is_eval_session {
            self.run_synthesis_and_persist(&ws).await;
        }
        self.finalize_session(Some(&ws)).await;
    } else {
        console_log!("end_session received, waiting for {} in-flight chunks", in_flight);
    }
}
```

Similarly, in the `should_finalize` block after chunk processing (line 258-266):
```rust
if should_finalize {
    console_log!("Last in-flight chunk completed, finalizing session");
    if !self.inner.borrow().is_eval_session {
        self.run_synthesis_and_persist(&ws).await;
    }
    self.finalize_session(Some(&ws)).await;
}
```

- [ ] **Step 5: Implement run_synthesis_and_persist()**

New method on `PracticeSession`:

```rust
async fn run_synthesis_and_persist(&self, ws: &WebSocket) {
    self.ensure_session_state().await;

    let (acc, ctx) = {
        let s = self.inner.borrow();
        if !s.accumulator.has_teaching_content() && s.accumulator.timeline.iter().all(|t| !t.has_audio) {
            console_log!("No teaching content and no audio detected, skipping synthesis");
            return;
        }

        let session_duration_ms = s.accumulator.timeline.last()
            .map(|t| t.timestamp_ms)
            .unwrap_or(0)
            .saturating_sub(s.accumulator.timeline.first().map(|t| t.timestamp_ms).unwrap_or(0));

        let ctx = crate::practice::synthesis::SynthesisContext {
            session_id: s.session_id.clone(),
            student_id: s.student_id.clone(),
            conversation_id: s.conversation_id.clone().unwrap_or_default(),
            baselines: s.baselines.clone(),
            piece_context: s.score_context.as_ref().map(|sc| serde_json::json!({
                "composer": sc.composer,
                "title": sc.title,
                "piece_id": sc.piece_id,
            })),
            student_memory: None, // TODO: load from memory system
            total_chunks: s.scored_chunks.len(),
            session_duration_ms,
        };
        (s.accumulator.clone(), ctx)
    };

    // Check conversation_id
    if ctx.conversation_id.is_empty() {
        console_error!("No conversation_id at synthesis time -- cannot persist");
        return;
    }

    let acc_size = format!("moments={}, transitions={}, drilling={}",
        acc.teaching_moments.len(), acc.mode_transitions.len(), acc.drilling_records.len());
    console_log!("Starting synthesis: {}", acc_size);

    // Build prompt and call LLM
    let prompt = crate::practice::synthesis::build_synthesis_prompt(&acc, &ctx);
    let result = crate::practice::synthesis::call_synthesis_llm(&self.env, &prompt).await;

    // Send to client via WS
    let synthesis_event = serde_json::json!({
        "type": "synthesis",
        "text": result.text,
        "is_fallback": result.is_fallback,
    });
    let _ = ws.send_with_str(&synthesis_event.to_string());

    // Persist synthesis message to D1
    match crate::practice::synthesis::persist_synthesis_message(
        &self.env, &ctx.conversation_id, &ctx.session_id, &result.text
    ).await {
        Ok(msg_id) => console_log!("Synthesis message persisted: {}", msg_id),
        Err(e) => console_error!("Failed to persist synthesis message: {}", e),
    }

    // Persist accumulated moments to observations table
    if let Err(e) = crate::practice::synthesis::persist_accumulated_moments(
        &self.env, &ctx.student_id, &ctx.session_id, &acc.teaching_moments
    ).await {
        console_error!("Failed to persist accumulated moments: {}", e);
    }
}
```

- [ ] **Step 6: Simplify finalize_session()**

Remove `generate_session_summary()` call and per-observation summary logic from `finalize_session()` (lines 1634-1764). The synthesis already happened before finalize. Keep:
- Memory synthesis (should_synthesize + run_synthesis)
- Observation count update (using accumulator moments count)
- Session_end message persist
- WebSocket close
- If synthesis didn't happen (e.g., alarm/disconnect path), persist accumulator to D1 with `needs_synthesis`

- [ ] **Step 7: Remove dead code**

Remove or gate behind `is_eval_session`:
- `try_generate_observation()` (lines 821-849) -- keep for eval only
- `generate_observation()` (lines 1052-1273) -- keep for eval only
- `mode_throttle_allows()` (lines 1004-1012) -- keep for eval only
- `ObservationRecord` struct (lines 74-85) -- keep for eval only
- `persist_observations()` (lines 1766-1813) -- keep for eval only

- [ ] **Step 8: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 9: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: wire accumulation into chunk pipeline, replace per-observation LLM with post-session synthesis"
```

---

### Task 6: Deferred Synthesis Endpoint

**Files:**
- Modify: `apps/api/src/server.rs` -- add POST /api/practice/synthesize route

- [ ] **Step 1: Add routes to server.rs**

After the existing practice routes (around line 160), add both GET (check) and POST (trigger) routes. Follow the existing pattern: `http::HeaderMap` for headers, `req.uri().query()` for query params, and `into_worker_response(with_cors(...))` for response wrapping.

```rust
// GET /api/practice/needs-synthesis?conversation_id=X -- check for sessions needing synthesis
if path == "/api/practice/needs-synthesis" && method == http::Method::GET {
    let headers = req.headers().clone();
    let query_string = req.uri().query().map(|q| q.to_string()).unwrap_or_default();
    let conv_id = query_string.split('&')
        .find_map(|pair| pair.strip_prefix("conversation_id="))
        .unwrap_or("");
    return into_worker_response(with_cors(
        crate::practice::synthesis::handle_check_needs_synthesis(&env, &headers, conv_id).await,
        origin.as_deref(),
    )).await;
}

// POST /api/practice/synthesize -- trigger deferred synthesis for a specific session
if path == "/api/practice/synthesize" && method == http::Method::POST {
    let headers = req.headers().clone();
    let body = req.into_body().collect().await
        .map(|b| b.to_bytes().to_vec()).unwrap_or_default();
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    return into_worker_response(with_cors(
        crate::practice::synthesis::handle_deferred_synthesis(&env, &headers, &body_json).await,
        origin.as_deref(),
    )).await;
}
```
```

- [ ] **Step 2: Implement handle_deferred_synthesis in synthesis.rs**

Add to `synthesis.rs`:

```rust
/// HTTP endpoint for deferred synthesis recovery.
/// Called by the web client when it detects a session with needs_synthesis=1.
pub async fn handle_deferred_synthesis(
    env: &Env,
    headers: &http::HeaderMap,
    body: &serde_json::Value,
) -> http::Response<axum::body::Body> {
    // Auth: verify JWT and extract student_id (same pattern as ask.rs line 248)
    let student_id = crate::auth::verify_auth_header(headers, env)
        .map_err(|e| format!("Auth failed: {}", e))?;

    let session_id = body.get("session_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing session_id")?;

    let db = env.d1("DB").map_err(|e| format!("D1: {:?}", e))?;

    // Load session and verify ownership
    let row = db.prepare(
        "SELECT student_id, conversation_id, accumulator_json, needs_synthesis FROM sessions WHERE id = ?1"
    )
    .bind(&[JsValue::from_str(session_id)])
    .map_err(|e| format!("Bind: {:?}", e))?
    .first::<serde_json::Value>(None)
    .await
    .map_err(|e| format!("Query: {:?}", e))?;

    let row = row.ok_or("Session not found")?;

    // Verify student owns this session
    let row_student = row.get("student_id").and_then(|v| v.as_str()).unwrap_or("");
    if row_student != student_id {
        return Err("Unauthorized: session belongs to a different student".into());
    }

    let needs = row.get("needs_synthesis").and_then(|v| v.as_i64()).unwrap_or(0);
    if needs != 1 {
        return worker::Response::ok("No synthesis needed")
            .map_err(|e| format!("{:?}", e));
    }

    let acc_json = row.get("accumulator_json")
        .and_then(|v| v.as_str())
        .ok_or("No accumulator data available")?;

    let acc: SessionAccumulator = serde_json::from_str(acc_json)
        .map_err(|e| {
            // Clear the flag so we don't retry forever
            let _ = clear_needs_synthesis(env, session_id);
            format!("Deserialization failed: {:?}", e)
        })?;

    // Load baselines for this student.
    // NOTE: load_baselines is an instance method on PracticeSession (session.rs:1477).
    // For the deferred path, inline the D1 query (same SQL as session.rs:1488-1493).
    let baselines = load_baselines_from_d1(env, &student_id).await.ok();

    // conversation_id already loaded from the same row above
    let conversation_id = row.get("conversation_id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or("No conversation_id for session")?;

    let ctx = SynthesisContext {
        session_id: session_id.to_string(),
        student_id: student_id.clone(),
        conversation_id: conversation_id.clone(),
        baselines,
        piece_context: None, // Not available in deferred path
        student_memory: None,
        total_chunks: acc.timeline.len(),
        session_duration_ms: acc.timeline.last().map(|t| t.timestamp_ms).unwrap_or(0)
            .saturating_sub(acc.timeline.first().map(|t| t.timestamp_ms).unwrap_or(0)),
    };

    let prompt = build_synthesis_prompt(&acc, &ctx);
    let result = call_synthesis_llm(env, &prompt).await;

    // Persist
    persist_synthesis_message(env, &conversation_id, session_id, &result.text).await?;
    persist_accumulated_moments(env, &student_id, session_id, &acc.teaching_moments).await?;

    // Clear the flag
    clear_needs_synthesis(env, session_id).await?;

    let resp = serde_json::json!({
        "status": "synthesized",
        "session_id": session_id,
        "is_fallback": result.is_fallback,
    });

    worker::Response::from_json(&resp).map_err(|e| format!("{:?}", e))
}

/// Load baselines from D1 directly (free function for use outside DO context).
/// Same SQL as PracticeSession::load_baselines (session.rs:1488-1493).
pub async fn load_baselines_from_d1(
    env: &Env,
    student_id: &str,
) -> Result<StudentBaselines, String> {
    let db = env.d1("DB").map_err(|e| format!("D1: {:?}", e))?;
    let stmt = db.prepare(
        "SELECT dimension, AVG(dimension_score) as avg_score \
         FROM observations WHERE student_id = ?1 \
         AND created_at > datetime('now', '-30 days') \
         GROUP BY dimension"
    )
    .bind(&[JsValue::from_str(student_id)])
    .map_err(|e| format!("Bind: {:?}", e))?;

    let results = stmt.all().await.map_err(|e| format!("Query: {:?}", e))?;
    let rows = results.results::<serde_json::Value>().unwrap_or_default();

    let defaults = crate::services::stop::SCALER_MEAN;
    let mut baselines = StudentBaselines {
        dynamics: defaults[0], timing: defaults[1], pedaling: defaults[2],
        articulation: defaults[3], phrasing: defaults[4], interpretation: defaults[5],
    };

    for row in &rows {
        let dim = row.get("dimension").and_then(|v| v.as_str()).unwrap_or("");
        let avg = row.get("avg_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        match dim {
            "dynamics" => baselines.dynamics = avg,
            "timing" => baselines.timing = avg,
            "pedaling" => baselines.pedaling = avg,
            "articulation" => baselines.articulation = avg,
            "phrasing" => baselines.phrasing = avg,
            "interpretation" => baselines.interpretation = avg,
            _ => {}
        }
    }
    Ok(baselines)
}

/// GET handler: check if a conversation has sessions needing synthesis.
/// Returns list of session_ids with needs_synthesis=1.
pub async fn handle_check_needs_synthesis(
    env: &Env,
    headers: &http::HeaderMap,
    conversation_id: &str,
) -> http::Response<axum::body::Body> {
    let student_id = crate::auth::verify_auth_header(headers, env)
        .map_err(|e| format!("Auth: {}", e))?;
    let db = env.d1("DB").map_err(|e| format!("D1: {:?}", e))?;
    let stmt = db.prepare(
        "SELECT id FROM sessions WHERE conversation_id = ?1 AND student_id = ?2 AND needs_synthesis = 1"
    )
    .bind(&[JsValue::from_str(conversation_id), JsValue::from_str(&student_id)])
    .map_err(|e| format!("Bind: {:?}", e))?;

    let results = stmt.all().await.map_err(|e| format!("Query: {:?}", e))?;
    let rows = results.results::<serde_json::Value>().unwrap_or_default();
    let session_ids: Vec<&str> = rows.iter()
        .filter_map(|r| r.get("id").and_then(|v| v.as_str()))
        .collect();

    worker::Response::from_json(&serde_json::json!({ "session_ids": session_ids }))
        .map_err(|e| format!("{:?}", e))
}

async fn clear_needs_synthesis(env: &Env, session_id: &str) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1: {:?}", e))?;
    db.prepare("UPDATE sessions SET needs_synthesis = 0 WHERE id = ?1")
        .bind(&[JsValue::from_str(session_id)])
        .map_err(|e| format!("Bind: {:?}", e))?
        .run().await
        .map_err(|e| format!("Run: {:?}", e))?;
    Ok(())
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/server.rs apps/api/src/practice/synthesis.rs
git commit -m "feat: add POST /api/practice/synthesize endpoint for deferred recovery"
```

---

### Task 7: Web Client Changes

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`
- Modify: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Update usePracticeSession.ts WS message handler**

In the `case "observation"` handler (line 203), wrap it to only process for eval sessions or remove it entirely. Add a new `case "synthesis"` handler:

```typescript
case "synthesis": {
    console.log(`[Practice] Session synthesis received (fallback=${data.is_fallback})`);
    setSummary(data.text);
    setState("summarizing");
    // The synthesis text will be added as a chat message by the parent component
    onSynthesis?.(data.text, data.is_fallback);
    break;
}
```

Remove the `observations` state and `observationMessages` from the return value. The `session_summary` handler (line 216) is replaced by the `synthesis` handler.

- [ ] **Step 2: Update ListeningMode.tsx**

Remove the `observations` prop and the `visibleObservations` display logic (line 134+). The listening mode overlay now shows only the waveform, timer, and chunk progress -- no observation toasts.

- [ ] **Step 3: Verify web build**

Run: `cd apps/web && bun run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/components/ListeningMode.tsx
git commit -m "feat: replace observation toasts with post-session synthesis in web client"
```

---

### Task 8: Integration Testing

**Files:**
- Manual testing with `just dev`

- [ ] **Step 1: Start local dev environment**

Run: `just dev-light` (uses production HF endpoints, starts API + Web)

- [ ] **Step 2: Test normal session flow**

1. Open `http://localhost:3000`, sign in
2. Enter listening mode, play piano for 30+ seconds
3. Exit listening mode
4. Verify: synthesis message appears in chat (not individual observation toasts)
5. Verify: D1 has a message with `message_type='synthesis'`

- [ ] **Step 3: Test short session (5 seconds)**

1. Enter listening mode, play 1-2 chunks
2. Exit listening mode
3. Verify: teacher still responds (no "play more" canned message)

- [ ] **Step 4: Test eval session (preserved per-chunk path)**

Run: Use the eval runner if available, or manually send `eval_chunk` messages in dev mode to verify per-chunk observations still work for eval sessions.

- [ ] **Step 5: Verify no observation toasts during listening**

1. Enter listening mode
2. Play for 60+ seconds (enough for STOP to trigger)
3. Verify: NO observation toasts appear during playing
4. Verify: console logs show "accumulating moment" instead of "generating observation"

- [ ] **Step 6: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration test fixes for session synthesis"
```

---

### Task 9: Finalize_session Safety Net (Deferred Persistence)

**Files:**
- Modify: `apps/api/src/practice/session.rs` -- update finalize_session and alarm handler

- [ ] **Step 1: Add synthesis_completed flag to SessionState**

Add `synthesis_completed: bool` to `SessionState` struct (default `false`). Set to `true` at the end of `run_synthesis_and_persist()`.

- [ ] **Step 2: Update finalize_session for unsynthesized sessions**

In `finalize_session()`, after the synthesis path, add the safety net for disconnect/alarm cases:

```rust
// If synthesis hasn't happened (no WS to receive it), persist accumulator for deferred recovery
let (has_synthesis, acc, session_id, student_id, conv_id) = {
    let s = self.inner.borrow();
    (!s.synthesis_completed && s.accumulator.has_teaching_content(),
     s.accumulator.clone(),
     s.session_id.clone(),
     s.student_id.clone(),
     s.conversation_id.clone())
};

if has_synthesis {
    if let Some(ref conv_id) = conv_id {
        let acc_json = serde_json::to_string(&acc).unwrap_or_default();
        if let Ok(db) = self.env.d1("DB") {
            if let Ok(stmt) = db.prepare(
                "UPDATE sessions SET accumulator_json = ?1, needs_synthesis = 1 WHERE id = ?2"
            ).bind(&[
                JsValue::from_str(&acc_json),
                JsValue::from_str(&session_id),
            ]) {
                if let Err(e) = stmt.run().await {
                    console_error!("Failed to persist accumulator for deferred synthesis: {:?}", e);
                }
            }
        }
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd apps/api && cargo check`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: add deferred synthesis persistence in finalize_session safety net"
```

---

### Task 10: Web Client Deferred Synthesis Check

**Files:**
- Modify: `apps/web/src/lib/practice-api.ts` or `apps/web/src/lib/api.ts`
- Modify: `apps/web/src/components/AppChat.tsx` -- check on conversation load

- [ ] **Step 1: Add API methods for checking and triggering deferred synthesis**

```typescript
// In api.ts or practice-api.ts

/** Check if any sessions in this conversation need deferred synthesis. */
export async function checkNeedsSynthesis(conversationId: string): Promise<string[]> {
    const resp = await fetch(
        `${API_BASE}/api/practice/needs-synthesis?conversation_id=${encodeURIComponent(conversationId)}`,
        { credentials: 'include' }
    );
    if (!resp.ok) return [];
    const data = await resp.json() as { session_ids: string[] };
    return data.session_ids ?? [];
}

/** Trigger deferred synthesis for a specific session. */
export async function triggerDeferredSynthesis(sessionId: string): Promise<boolean> {
    const resp = await fetch(`${API_BASE}/api/practice/synthesize`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    });
    return resp.ok;
}
```

- [ ] **Step 2: Wire into conversation load in AppChat.tsx**

On conversation load, check for `needs_synthesis` sessions and trigger deferred synthesis. This can be a `useEffect` that runs once when conversation messages load -- if the last message is a `session_end` without a following `synthesis`, call the endpoint.

- [ ] **Step 3: Verify web build**

Run: `cd apps/web && bun run build`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/lib/api.ts apps/web/src/components/AppChat.tsx
git commit -m "feat: add deferred synthesis check on conversation load"
```
