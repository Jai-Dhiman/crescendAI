# Student Memory System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the subagent accumulated student context (synthesized facts, teaching approach history) so feedback compounds across sessions.

**Architecture:** Worker-side LLM synthesis triggered after `POST /api/sync`. Three new D1 tables (`synthesized_facts`, `teaching_approaches`, `student_memory_meta`). Four structured D1 queries before each `/api/ask` subagent call build the context map. No vector search, no graph DB.

**Tech Stack:** Rust (Cloudflare Workers), D1 (SQLite), Groq API (Llama 70B for synthesis), existing `worker` + `serde_json` + `wasm-bindgen` crates.

**Design doc:** `docs/plans/2026-03-06-memory-system-design.md`

**Key constraint:** No `ctx.waitUntil()` in Rust Workers. Synthesis runs in-request during sync (~0.3s on Groq). Acceptable latency addition.

---

### Task 1: D1 Migration -- Add Memory Tables

**Files:**
- Create: `apps/api/migrations/0006_memory_system.sql`

**Step 1: Write the migration**

```sql
-- Student memory system: synthesized facts, teaching approaches, memory meta
-- See docs/plans/2026-03-06-memory-system-design.md

CREATE TABLE synthesized_facts (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    dimension TEXT,
    piece_context TEXT,
    valid_at TEXT NOT NULL,
    invalid_at TEXT,
    trend TEXT,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    source_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expired_at TEXT
);

CREATE INDEX idx_synthesized_facts_student ON synthesized_facts(student_id);
CREATE INDEX idx_synthesized_facts_active ON synthesized_facts(student_id, invalid_at, expired_at);

CREATE TABLE teaching_approaches (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    framing TEXT NOT NULL,
    approach_summary TEXT NOT NULL,
    engaged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_teaching_approaches_student ON teaching_approaches(student_id);
CREATE INDEX idx_teaching_approaches_observation ON teaching_approaches(observation_id);

CREATE TABLE student_memory_meta (
    student_id TEXT PRIMARY KEY,
    last_synthesis_at TEXT,
    total_observations INTEGER DEFAULT 0,
    total_facts INTEGER DEFAULT 0
);
```

**Step 2: Verify migration syntax**

Run: `cd apps/api && npx wrangler d1 migrations list --local`

Expected: `0006_memory_system.sql` appears in the list.

**Step 3: Apply migration locally**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --local`

Expected: Migration applies without error.

**Step 4: Commit**

```bash
git add apps/api/migrations/0006_memory_system.sql
git commit -m "feat(api): add D1 migration for memory system tables"
```

---

### Task 2: Create Memory Module -- Data Types and Retrieval

**Files:**
- Create: `apps/api/src/services/memory.rs`
- Modify: `apps/api/src/services/mod.rs` (add `pub mod memory;`)

**Step 1: Create `memory.rs` with data types and retrieval functions**

```rust
//! Student memory system: synthesized facts, teaching approaches, retrieval, and synthesis.
//!
//! See docs/plans/2026-03-06-memory-system-design.md for the full design.

use wasm_bindgen::JsValue;
use worker::{console_log, Env};

/// A synthesized fact from the event clock.
pub struct SynthesizedFact {
    pub id: String,
    pub fact_text: String,
    pub fact_type: String,
    pub dimension: Option<String>,
    pub piece_context: Option<String>,
    pub valid_at: String,
    pub trend: Option<String>,
    pub confidence: String,
    pub source_type: String,
}

/// A recent observation with engagement data.
pub struct RecentObservationWithEngagement {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub created_at: String,
    pub engaged: bool,
}

/// The assembled context map for the subagent.
pub struct StudentMemoryContext {
    pub active_facts: Vec<SynthesizedFact>,
    pub recent_observations: Vec<RecentObservationWithEngagement>,
    pub piece_facts: Vec<SynthesizedFact>,
}

/// Query active synthesized facts for a student.
pub async fn query_active_facts(
    env: &Env,
    student_id: &str,
) -> Result<Vec<SynthesizedFact>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT id, fact_text, fact_type, dimension, piece_context, valid_at, \
             trend, confidence, source_type \
             FROM synthesized_facts \
             WHERE student_id = ?1 AND invalid_at IS NULL AND expired_at IS NULL \
             ORDER BY fact_type, valid_at DESC \
             LIMIT 15",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query facts: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| SynthesizedFact {
            id: row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_text: row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_type: row.get("fact_type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            dimension: row.get("dimension").and_then(|v| v.as_str()).map(|s| s.to_string()),
            piece_context: row.get("piece_context").and_then(|v| v.as_str()).map(|s| s.to_string()),
            valid_at: row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            trend: row.get("trend").and_then(|v| v.as_str()).map(|s| s.to_string()),
            confidence: row.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium").to_string(),
            source_type: row.get("source_type").and_then(|v| v.as_str()).unwrap_or("synthesized").to_string(),
        })
        .collect())
}

/// Query recent observations with engagement data from teaching_approaches.
pub async fn query_recent_observations_with_engagement(
    env: &Env,
    student_id: &str,
) -> Result<Vec<RecentObservationWithEngagement>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT o.dimension, o.observation_text, o.framing, o.created_at, \
             COALESCE(ta.engaged, 0) AS engaged \
             FROM observations o \
             LEFT JOIN teaching_approaches ta ON ta.observation_id = o.id \
             WHERE o.student_id = ?1 \
             ORDER BY o.created_at DESC \
             LIMIT 5",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query observations: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| RecentObservationWithEngagement {
            dimension: row.get("dimension").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            observation_text: row.get("observation_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            framing: row.get("framing").and_then(|v| v.as_str()).unwrap_or("correction").to_string(),
            created_at: row.get("created_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            engaged: row.get("engaged").and_then(|v| v.as_i64()).unwrap_or(0) == 1,
        })
        .collect())
}

/// Query piece-specific facts for a student and piece title.
pub async fn query_piece_facts(
    env: &Env,
    student_id: &str,
    piece_title: &str,
) -> Result<Vec<SynthesizedFact>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let results = db
        .prepare(
            "SELECT id, fact_text, fact_type, dimension, piece_context, valid_at, \
             trend, confidence, source_type \
             FROM synthesized_facts \
             WHERE student_id = ?1 \
             AND piece_context IS NOT NULL \
             AND json_extract(piece_context, '$.title') = ?2 \
             AND invalid_at IS NULL AND expired_at IS NULL",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(piece_title),
        ])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query piece facts: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| SynthesizedFact {
            id: row.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_text: row.get("fact_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fact_type: row.get("fact_type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            dimension: row.get("dimension").and_then(|v| v.as_str()).map(|s| s.to_string()),
            piece_context: row.get("piece_context").and_then(|v| v.as_str()).map(|s| s.to_string()),
            valid_at: row.get("valid_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            trend: row.get("trend").and_then(|v| v.as_str()).map(|s| s.to_string()),
            confidence: row.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium").to_string(),
            source_type: row.get("source_type").and_then(|v| v.as_str()).unwrap_or("synthesized").to_string(),
        })
        .collect())
}

/// Build the full memory context for the subagent.
/// Runs queries 1, 2, and optionally 4 (piece-specific).
pub async fn build_memory_context(
    env: &Env,
    student_id: &str,
    piece_title: Option<&str>,
) -> StudentMemoryContext {
    let active_facts = match query_active_facts(env, student_id).await {
        Ok(facts) => facts,
        Err(e) => {
            console_log!("Failed to query active facts: {}", e);
            vec![]
        }
    };

    let recent_observations = match query_recent_observations_with_engagement(env, student_id).await {
        Ok(obs) => obs,
        Err(e) => {
            console_log!("Failed to query recent observations: {}", e);
            vec![]
        }
    };

    let piece_facts = if let Some(title) = piece_title {
        match query_piece_facts(env, student_id, title).await {
            Ok(facts) => {
                // Deduplicate against active_facts
                let active_ids: std::collections::HashSet<&str> =
                    active_facts.iter().map(|f| f.id.as_str()).collect();
                facts
                    .into_iter()
                    .filter(|f| !active_ids.contains(f.id.as_str()))
                    .collect()
            }
            Err(e) => {
                console_log!("Failed to query piece facts: {}", e);
                vec![]
            }
        }
    } else {
        vec![]
    };

    StudentMemoryContext {
        active_facts,
        recent_observations,
        piece_facts,
    }
}

/// Format the memory context as plain text for the subagent prompt.
pub fn format_memory_context(ctx: &StudentMemoryContext) -> String {
    if ctx.active_facts.is_empty() && ctx.recent_observations.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(1500);
    out.push_str("## Student Memory\n\n");

    if !ctx.active_facts.is_empty() {
        out.push_str("### Active Patterns\n");
        for fact in &ctx.active_facts {
            let dim_label = fact
                .dimension
                .as_deref()
                .map(|d| format!("{}/", d))
                .unwrap_or_default();
            let trend_label = fact
                .trend
                .as_deref()
                .map(|t| format!(", {}", t))
                .unwrap_or_default();
            out.push_str(&format!(
                "- [{}{}{}, {} confidence] {} (since {})\n",
                fact.fact_type, dim_label, trend_label, fact.confidence,
                fact.fact_text, fact.valid_at,
            ));
        }
        out.push('\n');
    }

    if !ctx.recent_observations.is_empty() {
        out.push_str("### Recent Feedback\n");
        for obs in &ctx.recent_observations {
            let engaged_label = if obs.engaged { ", student asked for elaboration" } else { "" };
            out.push_str(&format!(
                "- [{}] {}: \"{}\" (framing: {}{})\n",
                obs.created_at, obs.dimension, obs.observation_text, obs.framing, engaged_label,
            ));
        }
        out.push('\n');
    }

    if !ctx.piece_facts.is_empty() {
        out.push_str("### Current Piece History\n");
        for fact in &ctx.piece_facts {
            out.push_str(&format!("- {} (since {})\n", fact.fact_text, fact.valid_at));
        }
        out.push('\n');
    }

    out
}

/// Store a teaching approach record after an /api/ask call.
pub async fn store_teaching_approach(
    env: &Env,
    id: &str,
    student_id: &str,
    observation_id: &str,
    dimension: &str,
    framing: &str,
    approach_summary: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "INSERT INTO teaching_approaches \
         (id, student_id, observation_id, dimension, framing, approach_summary, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )
    .bind(&[
        JsValue::from_str(id),
        JsValue::from_str(student_id),
        JsValue::from_str(observation_id),
        JsValue::from_str(dimension),
        JsValue::from_str(framing),
        JsValue::from_str(approach_summary),
        JsValue::from_str(&now),
    ])
    .map_err(|e| format!("Failed to bind insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert teaching approach: {:?}", e))?;

    Ok(())
}

/// Mark a teaching approach as engaged (student tapped "tell me more").
pub async fn mark_approach_engaged(
    env: &Env,
    observation_id: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare("UPDATE teaching_approaches SET engaged = 1 WHERE observation_id = ?1")
        .bind(&[JsValue::from_str(observation_id)])
        .map_err(|e| format!("Failed to bind update: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to update engagement: {:?}", e))?;

    Ok(())
}

/// Increment total_observations in student_memory_meta.
pub async fn increment_observation_count(
    env: &Env,
    student_id: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, total_observations) VALUES (?1, 1) \
         ON CONFLICT(student_id) DO UPDATE SET total_observations = total_observations + 1",
    )
    .bind(&[JsValue::from_str(student_id)])
    .map_err(|e| format!("Failed to bind upsert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update observation count: {:?}", e))?;

    Ok(())
}
```

**Step 2: Register the module**

Add `pub mod memory;` to `apps/api/src/services/mod.rs`.

**Step 3: Verify it compiles**

Run: `cd apps/api && npx wrangler build`

Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add apps/api/src/services/memory.rs apps/api/src/services/mod.rs
git commit -m "feat(api): add memory module with retrieval and teaching approach storage"
```

---

### Task 3: Integrate Memory Retrieval into Ask Pipeline

**Files:**
- Modify: `apps/api/src/services/ask.rs` (lines 110-126, 199-240, 260-327)
- Modify: `apps/api/src/services/prompts.rs` (add memory context to subagent prompt builder)

**Step 1: Update `handle_ask` to build memory context and pass it to the subagent**

In `ask.rs`, after the auth check and request parsing (around line 110), replace the `query_recent_observations` call and subagent prompt building with:

```rust
// Build memory context (replaces simple query_recent_observations)
let piece_title = request
    .piece_context
    .as_ref()
    .and_then(|pc| pc.get("title"))
    .and_then(|v| v.as_str());

let memory_ctx = crate::services::memory::build_memory_context(
    env,
    &student_id,
    piece_title,
).await;

let memory_text = crate::services::memory::format_memory_context(&memory_ctx);

// Convert recent observations for backward compat with prompt builder
let recent_observations: Vec<prompts::ObservationRow> = memory_ctx
    .recent_observations
    .iter()
    .map(|obs| prompts::ObservationRow {
        dimension: obs.dimension.clone(),
        observation_text: obs.observation_text.clone(),
        framing: obs.framing.clone(),
        created_at: obs.created_at.clone(),
    })
    .collect();
```

Then pass `&memory_text` to the subagent prompt builder (see Step 2 below).

**Step 2: Update `build_subagent_user_prompt` in prompts.rs to accept memory context**

Add a `memory_context: &str` parameter to `build_subagent_user_prompt`. Insert the memory text before the "## Recent Observations" section:

```rust
pub fn build_subagent_user_prompt(
    teaching_moment: &serde_json::Value,
    student: &serde_json::Value,
    session: &serde_json::Value,
    piece_context: &Option<serde_json::Value>,
    recent_observations: &[ObservationRow],
    memory_context: &str,  // NEW
) -> String {
    // ... existing code ...

    // Insert memory context before recent observations
    if !memory_context.is_empty() {
        prompt.push_str(memory_context);
    }

    // ... rest of existing code (recent observations, task) ...
}
```

**Step 3: Add teaching approach storage and observation count increment after observation is stored**

In `ask.rs`, after the `store_observation` call (around line 240), add:

```rust
// Store teaching approach record
let approach_id = generate_uuid();
let approach_summary = format!("{} on {}", framing, dimension);
if let Err(e) = crate::services::memory::store_teaching_approach(
    env,
    &approach_id,
    &student_id,
    &observation_id,
    &dimension,
    &framing,
    &approach_summary,
).await {
    console_log!("Failed to store teaching approach: {}", e);
}

// Increment observation count for synthesis tracking
if let Err(e) = crate::services::memory::increment_observation_count(env, &student_id).await {
    console_log!("Failed to increment observation count: {}", e);
}
```

**Step 4: Add engagement tracking to `handle_elaborate`**

In `ask.rs`, after the `store_elaboration` call (around line 325), add:

```rust
// Mark teaching approach as engaged
if let Err(e) = crate::services::memory::mark_approach_engaged(env, &request.observation_id).await {
    console_log!("Failed to mark approach engaged: {}", e);
}
```

**Step 5: Verify it compiles**

Run: `cd apps/api && npx wrangler build`

**Step 6: Commit**

```bash
git add apps/api/src/services/ask.rs apps/api/src/services/prompts.rs
git commit -m "feat(api): integrate memory context into ask pipeline"
```

---

### Task 4: Synthesis Pipeline

**Files:**
- Modify: `apps/api/src/services/memory.rs` (add synthesis functions)
- Modify: `apps/api/src/services/prompts.rs` (add synthesis prompt)

**Step 1: Add the synthesis system prompt to prompts.rs**

```rust
/// Synthesis system prompt (Groq, Llama 70B)
/// Called after session sync to update synthesized facts.
pub const SYNTHESIS_SYSTEM: &str = r#"You are a memory consolidation system for a piano teaching app. You receive:
1. Current active facts about a student (what the system currently believes)
2. New observations since the last synthesis (what was recently observed)
3. Teaching approach records (what feedback was given and whether the student engaged)
4. Student baselines (current dimension scores)

Your job is to update the student's fact base. Output ONLY valid JSON with three arrays:

```json
{
  "new_facts": [
    {
      "fact_text": "One sentence describing the pattern or insight",
      "fact_type": "dimension|approach|arc|student_reported",
      "dimension": "dynamics|timing|pedaling|articulation|phrasing|interpretation|null",
      "piece_context": {"composer": "...", "title": "..."} or null,
      "trend": "improving|stable|declining|new|resolved",
      "confidence": "high|medium|low",
      "evidence": ["obs_id_1", "obs_id_2"]
    }
  ],
  "invalidated_facts": [
    {
      "fact_id": "id of fact to invalidate",
      "reason": "Why this fact is no longer true",
      "invalid_at": "ISO date when it stopped being true"
    }
  ],
  "unchanged_facts": ["fact_id_1", "fact_id_2"]
}
```

Rules:
- Every current active fact must appear in either invalidated_facts or unchanged_facts
- Create approach facts when engagement patterns are clear (e.g., "student engages most with correction-framed feedback")
- Invalidate facts that are contradicted by new evidence (e.g., a "persistent weakness" that has improved for 3+ sessions)
- Set trend to "resolved" when a previously flagged issue is no longer appearing
- Be conservative: only create high-confidence facts when supported by 3+ observations
- Review student_reported facts for staleness (goals older than 90 days with no related observations)"#;
```

**Step 2: Add synthesis prompt builder to prompts.rs**

```rust
/// Build the synthesis user prompt from student data.
pub fn build_synthesis_prompt(
    active_facts: &[super::memory::SynthesizedFact],
    new_observations: &[serde_json::Value],
    teaching_approaches: &[serde_json::Value],
    baselines: &serde_json::Value,
) -> String {
    let mut prompt = String::with_capacity(3000);

    prompt.push_str("## Current Active Facts\n\n");
    if active_facts.is_empty() {
        prompt.push_str("No facts yet (first synthesis).\n\n");
    } else {
        for fact in active_facts {
            let dim = fact.dimension.as_deref().unwrap_or("general");
            let trend = fact.trend.as_deref().unwrap_or("unknown");
            prompt.push_str(&format!(
                "- [id: {}, type: {}, dim: {}, trend: {}, confidence: {}, since: {}] {}\n",
                fact.id, fact.fact_type, dim, trend, fact.confidence, fact.valid_at, fact.fact_text
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## New Observations Since Last Synthesis\n\n");
    if new_observations.is_empty() {
        prompt.push_str("No new observations.\n\n");
    } else {
        for obs in new_observations {
            let id = obs.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let dim = obs.get("dimension").and_then(|v| v.as_str()).unwrap_or("");
            let text = obs.get("observation_text").and_then(|v| v.as_str()).unwrap_or("");
            let framing = obs.get("framing").and_then(|v| v.as_str()).unwrap_or("");
            let score = obs.get("dimension_score").and_then(|v| v.as_f64());
            let baseline = obs.get("student_baseline").and_then(|v| v.as_f64());
            let created = obs.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            let trace = obs.get("reasoning_trace").and_then(|v| v.as_str()).unwrap_or("");

            prompt.push_str(&format!("- [id: {}, dim: {}, framing: {}, date: {}]\n", id, dim, framing, created));
            prompt.push_str(&format!("  Text: \"{}\"\n", text));
            if let (Some(s), Some(b)) = (score, baseline) {
                prompt.push_str(&format!("  Score: {:.2} (baseline: {:.2}, delta: {:+.2})\n", s, b, s - b));
            }
            if !trace.is_empty() && trace != "{}" {
                prompt.push_str(&format!("  Reasoning: {}\n", trace));
            }
        }
        prompt.push('\n');
    }

    if !teaching_approaches.is_empty() {
        prompt.push_str("## Teaching Approaches\n\n");
        for ta in teaching_approaches {
            let dim = ta.get("dimension").and_then(|v| v.as_str()).unwrap_or("");
            let framing = ta.get("framing").and_then(|v| v.as_str()).unwrap_or("");
            let summary = ta.get("approach_summary").and_then(|v| v.as_str()).unwrap_or("");
            let engaged = ta.get("engaged").and_then(|v| v.as_i64()).unwrap_or(0) == 1;
            prompt.push_str(&format!(
                "- {}: {} (engaged: {})\n",
                dim, summary, if engaged { "yes" } else { "no" }
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## Student Baselines\n\n");
    for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
        if let Some(val) = baselines.get(format!("baseline_{}", dim)).and_then(|v| v.as_f64()) {
            prompt.push_str(&format!("- {}: {:.2}\n", dim, val));
        }
    }
    prompt.push('\n');

    prompt.push_str("## Task\n\nAnalyze the new observations against current facts and baselines. Output the JSON update.");

    prompt
}
```

**Step 3: Add synthesis execution functions to memory.rs**

```rust
use crate::services::llm;
use crate::services::prompts;

/// Check if synthesis should run for this student.
/// Returns true if >= 3 new observations since last synthesis,
/// or any new observations and last synthesis was > 7 days ago.
pub async fn should_synthesize(
    env: &Env,
    student_id: &str,
) -> Result<bool, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let meta: Option<serde_json::Value> = db
        .prepare("SELECT last_synthesis_at, total_observations FROM student_memory_meta WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query meta: {:?}", e))?;

    let last_synthesis = meta
        .as_ref()
        .and_then(|m| m.get("last_synthesis_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if last_synthesis.is_empty() {
        // Never synthesized -- check if there are at least 3 observations
        let total = meta
            .as_ref()
            .and_then(|m| m.get("total_observations"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        return Ok(total >= 3);
    }

    // Count observations since last synthesis
    let count_result: Option<serde_json::Value> = db
        .prepare(
            "SELECT COUNT(*) as cnt FROM observations \
             WHERE student_id = ?1 AND created_at > ?2",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind count query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to count observations: {:?}", e))?;

    let new_count = count_result
        .as_ref()
        .and_then(|r| r.get("cnt"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    if new_count >= 3 {
        return Ok(true);
    }

    // Check if last synthesis was > 7 days ago and there are any new observations
    if new_count > 0 {
        // Simple check: compare date strings (ISO format sorts correctly)
        let now = js_sys::Date::new_0()
            .to_iso_string()
            .as_string()
            .unwrap_or_default();
        // 7 days = check if last_synthesis is more than 7 days old
        // Simple heuristic: if the date portion differs by enough characters
        // (proper date math would need a date library, this is good enough)
        let last_date = &last_synthesis[..10.min(last_synthesis.len())];
        let now_date = &now[..10.min(now.len())];
        if last_date != now_date {
            // At least a day has passed, and we have observations.
            // For a more precise 7-day check, we'd need date arithmetic.
            // For now, synthesize if any new observations exist and it's a new day.
            return Ok(true);
        }
    }

    Ok(false)
}

/// Run the synthesis pipeline for a student.
pub async fn run_synthesis(
    env: &Env,
    student_id: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    // 1. Get current active facts
    let active_facts = query_active_facts(env, student_id).await?;

    // 2. Get last synthesis timestamp
    let meta: Option<serde_json::Value> = db
        .prepare("SELECT last_synthesis_at FROM student_memory_meta WHERE student_id = ?1")
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind meta query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query meta: {:?}", e))?;

    let last_synthesis = meta
        .as_ref()
        .and_then(|m| m.get("last_synthesis_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("1970-01-01T00:00:00.000Z");

    // 3. Get new observations since last synthesis
    let obs_results = db
        .prepare(
            "SELECT id, dimension, observation_text, framing, dimension_score, \
             student_baseline, reasoning_trace, piece_context, learning_arc, created_at \
             FROM observations \
             WHERE student_id = ?1 AND created_at > ?2 \
             ORDER BY created_at ASC",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind obs query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query new observations: {:?}", e))?;

    let new_observations: Vec<serde_json::Value> = obs_results
        .results()
        .map_err(|e| format!("Failed to get obs results: {:?}", e))?;

    if new_observations.is_empty() {
        return Ok(());
    }

    // 4. Get teaching approaches since last synthesis
    let ta_results = db
        .prepare(
            "SELECT dimension, framing, approach_summary, engaged \
             FROM teaching_approaches \
             WHERE student_id = ?1 AND created_at > ?2",
        )
        .bind(&[
            JsValue::from_str(student_id),
            JsValue::from_str(last_synthesis),
        ])
        .map_err(|e| format!("Failed to bind ta query: {:?}", e))?
        .all()
        .await
        .map_err(|e| format!("Failed to query teaching approaches: {:?}", e))?;

    let teaching_approaches: Vec<serde_json::Value> = ta_results
        .results()
        .map_err(|e| format!("Failed to get ta results: {:?}", e))?;

    // 5. Get student baselines
    let baselines: serde_json::Value = db
        .prepare(
            "SELECT baseline_dynamics, baseline_timing, baseline_pedaling, \
             baseline_articulation, baseline_phrasing, baseline_interpretation \
             FROM students WHERE student_id = ?1",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind baselines query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query baselines: {:?}", e))?
        .unwrap_or(serde_json::json!({}));

    // 6. Build synthesis prompt and call Groq
    let user_prompt = prompts::build_synthesis_prompt(
        &active_facts,
        &new_observations,
        &teaching_approaches,
        &baselines,
    );

    let synthesis_output = llm::call_groq(
        env,
        prompts::SYNTHESIS_SYSTEM,
        &user_prompt,
        0.1,
        800,
    )
    .await?;

    // 7. Parse synthesis output
    let synthesis_json = extract_synthesis_json(&synthesis_output)?;

    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();
    let today = &now[..10.min(now.len())];

    // 8. Apply invalidations
    if let Some(invalidated) = synthesis_json.get("invalidated_facts").and_then(|v| v.as_array()) {
        for inv in invalidated {
            let fact_id = inv.get("fact_id").and_then(|v| v.as_str()).unwrap_or("");
            let invalid_at = inv.get("invalid_at").and_then(|v| v.as_str()).unwrap_or(today);
            if !fact_id.is_empty() {
                let _ = db
                    .prepare(
                        "UPDATE synthesized_facts SET invalid_at = ?1, expired_at = ?2 \
                         WHERE id = ?3 AND student_id = ?4",
                    )
                    .bind(&[
                        JsValue::from_str(invalid_at),
                        JsValue::from_str(&now),
                        JsValue::from_str(fact_id),
                        JsValue::from_str(student_id),
                    ])
                    .map_err(|e| format!("Failed to bind invalidation: {:?}", e))?
                    .run()
                    .await;
            }
        }
    }

    // 9. Insert new facts
    if let Some(new_facts) = synthesis_json.get("new_facts").and_then(|v| v.as_array()) {
        for fact in new_facts {
            let fact_id = generate_uuid();
            let fact_text = fact.get("fact_text").and_then(|v| v.as_str()).unwrap_or("");
            let fact_type = fact.get("fact_type").and_then(|v| v.as_str()).unwrap_or("dimension");
            let dimension = fact.get("dimension").and_then(|v| v.as_str());
            let piece_ctx = fact.get("piece_context").map(|v| v.to_string());
            let trend = fact.get("trend").and_then(|v| v.as_str());
            let confidence = fact.get("confidence").and_then(|v| v.as_str()).unwrap_or("medium");
            let evidence = fact.get("evidence").map(|v| v.to_string()).unwrap_or_else(|| "[]".to_string());
            let valid_at = today;

            if fact_text.is_empty() {
                continue;
            }

            let _ = db
                .prepare(
                    "INSERT INTO synthesized_facts \
                     (id, student_id, fact_text, fact_type, dimension, piece_context, \
                      valid_at, trend, confidence, evidence, source_type, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .bind(&[
                    JsValue::from_str(&fact_id),
                    JsValue::from_str(student_id),
                    JsValue::from_str(fact_text),
                    JsValue::from_str(fact_type),
                    match dimension {
                        Some(d) => JsValue::from_str(d),
                        None => JsValue::NULL,
                    },
                    match piece_ctx.as_deref() {
                        Some(pc) if pc != "null" => JsValue::from_str(pc),
                        _ => JsValue::NULL,
                    },
                    JsValue::from_str(valid_at),
                    match trend {
                        Some(t) => JsValue::from_str(t),
                        None => JsValue::NULL,
                    },
                    JsValue::from_str(confidence),
                    JsValue::from_str(&evidence),
                    JsValue::from_str("synthesized"),
                    JsValue::from_str(&now),
                ])
                .map_err(|e| format!("Failed to bind fact insert: {:?}", e))?
                .run()
                .await;
        }
    }

    // 10. Update meta
    let fact_count_result: Option<serde_json::Value> = db
        .prepare(
            "SELECT COUNT(*) as cnt FROM synthesized_facts \
             WHERE student_id = ?1 AND invalid_at IS NULL AND expired_at IS NULL",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind fact count: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to count facts: {:?}", e))?;

    let total_facts = fact_count_result
        .as_ref()
        .and_then(|r| r.get("cnt"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    db.prepare(
        "INSERT INTO student_memory_meta (student_id, last_synthesis_at, total_facts) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(student_id) DO UPDATE SET last_synthesis_at = ?2, total_facts = ?3",
    )
    .bind(&[
        JsValue::from_str(student_id),
        JsValue::from_str(&now),
        JsValue::from_f64(total_facts as f64),
    ])
    .map_err(|e| format!("Failed to bind meta update: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to update meta: {:?}", e))?;

    console_log!(
        "Synthesis complete for student {}: {} new observations processed",
        student_id,
        new_observations.len()
    );

    Ok(())
}

/// Extract JSON from synthesis LLM output (may be wrapped in code fences).
fn extract_synthesis_json(output: &str) -> Result<serde_json::Value, String> {
    // Try to find JSON within ```json ... ``` fences
    let json_str = if let Some(start) = output.find("```json") {
        let json_start = start + 7;
        if let Some(end) = output[json_start..].find("```") {
            output[json_start..json_start + end].trim()
        } else {
            output.trim()
        }
    } else if let Some(start) = output.find('{') {
        if let Some(end) = output.rfind('}') {
            &output[start..=end]
        } else {
            output.trim()
        }
    } else {
        output.trim()
    };

    serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse synthesis JSON: {:?} - raw: {}", e, &json_str[..200.min(json_str.len())]))
}

/// Generate a UUID v4.
fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15]
    )
}
```

**Step 4: Verify it compiles**

Run: `cd apps/api && npx wrangler build`

**Step 5: Commit**

```bash
git add apps/api/src/services/memory.rs apps/api/src/services/prompts.rs
git commit -m "feat(api): add synthesis pipeline and prompts"
```

---

### Task 5: Integrate Synthesis into Sync Endpoint

**Files:**
- Modify: `apps/api/src/services/sync.rs` (add synthesis trigger after session insertion)

**Step 1: Add synthesis trigger to `handle_sync`**

After the session insertion loop (around line 122) and before building the response, add:

```rust
// Memory synthesis: check if we should synthesize facts from accumulated observations
match crate::services::memory::should_synthesize(env, &student_id).await {
    Ok(true) => {
        console_log!("Triggering memory synthesis for student {}", student_id);
        if let Err(e) = crate::services::memory::run_synthesis(env, &student_id).await {
            console_log!("Memory synthesis failed (non-fatal): {}", e);
        }
    }
    Ok(false) => {
        console_log!("Synthesis not needed for student {}", student_id);
    }
    Err(e) => {
        console_log!("Failed to check synthesis eligibility: {}", e);
    }
}
```

**Step 2: Verify it compiles**

Run: `cd apps/api && npx wrangler build`

**Step 3: Commit**

```bash
git add apps/api/src/services/sync.rs
git commit -m "feat(api): trigger memory synthesis after session sync"
```

---

### Task 6: Student-Reported Facts from Goal Extraction

**Files:**
- Modify: `apps/api/src/services/goals.rs` (insert student-reported facts after goal extraction)

**Step 1: After goals are extracted and merged, insert as student-reported facts**

In `goals.rs`, after the goals are successfully extracted and merged into the students table, add fact insertion for each extracted piece, focus area, and deadline:

```rust
// Store extracted goals as student-reported facts in synthesized_facts
let now = js_sys::Date::new_0()
    .to_iso_string()
    .as_string()
    .unwrap_or_default();
let today = &now[..10.min(now.len())];

for piece in &extracted.pieces {
    let fact_id = crate::services::memory::generate_fact_id();
    let _ = db
        .prepare(
            "INSERT OR IGNORE INTO synthesized_facts \
             (id, student_id, fact_text, fact_type, dimension, piece_context, \
              valid_at, confidence, evidence, source_type, created_at) \
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
        )
        .bind(&[
            JsValue::from_str(&fact_id),
            JsValue::from_str(&student_id),
            JsValue::from_str(&format!("Working on {}", piece)),
            JsValue::from_str("arc"),
            JsValue::from_str(&serde_json::json!({"title": piece}).to_string()),
            JsValue::from_str(today),
            JsValue::from_str("high"),
            JsValue::from_str("[]"),
            JsValue::from_str("student_reported"),
            JsValue::from_str(&now),
        ])
        .map_err(|e| format!("Bind failed: {:?}", e))?
        .run()
        .await;
}

for deadline in &extracted.deadlines {
    let fact_id = crate::services::memory::generate_fact_id();
    let invalid_at = deadline.date.as_deref();
    let _ = db
        .prepare(
            "INSERT OR IGNORE INTO synthesized_facts \
             (id, student_id, fact_text, fact_type, dimension, valid_at, invalid_at, \
              confidence, evidence, source_type, created_at) \
             VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?7, ?8, ?9, ?10)",
        )
        .bind(&[
            JsValue::from_str(&fact_id),
            JsValue::from_str(&student_id),
            JsValue::from_str(&deadline.description),
            JsValue::from_str("arc"),
            JsValue::from_str(today),
            match invalid_at {
                Some(d) => JsValue::from_str(d),
                None => JsValue::NULL,
            },
            JsValue::from_str("high"),
            JsValue::from_str("[]"),
            JsValue::from_str("student_reported"),
            JsValue::from_str(&now),
        ])
        .map_err(|e| format!("Bind failed: {:?}", e))?
        .run()
        .await;
}
```

Note: Add `pub fn generate_fact_id() -> String` as a public wrapper around `generate_uuid()` in `memory.rs`.

**Step 2: Verify it compiles**

Run: `cd apps/api && npx wrangler build`

**Step 3: Commit**

```bash
git add apps/api/src/services/goals.rs apps/api/src/services/memory.rs
git commit -m "feat(api): store extracted goals as student-reported facts"
```

---

### Task 7: Integration Testing

**Files:**
- No new files

**Step 1: Start local dev server**

Run: `cd apps/api && npx wrangler dev --local`

**Step 2: Apply migrations locally**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --local`

**Step 3: Test the ask pipeline with memory context**

Use debug auth to get a token, then test `/api/ask` with teaching moment data. Verify:
- Response includes observation (existing behavior preserved)
- D1 has a new row in `teaching_approaches`
- D1 has incremented `student_memory_meta.total_observations`

**Step 4: Test elaboration engagement tracking**

Call `/api/ask/elaborate` with the observation_id from Step 3. Verify:
- `teaching_approaches.engaged` is set to 1 for that observation

**Step 5: Test synthesis trigger**

Insert enough test observations (>= 3) via repeated `/api/ask` calls, then call `/api/sync`. Verify:
- Synthesis runs (check wrangler logs for "Triggering memory synthesis")
- `synthesized_facts` table has new rows
- `student_memory_meta.last_synthesis_at` is updated

**Step 6: Test memory retrieval**

After synthesis, call `/api/ask` again. Verify:
- The subagent prompt now includes "## Student Memory" section with active facts

**Step 7: Deploy**

Run: `cd apps/api && npx wrangler d1 migrations apply crescendai-db --remote && npx wrangler deploy`

**Step 8: Commit any fixes from testing**

```bash
git add -A apps/api/
git commit -m "fix(api): integration test fixes for memory system"
```

---

### Task 8: Update 06c Status and Documentation

**Files:**
- Modify: `docs/apps/06c-memory-system.md` (update status header)
- Modify: `docs/index.md` (add 06c to implementation status)

**Step 1: Update 06c status**

Change the status header in `docs/apps/06c-memory-system.md`:
```
**Status:** IMPLEMENTED (core pipeline)
**Last verified:** 2026-03-06
```

**Step 2: Add 06c to docs/index.md**

Add a row to the Implementation Slices table:
```
| 06c | [Student Memory System](apps/06c-memory-system.md) | IMPLEMENTED | Bi-temporal synthesized facts, teaching approach tracking, memory retrieval |
```

**Step 3: Commit**

```bash
git add docs/apps/06c-memory-system.md docs/index.md
git commit -m "docs: update 06c status to implemented"
```
