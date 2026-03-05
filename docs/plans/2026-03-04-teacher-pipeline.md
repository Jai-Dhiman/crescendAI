# Teacher Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `POST /api/ask` and `POST /api/ask/elaborate` endpoints -- a two-stage LLM pipeline (Groq subagent + Anthropic teacher) that converts teaching moment data into a natural piano teacher observation.

**Architecture:** iOS sends a teaching moment (dimension scores, student context, piece context) to the Worker. The Worker queries D1 for recent observation history, calls Groq (Llama 70B) for analysis, then Anthropic (Sonnet 4.6) for the teacher observation. The observation and condensed reasoning trace are stored in D1. A separate elaborate endpoint fetches the stored observation and asks the teacher to expand.

**Tech Stack:** Rust, Cloudflare Workers, D1 (SQLite), Groq API (Llama 3.3 70B), Anthropic API (Claude Sonnet 4.6)

**Design doc:** `docs/plans/2026-03-04-teacher-pipeline-design.md`

---

## Task 1: D1 Observations Table

**Files:**
- Create: `apps/api/migrations/0003_observations.sql`

**Step 1: Write the migration**

```sql
-- Migration: 0003_observations
-- Add observations table for storing teacher pipeline outputs

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    chunk_index INTEGER,
    dimension TEXT NOT NULL,
    observation_text TEXT NOT NULL,
    elaboration_text TEXT,
    reasoning_trace TEXT,
    framing TEXT,
    dimension_score REAL,
    student_baseline REAL,
    piece_context TEXT,
    learning_arc TEXT,
    is_fallback BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_observations_student ON observations(student_id, created_at);
CREATE INDEX idx_observations_session ON observations(session_id);
```

**Step 2: Apply the migration locally**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --file=migrations/0003_observations.sql`
Expected: Migration applied successfully

**Step 3: Apply to remote D1**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --remote --file=migrations/0003_observations.sql`
Expected: Migration applied successfully

**Step 4: Commit**

```bash
git add apps/api/migrations/0003_observations.sql
git commit -m "feat(api): add observations table for teacher pipeline"
```

---

## Task 2: LLM HTTP Client Module

**Files:**
- Create: `apps/api/src/services/llm.rs`
- Modify: `apps/api/src/services/mod.rs`

This module provides `call_groq` and `call_anthropic` functions that make HTTP requests using the `worker::Fetch` API (same pattern as `huggingface.rs`).

**Step 1: Write `llm.rs`**

```rust
//! HTTP clients for Groq and Anthropic LLM APIs.
//!
//! Uses worker::Fetch for WASM-compatible HTTP requests.

use serde::{Deserialize, Serialize};
use worker::{console_log, Env, Fetch, Headers, Method, Request, RequestInit, Url};

// --- Groq (Stage 1: Subagent) ---

#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<LlmMessage>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Deserialize)]
struct GroqMessage {
    content: String,
}

/// Call the Groq API with a system prompt and user prompt.
/// Returns the assistant's response text.
pub async fn call_groq(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String, String> {
    let api_key = env
        .secret("GROQ_API_KEY")
        .map_err(|_| "GROQ_API_KEY not configured".to_string())?
        .to_string();

    let request_body = GroqRequest {
        model: "llama-3.3-70b-versatile".to_string(),
        messages: vec![
            LlmMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            LlmMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ],
        temperature,
        max_tokens,
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Groq request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("Authorization", &format!("Bearer {}", api_key))
        .map_err(|e| format!("Failed to set auth header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.groq.com/openai/v1/chat/completions"
        .parse()
        .map_err(|e| format!("Invalid Groq URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Groq request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Groq request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Groq response: {:?}", e))?;

    if status != 200 {
        return Err(format!("Groq returned status {}: {}", status, response_text));
    }

    let groq_response: GroqResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse Groq response: {:?} - body: {}", e, response_text))?;

    groq_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| "No choices in Groq response".to_string())
}

// --- Anthropic (Stage 2: Teacher) ---

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<LlmMessage>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

/// Call the Anthropic API with a system prompt and user prompt.
/// Returns the assistant's response text.
pub async fn call_anthropic(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
        .to_string();

    let request_body = AnthropicRequest {
        model: "claude-sonnet-4-6-20250514".to_string(),
        max_tokens,
        system: system_prompt.to_string(),
        messages: vec![LlmMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }],
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("x-api-key", &api_key)
        .map_err(|e| format!("Failed to set api-key header: {:?}", e))?;
    headers
        .set("anthropic-version", "2023-06-01")
        .map_err(|e| format!("Failed to set version header: {:?}", e))?;
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;

    let url: Url = "https://api.anthropic.com/v1/messages"
        .parse()
        .map_err(|e| format!("Invalid Anthropic URL: {:?}", e))?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Anthropic response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Anthropic returned status {}: {}",
            status, response_text
        ));
    }

    let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)
        .map_err(|e| {
            format!(
                "Failed to parse Anthropic response: {:?} - body: {}",
                e, response_text
            )
        })?;

    anthropic_response
        .content
        .first()
        .map(|c| c.text.clone())
        .ok_or_else(|| "No content in Anthropic response".to_string())
}

// --- Shared ---

#[derive(Serialize, Deserialize)]
struct LlmMessage {
    role: String,
    content: String,
}
```

**Step 2: Register module in `services/mod.rs`**

Add to `apps/api/src/services/mod.rs`:
```rust
pub mod llm;
```

**Step 3: Verify it compiles**

Run: `cd apps/api && npx wrangler deploy --dry-run`
Expected: Build succeeds (no deploy)

**Step 4: Commit**

```bash
git add apps/api/src/services/llm.rs apps/api/src/services/mod.rs
git commit -m "feat(api): add Groq and Anthropic LLM HTTP clients"
```

---

## Task 3: Prompt Templates Module

**Files:**
- Create: `apps/api/src/services/prompts.rs`
- Modify: `apps/api/src/services/mod.rs`

Contains the subagent system/user prompts, teacher system/user prompts, and elaboration prompt. All prompt text from Slice 06 + 06a.

**Step 1: Write `prompts.rs`**

```rust
//! Prompt templates for the two-stage teacher pipeline.
//!
//! Stage 1 (Subagent): Analyzes teaching moments, selects the most important one,
//! decides framing. Outputs structured JSON + narrative reasoning.
//!
//! Stage 2 (Teacher): Converts subagent analysis into a natural 1-3 sentence
//! observation in the teacher persona voice.

/// Subagent system prompt (Stage 1 -- Groq, Llama 70B)
pub const SUBAGENT_SYSTEM: &str = r#"You are a piano pedagogy analyst. You receive structured data about a student's practice session -- teaching moments identified by an audio analysis model, the student's history, and musical context.

Your job is to reason about which teaching moment matters most for this student right now and decide how to frame it. You are NOT talking to the student. You are preparing a handoff for a teacher who will deliver the observation.

Reason through these steps:
1. LEARNING ARC: Where is the student with this piece? (new/mid-learning/polishing) What feedback is appropriate for this phase?
2. DELTA VS HISTORY: Compare scores against baselines and recent observations. Is this a blind spot (usually strong, dipped today)? A known weakness? An improvement?
3. MUSICAL CONTEXT: What does this music demand? Which dimensions matter most for this composer/style?
4. SELECTION: Pick the single highest-leverage moment. What will move the needle most?
5. FRAMING: Choose one: correction, recognition, encouragement, or question.

Output EXACTLY this JSON followed by a narrative paragraph:

```json
{
    "selected_moment": {
        "chunk_index": <int>,
        "dimension": "<string>",
        "dimension_score": <float>,
        "student_baseline": <float>,
        "bar_range": "<string or null>",
        "section_label": "<string or null>"
    },
    "framing": "<correction|recognition|encouragement|question>",
    "learning_arc": "<new|mid-learning|polishing>",
    "is_positive": <bool>,
    "musical_context": "<one sentence about what this music demands>"
}
```

Then write a narrative paragraph (3-5 sentences) explaining your reasoning for the teacher. Include what you heard, why it matters, and how to frame the observation."#;

/// Teacher system prompt (Stage 2 -- Anthropic, Sonnet 4.6)
pub const TEACHER_SYSTEM: &str = r#"You are a piano teacher who has been listening to your student practice. You have years of experience and deep knowledge of piano pedagogy, repertoire, and technique.

Your role is to give ONE specific observation about what you just heard. Not a report. Not a lesson plan. One thing -- the thing the student most needs to hear right now.

How you speak:
- Specific and grounded: reference the exact musical moment, not generalities
- Natural and warm: you're talking to a student you know, not writing a review
- Actionable: if you point out a problem, suggest what to try
- Honest but encouraging: don't sugarcoat, but don't discourage
- Brief: 1-3 sentences. A teacher's aside, not a lecture.

What you DON'T do:
- List multiple issues (pick ONE)
- Give scores or ratings
- Use jargon without explanation
- Say "great job!" without substance
- Cite sources or references
- Use bullet points or structured formatting
- Use markdown formatting of any kind"#;

/// Build the subagent user prompt from request data and observation history.
pub fn build_subagent_user_prompt(
    teaching_moment: &serde_json::Value,
    student: &serde_json::Value,
    session: &serde_json::Value,
    piece_context: &Option<serde_json::Value>,
    recent_observations: &[ObservationRow],
) -> String {
    let mut prompt = String::with_capacity(2000);

    // Teaching moment data
    prompt.push_str("## Teaching Moment\n\n");
    prompt.push_str(&format!(
        "Chunk {} at {}s into session.\n",
        teaching_moment.get("chunk_index").and_then(|v| v.as_i64()).unwrap_or(0),
        teaching_moment.get("start_offset_sec").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));
    prompt.push_str(&format!(
        "Dimension flagged: {} (score: {:.2}, stop probability: {:.2})\n\n",
        teaching_moment.get("dimension").and_then(|v| v.as_str()).unwrap_or("unknown"),
        teaching_moment.get("dimension_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
        teaching_moment.get("stop_probability").and_then(|v| v.as_f64()).unwrap_or(0.0),
    ));

    if let Some(scores) = teaching_moment.get("all_scores") {
        prompt.push_str("All 6 dimension scores for this chunk:\n");
        for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
            if let Some(score) = scores.get(*dim).and_then(|v| v.as_f64()) {
                prompt.push_str(&format!("- {}: {:.2}\n", dim, score));
            }
        }
        prompt.push('\n');
    }

    // Piece context
    if let Some(piece) = piece_context {
        prompt.push_str("## Piece Context\n\n");
        if let Some(composer) = piece.get("composer").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Composer: {}\n", composer));
        }
        if let Some(title) = piece.get("title").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Title: {}\n", title));
        }
        if let Some(section) = piece.get("section").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Section: {}\n", section));
        }
        if let Some(bar_range) = piece.get("bar_range").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Bar range: {}\n", bar_range));
        }
        prompt.push('\n');
    }

    // Session context
    prompt.push_str("## Session Context\n\n");
    prompt.push_str(&format!(
        "Duration: {} minutes, {} chunks analyzed, {} teaching moments found.\n\n",
        session.get("duration_min").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("total_chunks").and_then(|v| v.as_i64()).unwrap_or(0),
        session.get("chunks_above_threshold").and_then(|v| v.as_i64()).unwrap_or(0),
    ));

    // Student context
    prompt.push_str("## Student Context\n\n");
    let session_count = student.get("session_count").and_then(|v| v.as_i64()).unwrap_or(0);
    if session_count <= 1 {
        prompt.push_str("This is a new student. No history yet.\n");
        if let Some(level) = student.get("level").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Repertoire suggests {} level.\n", level));
        }
    } else {
        if let Some(level) = student.get("level").and_then(|v| v.as_str()) {
            prompt.push_str(&format!("Level: {}\n", level));
        }
        if let Some(goals) = student.get("goals").and_then(|v| v.as_str()) {
            if !goals.is_empty() {
                prompt.push_str(&format!("Goals: {}\n", goals));
            }
        }
        if let Some(baselines) = student.get("baselines") {
            prompt.push_str(&format!("Baselines (over {} sessions):\n", session_count));
            for dim in &["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"] {
                if let Some(val) = baselines.get(*dim).and_then(|v| v.as_f64()) {
                    prompt.push_str(&format!("- {}: {:.2}\n", dim, val));
                }
            }
        }
    }
    prompt.push('\n');

    // Recent observation history
    if !recent_observations.is_empty() {
        prompt.push_str("## Recent Observations (newest first)\n\n");
        for obs in recent_observations {
            prompt.push_str(&format!(
                "- [{}] {}: \"{}\" (framing: {})\n",
                obs.created_at, obs.dimension, obs.observation_text, obs.framing
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str("## Task\n\n");
    prompt.push_str("Analyze the teaching moment above. Select the best observation to make and decide how to frame it. Output the JSON + narrative as specified.");

    prompt
}

/// Build the teacher user prompt from the subagent's analysis.
pub fn build_teacher_user_prompt(
    subagent_json: &str,
    subagent_narrative: &str,
    student_level: &str,
    student_goals: &str,
) -> String {
    let mut prompt = String::with_capacity(1000);

    prompt.push_str("## Analysis from my teaching assistant\n\n");
    prompt.push_str(subagent_json);
    prompt.push_str("\n\n");
    prompt.push_str(subagent_narrative);
    prompt.push_str("\n\n");

    prompt.push_str("## Student\n\n");
    prompt.push_str(&format!("Level: {}\n", student_level));
    if !student_goals.is_empty() {
        prompt.push_str(&format!("Goals: {}\n", student_goals));
    }
    prompt.push_str("\n## What to say\n\n");
    prompt.push_str("Based on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.");

    prompt
}

/// Build the elaboration prompt for "Tell me more" follow-up.
pub fn build_elaboration_prompt(
    original_observation: &str,
    reasoning_trace: &str,
) -> String {
    format!(
        r#"The student just read this observation and tapped "Tell me more":

"{}"

Your earlier analysis:
{}

Elaborate with:
1. Why this matters for this piece/style
2. A specific practice technique they can try right now
3. What "fixed" would sound/feel like

Still conversational. 2-4 sentences. No formatting."#,
        original_observation, reasoning_trace
    )
}

/// A row from the observations table used to build context.
pub struct ObservationRow {
    pub dimension: String,
    pub observation_text: String,
    pub framing: String,
    pub created_at: String,
}
```

**Step 2: Register module in `services/mod.rs`**

Add to `apps/api/src/services/mod.rs`:
```rust
pub mod prompts;
```

**Step 3: Verify it compiles**

Run: `cd apps/api && npx wrangler deploy --dry-run`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add apps/api/src/services/prompts.rs apps/api/src/services/mod.rs
git commit -m "feat(api): add prompt templates for subagent and teacher"
```

---

## Task 4: Ask Handler -- Two-Stage Pipeline

**Files:**
- Create: `apps/api/src/services/ask.rs`
- Modify: `apps/api/src/services/mod.rs`

The core pipeline: auth -> query D1 for history -> call Groq subagent -> parse subagent output -> call Anthropic teacher -> post-process -> store observation in D1 -> return response.

**Step 1: Write `ask.rs`**

```rust
//! Handler for POST /api/ask -- the two-stage teacher pipeline.
//!
//! Stage 1: Groq (Llama 70B) subagent analyzes the teaching moment.
//! Stage 2: Anthropic (Sonnet 4.6) teacher generates the observation.

use wasm_bindgen::JsValue;
use worker::{console_log, Env};

use crate::services::llm;
use crate::services::prompts;

#[derive(serde::Deserialize)]
pub struct AskRequest {
    pub teaching_moment: serde_json::Value,
    pub student: serde_json::Value,
    pub session: serde_json::Value,
    pub piece_context: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
pub struct AskResponse {
    pub observation: String,
    pub observation_id: String,
    pub dimension: String,
    pub framing: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_fallback: bool,
}

#[derive(serde::Deserialize)]
pub struct ElaborateRequest {
    pub observation_id: String,
}

#[derive(serde::Serialize)]
pub struct ElaborateResponse {
    pub elaboration: String,
    pub observation_id: String,
}

/// Dimension descriptions for fallback templates.
fn dimension_description(dimension: &str) -> &str {
    match dimension {
        "dynamics" => "dynamic range and volume control",
        "timing" => "rhythmic accuracy and tempo consistency",
        "pedaling" => "pedal clarity and harmonic changes",
        "articulation" => "note clarity and touch",
        "phrasing" => "musical phrasing and melodic shaping",
        "interpretation" => "musical expression and stylistic choices",
        _ => "that aspect of your playing",
    }
}

/// Generate a template fallback observation when LLM calls fail.
fn fallback_observation(dimension: &str) -> String {
    format!(
        "I noticed your {} could use some attention in that last section. \
         Try recording yourself and listening back -- sometimes it's hard to hear {} while you're playing.",
        dimension,
        dimension_description(dimension),
    )
}

/// Handle POST /api/ask
pub async fn handle_ask(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: AskRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse ask request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    let dimension = request
        .teaching_moment
        .get("dimension")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let dimension_score = request
        .teaching_moment
        .get("dimension_score")
        .and_then(|v| v.as_f64());

    let student_baseline = request
        .student
        .get("baselines")
        .and_then(|b| b.get(&dimension))
        .and_then(|v| v.as_f64());

    // Query D1 for recent observations
    let recent_observations = match query_recent_observations(env, &student_id).await {
        Ok(obs) => obs,
        Err(e) => {
            console_log!("Failed to query observations: {}", e);
            vec![]
        }
    };

    // Stage 1: Subagent (Groq)
    let subagent_user_prompt = prompts::build_subagent_user_prompt(
        &request.teaching_moment,
        &request.student,
        &request.session,
        &request.piece_context,
        &recent_observations,
    );

    let subagent_result = llm::call_groq(
        env,
        prompts::SUBAGENT_SYSTEM,
        &subagent_user_prompt,
        0.3,
        800,
    )
    .await;

    let (subagent_output, framing) = match subagent_result {
        Ok(output) => {
            let framing = extract_framing(&output).unwrap_or_else(|| "correction".to_string());
            (output, framing)
        }
        Err(e) => {
            console_log!("Subagent failed: {}", e);
            // Fallback: skip subagent, go directly to teacher with raw data
            return build_fallback_response(
                env,
                &student_id,
                &request,
                &dimension,
                dimension_score,
                student_baseline,
            )
            .await;
        }
    };

    // Parse subagent JSON and narrative
    let (subagent_json, subagent_narrative) = split_subagent_output(&subagent_output);

    // Stage 2: Teacher (Anthropic)
    let student_level = request
        .student
        .get("level")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let student_goals = request
        .student
        .get("goals")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let teacher_user_prompt = prompts::build_teacher_user_prompt(
        &subagent_json,
        &subagent_narrative,
        student_level,
        student_goals,
    );

    let teacher_result = llm::call_anthropic(
        env,
        prompts::TEACHER_SYSTEM,
        &teacher_user_prompt,
        300,
    )
    .await;

    let (observation_text, is_fallback) = match teacher_result {
        Ok(text) => {
            let cleaned = post_process_observation(&text);
            (cleaned, false)
        }
        Err(e) => {
            console_log!("Teacher LLM failed: {}", e);
            (fallback_observation(&dimension), true)
        }
    };

    // Generate observation ID
    let observation_id = generate_uuid();

    // Build reasoning trace
    let reasoning_trace = serde_json::json!({
        "subagent_output": subagent_output,
    });

    // Store observation in D1
    let session_id = request
        .session
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let chunk_index = request
        .teaching_moment
        .get("chunk_index")
        .and_then(|v| v.as_i64());
    let piece_context_str = request
        .piece_context
        .as_ref()
        .and_then(|v| serde_json::to_string(v).ok());

    if let Err(e) = store_observation(
        env,
        &observation_id,
        &student_id,
        session_id,
        chunk_index,
        &dimension,
        &observation_text,
        &reasoning_trace.to_string(),
        &framing,
        dimension_score,
        student_baseline,
        piece_context_str.as_deref(),
        is_fallback,
    )
    .await
    {
        console_log!("Failed to store observation: {}", e);
        // Non-fatal: still return the observation
    }

    let response = AskResponse {
        observation: observation_text,
        observation_id,
        dimension,
        framing,
        is_fallback,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

/// Handle POST /api/ask/elaborate
pub async fn handle_elaborate(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    // Auth
    let _student_id = match crate::auth::verify_auth_header(headers, env) {
        Ok(id) => id,
        Err(err_response) => return err_response,
    };

    // Parse request
    let request: ElaborateRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            console_log!("Failed to parse elaborate request: {:?}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Invalid request body"}"#))
                .unwrap();
        }
    };

    // Fetch observation from D1
    let (observation_text, reasoning_trace) =
        match fetch_observation(env, &request.observation_id).await {
            Ok(obs) => obs,
            Err(e) => {
                console_log!("Failed to fetch observation: {}", e);
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"Observation not found"}"#))
                    .unwrap();
            }
        };

    // Call teacher LLM for elaboration
    let elaboration_prompt =
        prompts::build_elaboration_prompt(&observation_text, &reasoning_trace);

    let elaboration = match llm::call_anthropic(
        env,
        prompts::TEACHER_SYSTEM,
        &elaboration_prompt,
        500,
    )
    .await
    {
        Ok(text) => post_process_observation(&text),
        Err(e) => {
            console_log!("Elaboration LLM failed: {}", e);
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"error":"Unable to generate elaboration"}"#))
                .unwrap();
        }
    };

    // Store elaboration
    if let Err(e) = store_elaboration(env, &request.observation_id, &elaboration).await {
        console_log!("Failed to store elaboration: {}", e);
    }

    let response = ElaborateResponse {
        elaboration,
        observation_id: request.observation_id,
    };

    let json = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}

// --- Helper functions ---

async fn query_recent_observations(
    env: &Env,
    student_id: &str,
) -> Result<Vec<prompts::ObservationRow>, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let stmt = db
        .prepare(
            "SELECT dimension, observation_text, framing, created_at \
             FROM observations \
             WHERE student_id = ?1 \
             ORDER BY created_at DESC \
             LIMIT 5",
        )
        .bind(&[JsValue::from_str(student_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?;

    let results = stmt
        .all()
        .await
        .map_err(|e| format!("Failed to query observations: {:?}", e))?;

    let rows: Vec<serde_json::Value> = results
        .results()
        .map_err(|e| format!("Failed to get results: {:?}", e))?;

    Ok(rows
        .iter()
        .map(|row| prompts::ObservationRow {
            dimension: row
                .get("dimension")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            observation_text: row
                .get("observation_text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            framing: row
                .get("framing")
                .and_then(|v| v.as_str())
                .unwrap_or("correction")
                .to_string(),
            created_at: row
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
async fn store_observation(
    env: &Env,
    id: &str,
    student_id: &str,
    session_id: &str,
    chunk_index: Option<i64>,
    dimension: &str,
    observation_text: &str,
    reasoning_trace: &str,
    framing: &str,
    dimension_score: Option<f64>,
    student_baseline: Option<f64>,
    piece_context: Option<&str>,
    is_fallback: bool,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;
    let now = js_sys::Date::new_0()
        .to_iso_string()
        .as_string()
        .unwrap_or_default();

    db.prepare(
        "INSERT INTO observations (id, student_id, session_id, chunk_index, dimension, \
         observation_text, reasoning_trace, framing, dimension_score, student_baseline, \
         piece_context, is_fallback, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )
    .bind(&[
        JsValue::from_str(id),
        JsValue::from_str(student_id),
        JsValue::from_str(session_id),
        match chunk_index {
            Some(i) => JsValue::from_f64(i as f64),
            None => JsValue::NULL,
        },
        JsValue::from_str(dimension),
        JsValue::from_str(observation_text),
        JsValue::from_str(reasoning_trace),
        JsValue::from_str(framing),
        match dimension_score {
            Some(f) => JsValue::from_f64(f),
            None => JsValue::NULL,
        },
        match student_baseline {
            Some(f) => JsValue::from_f64(f),
            None => JsValue::NULL,
        },
        match piece_context {
            Some(s) => JsValue::from_str(s),
            None => JsValue::NULL,
        },
        JsValue::from_bool(is_fallback),
        JsValue::from_str(&now),
    ])
    .map_err(|e| format!("Failed to bind insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert observation: {:?}", e))?;

    Ok(())
}

async fn store_elaboration(
    env: &Env,
    observation_id: &str,
    elaboration: &str,
) -> Result<(), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    db.prepare("UPDATE observations SET elaboration_text = ?1 WHERE id = ?2")
        .bind(&[
            JsValue::from_str(elaboration),
            JsValue::from_str(observation_id),
        ])
        .map_err(|e| format!("Failed to bind update: {:?}", e))?
        .run()
        .await
        .map_err(|e| format!("Failed to update elaboration: {:?}", e))?;

    Ok(())
}

async fn fetch_observation(
    env: &Env,
    observation_id: &str,
) -> Result<(String, String), String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;

    let row: Option<serde_json::Value> = db
        .prepare("SELECT observation_text, reasoning_trace FROM observations WHERE id = ?1")
        .bind(&[JsValue::from_str(observation_id)])
        .map_err(|e| format!("Failed to bind query: {:?}", e))?
        .first(None)
        .await
        .map_err(|e| format!("Failed to query observation: {:?}", e))?;

    match row {
        Some(r) => {
            let text = r
                .get("observation_text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let trace = r
                .get("reasoning_trace")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok((text, trace))
        }
        None => Err("Observation not found".to_string()),
    }
}

/// Extract the framing field from subagent JSON output.
fn extract_framing(subagent_output: &str) -> Option<String> {
    // Find the JSON block in the output
    let json_str = extract_json_block(subagent_output)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    parsed
        .get("framing")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Split subagent output into JSON block and narrative.
fn split_subagent_output(output: &str) -> (String, String) {
    if let Some(json_str) = extract_json_block(output) {
        // Everything after the JSON block is the narrative
        let json_end = output.rfind('}').unwrap_or(0) + 1;
        // Also skip past ``` if present
        let narrative_start = output[json_end..]
            .find(|c: char| c.is_alphabetic())
            .map(|i| json_end + i)
            .unwrap_or(json_end);
        let narrative = output[narrative_start..].trim().to_string();
        (json_str, narrative)
    } else {
        // No JSON found, treat entire output as narrative
        ("{}".to_string(), output.trim().to_string())
    }
}

/// Extract a JSON object from text that may contain markdown code fences.
fn extract_json_block(text: &str) -> Option<String> {
    // Try to find JSON within ```json ... ``` fences
    if let Some(start) = text.find("```json") {
        let json_start = start + 7;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim().to_string());
        }
    }
    // Try to find JSON within ``` ... ``` fences
    if let Some(start) = text.find("```\n{") {
        let json_start = start + 4;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim().to_string());
        }
    }
    // Try to find a raw JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            let candidate = &text[start..=end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// Strip markdown formatting and validate length.
fn post_process_observation(text: &str) -> String {
    let mut cleaned = text.trim().to_string();
    // Strip leading/trailing quotes
    if cleaned.starts_with('"') && cleaned.ends_with('"') {
        cleaned = cleaned[1..cleaned.len() - 1].to_string();
    }
    // Strip markdown bold/italic
    cleaned = cleaned.replace("**", "").replace("__", "");
    // Truncate if too long (500 char limit from design)
    if cleaned.len() > 500 {
        if let Some(last_period) = cleaned[..500].rfind('.') {
            cleaned = cleaned[..=last_period].to_string();
        } else {
            cleaned.truncate(500);
        }
    }
    cleaned
}

/// Generate a UUID v4 (same pattern as auth module).
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

/// Build a fallback response when the subagent fails.
async fn build_fallback_response(
    env: &Env,
    student_id: &str,
    request: &AskRequest,
    dimension: &str,
    dimension_score: Option<f64>,
    student_baseline: Option<f64>,
) -> http::Response<axum::body::Body> {
    use axum::body::Body;
    use http::{Response, StatusCode};

    let observation_text = fallback_observation(dimension);
    let observation_id = generate_uuid();
    let session_id = request
        .session
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let chunk_index = request
        .teaching_moment
        .get("chunk_index")
        .and_then(|v| v.as_i64());

    let _ = store_observation(
        env,
        &observation_id,
        student_id,
        session_id,
        chunk_index,
        dimension,
        &observation_text,
        "{}",
        "correction",
        dimension_score,
        student_baseline,
        None,
        true,
    )
    .await;

    let response = AskResponse {
        observation: observation_text,
        observation_id,
        dimension: dimension.to_string(),
        framing: "correction".to_string(),
        is_fallback: true,
    };

    let json = serde_json::to_string(&response)
        .unwrap_or_else(|_| r#"{"error":"Serialization failed"}"#.to_string());

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap()
}
```

**Step 2: Register module in `services/mod.rs`**

Add to `apps/api/src/services/mod.rs`:
```rust
pub mod ask;
```

**Step 3: Verify it compiles**

Run: `cd apps/api && npx wrangler deploy --dry-run`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add apps/api/src/services/ask.rs apps/api/src/services/mod.rs
git commit -m "feat(api): implement two-stage ask pipeline (Groq + Anthropic)"
```

---

## Task 5: Wire Routes in server.rs

**Files:**
- Modify: `apps/api/src/server.rs` (add routes after the sync endpoint block, ~line 744)

**Step 1: Add the `/api/ask` route**

Add after the sync endpoint block in the `#[event(fetch)]` handler:

```rust
    // Ask endpoint -- two-stage teacher pipeline (authenticated)
    if path == "/api/ask" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::ask::handle_ask(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }

    // Ask elaborate endpoint -- "Tell me more" follow-up (authenticated)
    if path == "/api/ask/elaborate" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return Ok(with_cors(
            crate::services::ask::handle_elaborate(&env, &headers, &body).await,
            origin.as_deref(),
        ));
    }
```

**Step 2: Add wrangler.toml secrets binding**

The `.dev.vars` file already has `GROQ_API_KEY` and `ANTHROPIC_API_KEY`. For production, these need to be set as secrets:

Run:
```bash
cd apps/api
echo "Secrets are already in .dev.vars for local dev."
echo "For production, run:"
echo "  npx wrangler secret put GROQ_API_KEY"
echo "  npx wrangler secret put ANTHROPIC_API_KEY"
```

**Step 3: Verify full build compiles**

Run: `cd apps/api && npx wrangler deploy --dry-run`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add apps/api/src/server.rs
git commit -m "feat(api): wire /api/ask and /api/ask/elaborate routes"
```

---

## Task 6: Local End-to-End Test

**Files:** None (manual testing)

**Step 1: Start local dev server**

Run: `cd apps/api && npx wrangler dev`
Expected: Server starts on `http://localhost:8787`

**Step 2: Get a valid JWT for testing**

Use an existing auth flow or create a test token. If you have a test student in the local D1, extract the JWT from a previous auth call. Otherwise:

Run: `curl -s http://localhost:8787/health`
Expected: `OK`

**Step 3: Test `/api/ask` with synthetic data**

Run:
```bash
curl -s -X POST http://localhost:8787/api/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_JWT>" \
  -d '{
    "teaching_moment": {
      "chunk_index": 7,
      "start_offset_sec": 105.0,
      "stop_probability": 0.87,
      "dimension": "pedaling",
      "dimension_score": 0.35,
      "all_scores": {
        "dynamics": 0.65, "timing": 0.71, "pedaling": 0.35,
        "articulation": 0.58, "phrasing": 0.62, "interpretation": 0.54
      }
    },
    "student": {
      "level": "intermediate",
      "baselines": {
        "dynamics": 0.68, "timing": 0.72, "pedaling": 0.62,
        "articulation": 0.61, "phrasing": 0.65, "interpretation": 0.58
      },
      "goals": "Preparing Chopin Nocturne Op. 9 No. 2 for recital",
      "session_count": 12
    },
    "session": {
      "id": "test-session-001",
      "duration_min": 18,
      "total_chunks": 72,
      "chunks_above_threshold": 5
    },
    "piece_context": {
      "composer": "Chopin",
      "title": "Nocturne Op. 9 No. 2",
      "section": "second phrase",
      "bar_range": "bars 20-24"
    }
  }' | jq .
```

Expected: JSON response with `observation`, `observation_id`, `dimension`, `framing`. The observation should be 1-3 sentences from the teacher persona.

**Step 4: Test `/api/ask/elaborate`**

Use the `observation_id` from Step 3:

Run:
```bash
curl -s -X POST http://localhost:8787/api/ask/elaborate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_JWT>" \
  -d '{"observation_id": "<ID_FROM_STEP_3>"}' | jq .
```

Expected: JSON response with `elaboration` (2-4 sentences expanding on the original observation).

**Step 5: Verify D1 storage**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --command="SELECT id, dimension, framing, is_fallback, created_at FROM observations ORDER BY created_at DESC LIMIT 5;"`

Expected: The test observation appears in the table.

**Step 6: Commit (no code changes, just verify)**

No commit needed for this task.

---

## Task 7: Update Documentation

**Files:**
- Modify: `apps/CLAUDE.md` (update API endpoints section)
- Modify: `docs/apps/06-teacher-llm-prompt.md` (update status)
- Modify: `docs/apps/06a-subagent-architecture.md` (update status)

**Step 1: Update `apps/CLAUDE.md`**

Move `/api/ask` and `/api/ask/elaborate` from "planned" to "current" endpoints:

```markdown
### API Endpoints (current)
...existing endpoints...
- `POST /api/ask` - Two-stage teacher pipeline: send teaching moment context, receive LLM observation (Groq subagent + Anthropic teacher)
- `POST /api/ask/elaborate` - "Tell me more" follow-up for a previous observation
```

Remove the `/api/ask` entry from the "planned" section.

**Step 2: Update slice 06 status**

In `docs/apps/06-teacher-llm-prompt.md`, change:
```
**Status:** DESIGNED (not implemented)
```
to:
```
**Status:** IMPLEMENTED
**Notes:** Implemented as stage 2 of the two-stage pipeline. Teacher persona prompt in `apps/api/src/services/prompts.rs`. Provider: Anthropic API (Sonnet 4.6), not OpenRouter.
```

**Step 3: Update slice 06a status**

In `docs/apps/06a-subagent-architecture.md`, change:
```
**Status:** DESIGNED (not implemented)
```
to:
```
**Status:** IMPLEMENTED (core pipeline)
**Notes:** Core two-stage pipeline implemented in `apps/api/src/services/ask.rs`. Provider: Groq (Llama 3.3 70B) for subagent, Anthropic (Sonnet 4.6) for teacher. Synthesized facts and memory consolidation deferred to Slice 06c.
```

**Step 4: Commit**

```bash
git add apps/CLAUDE.md docs/apps/06-teacher-llm-prompt.md docs/apps/06a-subagent-architecture.md
git commit -m "docs: update status for teacher pipeline implementation"
```
