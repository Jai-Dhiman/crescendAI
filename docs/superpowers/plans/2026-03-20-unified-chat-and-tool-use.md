# Unified Chat + Teacher Tool Use Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the practice observation path and chat path into a single conversation, then add Anthropic native tool_use so the teacher LLM can declare exercise artifacts inline.

**Architecture:** Practice sessions become part of conversations (linked via `conversation_id`). Observations during recording are persisted as messages in D1 and rendered inline in the chat thread (not ephemeral toasts). The teacher LLM gains a `create_exercise` tool (Anthropic tool_use, `tool_choice: "auto"`) that produces `ExerciseSetConfig` artifacts. The client-side artifact rendering infrastructure (Artifact.tsx, ExerciseSetCard, Zustand store) is already complete and unchanged.

**Tech Stack:** Rust/WASM (Cloudflare Workers), D1 (SQLite), Anthropic Messages API with tool_use, React 19 + TanStack Start, Zustand, WebSocket

**Task Dependency Chain:**
```
1 (migration) → 2 (llm tools) → 3 (prompts) → 4 (ask pipeline) → 5 (start + server.rs)
    → 8 (web types) → 9 (hook + WS plumbing) → 6 (DO persistence) → 7 (chat handler)
    → 10 (recording UI) → 11 (integration test) → 12 (docs)
```
Tasks 6 and 7 depend on the web plumbing (Tasks 8-9) because the DO needs `conversation_id` passed via WebSocket URL from the client. Execute in dependency order, not numerical order.

---

## File Map

### API (Rust) -- Modified

| File | Responsibility | Changes |
|------|---------------|---------|
| `apps/api/migrations/0007_unified_chat.sql` | Schema migration | Add `conversation_id` to sessions, extend messages table with observation fields |
| `apps/api/src/services/llm.rs` | LLM HTTP clients | New `call_anthropic_with_tools()` function returning structured `AnthropicToolResponse` |
| `apps/api/src/services/prompts.rs` | Prompt templates + tool defs | Add `exercise_tool_definition()`, `build_teacher_user_prompt_with_catalog()`. Requires new `use crate::services::llm;` import |
| `apps/api/src/services/ask.rs` | Two-stage pipeline | Extend `AskInnerResponse` with `components` field, parse tool_use blocks, persist exercises |
| `apps/api/src/practice/start.rs` | POST /api/practice/start | Accept `body: &[u8]` param, parse optional `conversation_id`, create conversation if absent, link session |
| `apps/api/src/server.rs` | Route handler | Pass request body to `handle_start`, pass `conversation_id` in DO WebSocket URL query params |
| `apps/api/src/practice/session.rs` | Durable Object session brain | Add `conversation_id` to `SessionState`, persist observations as messages in D1, include components in WebSocket events. Update `ObservationRecord` to include `components_json` |
| `apps/api/src/services/chat.rs` | Chat handler | Update `MessageRow` struct, load all message types in conversation history, prefix observations in LLM context |

### Web (TypeScript) -- Modified

| File | Responsibility | Changes |
|------|---------------|---------|
| `apps/web/src/lib/practice-api.ts` | API types + client | Add `conversationId` to start(), add `components` to WS event types |
| `apps/web/src/lib/types.ts` | Message types | Add `message_type`, `session_id` to RichMessage |
| `apps/web/src/hooks/usePracticeSession.ts` | Recording hook | Pass conversationId, parse components from WS, emit observations as messages |
| `apps/web/src/components/AppChat.tsx` | Main chat component | Replace ListeningMode overlay with inline RecordingBanner, handle observation messages in thread |
| `apps/web/src/components/RecordingBanner.tsx` | NEW: inline recording UI | Compact waveform + timer + stop button that sits above chat input |
| `apps/web/src/components/ChatMessages.tsx` | Message rendering | Render observation messages with dimension badge + components |

### Tests

| File | Responsibility |
|------|---------------|
| `apps/web/src/lib/practice-api.test.ts` | Type validation for new WS event shape |
| `apps/api/tests/exercises.test.ts` | Extend with generated exercise persistence tests |

---

## Task 1: D1 Schema Migration

**Files:**
- Create: `apps/api/migrations/0007_unified_chat.sql`

- [ ] **Step 1: Write the migration SQL**

```sql
-- Link practice sessions to conversations
ALTER TABLE sessions ADD COLUMN conversation_id TEXT;
CREATE INDEX IF NOT EXISTS idx_sessions_conversation ON sessions(conversation_id);

-- Extend messages to carry observation metadata
ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT 'chat';
ALTER TABLE messages ADD COLUMN dimension TEXT;
ALTER TABLE messages ADD COLUMN framing TEXT;
ALTER TABLE messages ADD COLUMN components_json TEXT;
ALTER TABLE messages ADD COLUMN session_id TEXT;
ALTER TABLE messages ADD COLUMN observation_id TEXT;

-- Back-link observations to messages and conversations
ALTER TABLE observations ADD COLUMN message_id TEXT;
ALTER TABLE observations ADD COLUMN conversation_id TEXT;
```

- [ ] **Step 2: Apply migration locally**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --file=migrations/0007_unified_chat.sql`
Expected: Migration applies without errors

- [ ] **Step 3: Verify schema**

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --command="PRAGMA table_info(sessions);" | grep conversation_id`
Expected: Row showing `conversation_id TEXT`

Run: `cd apps/api && npx wrangler d1 execute crescendai-db --local --command="PRAGMA table_info(messages);" | grep message_type`
Expected: Row showing `message_type TEXT`

- [ ] **Step 4: Commit**

```bash
git add apps/api/migrations/0007_unified_chat.sql
git commit -m "feat: add D1 migration for unified chat + practice sessions"
```

---

## Task 2: Anthropic Tool Use in LLM Client

**Files:**
- Modify: `apps/api/src/services/llm.rs:118-218`

This task adds a new function `call_anthropic_with_tools()` alongside the existing `call_anthropic()`. The existing function stays unchanged (used by elaboration, summary, etc.).

- [ ] **Step 1: Add tool-related structs**

After line 218 in `llm.rs` (after the `call_anthropic()` function, before the Workers AI section at line 220), add:

```rust
// --- Anthropic Tool Use ---

#[derive(Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicToolRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<LlmMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
pub struct AnthropicToolResponse {
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Parsed result from a tool-enabled Anthropic call.
pub struct AnthropicToolResult {
    /// The text content blocks concatenated.
    pub text: String,
    /// Any tool calls made by the model.
    pub tool_calls: Vec<ToolCall>,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}
```

- [ ] **Step 2: Add `call_anthropic_with_tools` function**

After the existing `call_anthropic()` function (after line 218), add:

```rust
/// Call the Anthropic API with optional tool definitions.
/// Returns structured result with text + tool calls separated.
pub async fn call_anthropic_with_tools(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    tools: Option<Vec<AnthropicTool>>,
) -> Result<AnthropicToolResult, String> {
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
        .to_string();

    let tool_choice = tools.as_ref().map(|_| serde_json::json!({"type": "auto"}));

    let request_body = AnthropicToolRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: system_prompt.to_string(),
        messages: vec![LlmMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        }],
        tools,
        tool_choice,
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic tool request: {:?}", e))?;

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
        .map_err(|e| format!("Failed to create Anthropic tool request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic tool request failed: {:?}", e))?;

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Anthropic tool response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Anthropic returned status {}: {}",
            status, response_text
        ));
    }

    let parsed: AnthropicToolResponse =
        serde_json::from_str(&response_text).map_err(|e| {
            format!(
                "Failed to parse Anthropic tool response: {:?} - body: {}",
                e, response_text
            )
        })?;

    // Separate text and tool_use blocks
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in parsed.content {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_parts.push(text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall { id, name, input });
            }
        }
    }

    Ok(AnthropicToolResult {
        text: text_parts.join(""),
        tool_calls,
    })
}
```

- [ ] **Step 3: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors (new code is additive, no existing callers change)

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: add call_anthropic_with_tools() for teacher tool use"
```

---

## Task 3: Exercise Tool Definition + Catalog Injection

**Files:**
- Modify: `apps/api/src/services/prompts.rs`

- [ ] **Step 1: Add import for llm types**

At the top of `prompts.rs`, add:
```rust
use crate::services::llm;
```

- [ ] **Step 2: Add exercise tool definition constant**

After `TEACHER_SYSTEM` (after line 85, before `build_subagent_user_prompt` at line 88), add:

```rust
/// Tool definition for teacher exercise creation (Anthropic tool_use).
pub fn exercise_tool_definition() -> llm::AnthropicTool {
    llm::AnthropicTool {
        name: "create_exercise".to_string(),
        description: "Create a focused practice exercise when the student would benefit from \
            structured practice on a specific passage or technique. Use sparingly -- only when \
            a concrete drill would be more helpful than verbal guidance alone. Most observations \
            should be text-only."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "source_passage": {
                    "type": "string",
                    "description": "The passage this exercise targets (e.g., 'measures 12-16' or 'the opening phrase')"
                },
                "target_skill": {
                    "type": "string",
                    "description": "The specific skill being developed (e.g., 'Voice balancing between hands')"
                },
                "exercises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short exercise name"
                            },
                            "instruction": {
                                "type": "string",
                                "description": "Concrete steps the student should follow. 2-4 sentences."
                            },
                            "focus_dimension": {
                                "type": "string",
                                "enum": ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
                            },
                            "hands": {
                                "type": "string",
                                "enum": ["left", "right", "both"]
                            }
                        },
                        "required": ["title", "instruction", "focus_dimension"]
                    },
                    "minItems": 1,
                    "maxItems": 3
                }
            },
            "required": ["source_passage", "target_skill", "exercises"]
        }),
    }
}
```

- [ ] **Step 3: Add catalog injection to teacher prompt**

After `build_teacher_user_prompt` (after line 248), add:

```rust
/// Build teacher user prompt with catalog exercises injected.
/// When matching catalog exercises exist, the teacher can reference them by ID.
pub fn build_teacher_user_prompt_with_catalog(
    subagent_json: &str,
    subagent_narrative: &str,
    student_level: &str,
    student_goals: &str,
    catalog_exercises: &[(String, String, String, String)],  // (id, title, description, difficulty)
) -> String {
    let mut prompt = build_teacher_user_prompt(subagent_json, subagent_narrative, student_level, student_goals);

    if !catalog_exercises.is_empty() {
        // Insert catalog context before the closing </task> tag
        let task_tag = "<task>\n";
        if let Some(pos) = prompt.rfind(task_tag) {
            let catalog_section = {
                let mut s = String::from("\n<available_exercises>\n");
                s.push_str("These curated exercises are available. If one fits, you can reference it by ID in your exercise tool call.\n");
                for (id, title, description, difficulty) in catalog_exercises {
                    s.push_str(&format!("- [{}] {} ({}): {}\n", id, title, difficulty, description));
                }
                s.push_str("</available_exercises>\n\n");
                s
            };
            prompt.insert_str(pos, &catalog_section);
        }
    }

    // Update the task instruction to mention tool availability
    let old_task = "<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n</task>";
    let new_task = "<task>\nBased on the analysis above, give one observation to the student. Be specific about what you heard and what to try. 1-3 sentences, no formatting.\n\nIf the student would benefit from a concrete practice drill, use the create_exercise tool to attach one. Most observations should be text-only -- only create an exercise when structured practice would genuinely help more than verbal guidance.\n</task>";
    prompt = prompt.replace(old_task, new_task);

    prompt
}
```

- [ ] **Step 4: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/prompts.rs
git commit -m "feat: add exercise tool definition and catalog-aware teacher prompt"
```

---

## Task 4: Extend Pipeline to Parse Tool Use + Persist Exercises

**Files:**
- Modify: `apps/api/src/services/ask.rs:40-196`

- [ ] **Step 1: Add components field to `AskInnerResponse`**

Change the struct at line 49:

```rust
/// Output from the core LLM pipeline (no D1 side effects).
pub struct AskInnerResponse {
    pub observation_text: String,
    pub dimension: String,
    pub framing: String,
    pub reasoning_trace: String,
    pub is_fallback: bool,
    /// Optional exercise artifact from teacher tool use.
    pub components_json: Option<String>,
}
```

- [ ] **Step 2: Add catalog lookup helper**

After `generate_uuid()` (after line 643), add:

```rust
/// Look up catalog exercises matching a dimension and student level.
/// Returns up to 5 matching exercises as (id, title, description, difficulty).
async fn lookup_catalog_exercises(
    env: &Env,
    dimension: &str,
    _student_level: &str,
) -> Vec<(String, String, String, String)> {
    let db = match env.d1("DB") {
        Ok(db) => db,
        Err(e) => {
            console_error!("D1 binding failed for catalog lookup: {:?}", e);
            return Vec::new();
        }
    };

    let query = match db
        .prepare(
            "SELECT e.id, e.title, e.description, e.difficulty \
             FROM exercises e \
             JOIN exercise_dimensions ed ON e.id = ed.exercise_id \
             WHERE ed.dimension = ?1 AND e.source = 'curated' \
             LIMIT 5",
        )
        .bind(&[JsValue::from_str(dimension)])
    {
        Ok(q) => q,
        Err(e) => {
            console_error!("Catalog query bind failed: {:?}", e);
            return Vec::new();
        }
    };

    match query.all().await {
        Ok(result) => {
            let rows: Vec<serde_json::Value> = result
                .results()
                .unwrap_or_default();
            rows.iter()
                .filter_map(|row| {
                    let id = row.get("id")?.as_str()?.to_string();
                    let title = row.get("title")?.as_str()?.to_string();
                    let desc = row.get("description")?.as_str()?.to_string();
                    let diff = row.get("difficulty")?.as_str()?.to_string();
                    Some((id, title, desc, diff))
                })
                .collect()
        }
        Err(e) => {
            console_error!("Catalog query failed: {:?}", e);
            Vec::new()
        }
    }
}
```

- [ ] **Step 3: Add exercise persistence helper**

```rust
/// Persist a teacher-generated exercise to D1 and return the exercise_id.
async fn persist_generated_exercise(
    env: &Env,
    title: &str,
    instruction: &str,
    focus_dimension: &str,
    source_passage: &str,
    target_skill: &str,
) -> Result<String, String> {
    let db = env.d1("DB").map_err(|e| format!("D1 binding failed: {:?}", e))?;
    let exercise_id = generate_uuid();
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

    db.prepare(
        "INSERT INTO exercises (id, title, description, instructions, difficulty, category, source, created_at) \
         VALUES (?1, ?2, ?3, ?4, 'intermediate', 'generated', 'teacher_llm', ?5)",
    )
    .bind(&[
        JsValue::from_str(&exercise_id),
        JsValue::from_str(title),
        JsValue::from_str(&format!("{} -- {}", target_skill, source_passage)),
        JsValue::from_str(instruction),
        JsValue::from_str(&now),
    ])
    .map_err(|e| format!("Failed to bind exercise insert: {:?}", e))?
    .run()
    .await
    .map_err(|e| format!("Failed to insert exercise: {:?}", e))?;

    // Link to dimension
    let _ = db
        .prepare("INSERT INTO exercise_dimensions (exercise_id, dimension) VALUES (?1, ?2)")
        .bind(&[
            JsValue::from_str(&exercise_id),
            JsValue::from_str(focus_dimension),
        ])
        .ok()
        .map(|q| q.run());

    Ok(exercise_id)
}
```

- [ ] **Step 4: Update `handle_ask_inner` to use tool-enabled call**

Replace the Stage 2 teacher call in `handle_ask_inner` (lines 162-195). The new version:

1. Looks up catalog exercises matching the dimension
2. Builds the catalog-aware prompt
3. Calls `call_anthropic_with_tools()` instead of `call_anthropic()`
4. Parses tool_use blocks into `ExerciseSetConfig` JSON
5. Persists generated exercises to D1 and attaches `exercise_id`s

```rust
    // Stage 2: Teacher (Anthropic) with tool use
    let catalog = lookup_catalog_exercises(env, &dimension, "intermediate").await;

    let teacher_user_prompt = if catalog.is_empty() {
        prompts::build_teacher_user_prompt(
            &subagent_json,
            &subagent_narrative,
            "intermediate",
            "",
        )
    } else {
        prompts::build_teacher_user_prompt_with_catalog(
            &subagent_json,
            &subagent_narrative,
            "intermediate",
            "",
            &catalog,
        )
    };

    let tools = vec![prompts::exercise_tool_definition()];

    let teacher_result = llm::call_anthropic_with_tools(
        env,
        prompts::TEACHER_SYSTEM,
        &teacher_user_prompt,
        500,  // slightly higher to accommodate tool use
        Some(tools),
    ).await;

    let (observation_text, is_fallback, components_json) = match teacher_result {
        Ok(result) => {
            let text = post_process_observation(&result.text);

            // Process any exercise tool calls
            let components = if let Some(tc) = result.tool_calls.first() {
                if tc.name == "create_exercise" {
                    match process_exercise_tool_call(env, &tc.input).await {
                        Ok(json) => Some(json),
                        Err(e) => {
                            console_error!("Exercise tool call processing failed: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            (text, false, components)
        }
        Err(e) => {
            console_error!("Teacher LLM failed: {}", e);
            (fallback_observation(&dimension), true, None)
        }
    };

    let reasoning_trace = serde_json::json!({
        "subagent_output": subagent_output,
    }).to_string();

    AskInnerResponse {
        observation_text,
        dimension,
        framing,
        reasoning_trace,
        is_fallback,
        components_json,
    }
```

- [ ] **Step 5: Add exercise tool call processor**

```rust
/// Process a create_exercise tool call: validate, persist each exercise, return components JSON.
async fn process_exercise_tool_call(
    env: &Env,
    input: &serde_json::Value,
) -> Result<String, String> {
    let source_passage = input.get("source_passage")
        .and_then(|v| v.as_str())
        .ok_or("Missing source_passage")?;
    let target_skill = input.get("target_skill")
        .and_then(|v| v.as_str())
        .ok_or("Missing target_skill")?;
    let exercises = input.get("exercises")
        .and_then(|v| v.as_array())
        .ok_or("Missing exercises array")?;

    let mut processed_exercises = Vec::new();

    for ex in exercises {
        let title = ex.get("title").and_then(|v| v.as_str()).unwrap_or("Practice Drill");
        let instruction = ex.get("instruction").and_then(|v| v.as_str()).unwrap_or("");
        let focus_dim = ex.get("focus_dimension").and_then(|v| v.as_str()).unwrap_or("dynamics");
        let hands = ex.get("hands").and_then(|v| v.as_str());

        // Check if this references a catalog exercise by ID
        let exercise_id = if let Some(id) = ex.get("exercise_id").and_then(|v| v.as_str()) {
            id.to_string()
        } else {
            // Generate and persist new exercise
            persist_generated_exercise(env, title, instruction, focus_dim, source_passage, target_skill)
                .await
                .unwrap_or_else(|e| {
                    console_error!("Failed to persist exercise: {}", e);
                    generate_uuid()  // ID without persistence -- assignment will fail gracefully
                })
        };

        let mut ex_json = serde_json::json!({
            "title": title,
            "instruction": instruction,
            "focus_dimension": focus_dim,
            "exercise_id": exercise_id,
        });
        if let Some(h) = hands {
            ex_json["hands"] = serde_json::json!(h);
        }
        processed_exercises.push(ex_json);
    }

    let component = serde_json::json!([{
        "type": "exercise_set",
        "config": {
            "source_passage": source_passage,
            "target_skill": target_skill,
            "exercises": processed_exercises,
        }
    }]);

    serde_json::to_string(&component)
        .map_err(|e| format!("Failed to serialize components: {:?}", e))
}
```

- [ ] **Step 6: Fix existing `AskInnerResponse` usage in `handle_ask`**

At line 259, after `let is_fallback = inner_resp.is_fallback;`, add:

```rust
    let _components_json = inner_resp.components_json;
    // HTTP path does not use components yet (observations go through DO WebSocket)
```

- [ ] **Step 7: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 8: Commit**

```bash
git add apps/api/src/services/ask.rs
git commit -m "feat: extend teacher pipeline with tool_use parsing and exercise persistence"
```

---

## Task 5: Link Practice Sessions to Conversations

**Files:**
- Modify: `apps/api/src/practice/start.rs`
- Modify: `apps/api/src/server.rs:98-105`

**Note:** The current `handle_start` does NOT accept a request body (signature is `handle_start(env, headers)`) and does NOT insert into the `sessions` table. The DO creates session state implicitly. We need to: (1) change the function signature, (2) update the caller in `server.rs`, and (3) pass `conversation_id` to the DO via WebSocket URL query params.

- [ ] **Step 1: Update `handle_start` signature to accept body**

In `start.rs`, change the function signature from:
```rust
pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
) -> http::Response<axum::body::Body> {
```
to:
```rust
pub async fn handle_start(
    env: &Env,
    headers: &http::HeaderMap,
    body: &[u8],
) -> http::Response<axum::body::Body> {
```

- [ ] **Step 2: Parse optional `conversation_id` and create conversation if absent**

After auth validation (line 14), add:

```rust
    // Parse optional conversation_id from body
    let body_json: serde_json::Value = serde_json::from_slice(body)
        .unwrap_or(serde_json::json!({}));
    let provided_conv_id = body_json
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Use provided conversation_id or create a new conversation
    let conversation_id = match provided_conv_id {
        Some(id) => id,
        None => {
            let new_conv_id = crate::services::ask::generate_uuid();
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
            if let Ok(db) = env.d1("DB") {
                let _ = db.prepare(
                    "INSERT INTO conversations (id, student_id, title, created_at, updated_at) \
                     VALUES (?1, ?2, NULL, ?3, ?3)"
                )
                .bind(&[
                    JsValue::from_str(&new_conv_id),
                    JsValue::from_str(&_student_id),
                    JsValue::from_str(&now),
                ])
                .ok()
                .map(|q| wasm_bindgen_futures::spawn_local(async move { let _ = q.run().await; }));
            }
            new_conv_id
        }
    };

    // Insert session row linked to conversation
    let session_id = crate::services::ask::generate_uuid();
    let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
    if let Ok(db) = env.d1("DB") {
        let _ = db.prepare(
            "INSERT INTO sessions (id, student_id, conversation_id, started_at) \
             VALUES (?1, ?2, ?3, ?4)"
        )
        .bind(&[
            JsValue::from_str(&session_id),
            JsValue::from_str(&_student_id),
            JsValue::from_str(&conversation_id),
            JsValue::from_str(&now),
        ])
        .ok()
        .map(|q| wasm_bindgen_futures::spawn_local(async move { let _ = q.run().await; }));
    }
```

Update the response to include both IDs:
```rust
    let resp = serde_json::json!({
        "sessionId": session_id,
        "conversationId": conversation_id,
    });
```

- [ ] **Step 3: Insert "session_start" message into conversation**

After creating the session:
```rust
    // Insert session_start message
    if let Ok(db) = env.d1("DB") {
        let msg_id = crate::services::ask::generate_uuid();
        let _ = db.prepare(
            "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
             VALUES (?1, ?2, 'assistant', 'Recording started', 'session_start', ?3, ?4)"
        )
        .bind(&[
            JsValue::from_str(&msg_id),
            JsValue::from_str(&conversation_id),
            JsValue::from_str(&session_id),
            JsValue::from_str(&now),
        ])
        .ok()
        .map(|q| wasm_bindgen_futures::spawn_local(async move { let _ = q.run().await; }));
    }
```

- [ ] **Step 4: Update `server.rs` to pass body to `handle_start`**

In `server.rs` at lines 98-105, change:
```rust
    if path == "/api/practice/start" && method == http::Method::POST {
        let headers = req.headers().clone();
        return into_worker_response(with_cors(
            crate::practice::start::handle_start(&env, &headers).await,
            origin.as_deref(),
        )).await;
    }
```
to:
```rust
    if path == "/api/practice/start" && method == http::Method::POST {
        let headers = req.headers().clone();
        let body = req
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes().to_vec())
            .unwrap_or_default();
        return into_worker_response(with_cors(
            crate::practice::start::handle_start(&env, &headers, &body).await,
            origin.as_deref(),
        )).await;
    }
```

- [ ] **Step 5: Pass `conversation_id` to DO via WebSocket URL**

In `server.rs` at lines 86-94, update the DO URL to include `conversation_id`. The client will need to pass it as a query param when connecting:

In `server.rs`, the WebSocket URL construction at line 88-91 changes to:
```rust
    // Extract conversation_id from query params (set by client)
    let ws_query = req.uri().query().unwrap_or("");
    let conv_id = ws_query
        .split('&')
        .find_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            (k == "conversationId").then(|| v.to_string())
        })
        .unwrap_or_default();

    let url = format!(
        "https://do.internal/ws/{}?student_id={}&conversation_id={}",
        session_id, student_id, conv_id
    );
```

- [ ] **Step 6: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/practice/start.rs apps/api/src/server.rs
git commit -m "feat: link practice sessions to conversations, pass conversation_id to DO"
```

---

## Task 6: Persist Observations as Messages in DO

**Files:**
- Modify: `apps/api/src/practice/session.rs:37-90` (structs), `apps/api/src/practice/session.rs:109-127` (fetch/init), `apps/api/src/practice/session.rs:667-725` (observation generation), `apps/api/src/practice/session.rs:977-1059` (finalization)

**Prerequisite:** Tasks 4 and 5 must be complete (AskInnerResponse has `components_json`, and `conversation_id` is passed via WebSocket URL query params).

- [ ] **Step 1: Add `conversation_id` to `SessionState` and `ObservationRecord`**

In `SessionState` struct (line 49), add:
```rust
    conversation_id: Option<String>,
```

Initialize as `None` in the `Default` impl (line 69).

In `ObservationRecord` struct (line 37), add:
```rust
    pub components_json: Option<String>,
```

- [ ] **Step 2: Extract `conversation_id` from WebSocket URL in `fetch()`**

In the DO's `fetch` method (line 109-127), after extracting `student_id` from query params, add:

```rust
        let conversation_id = url.query_pairs()
            .find(|(k, _)| k == "conversation_id")
            .map(|(_, v)| v.to_string())
            .filter(|s| !s.is_empty());
```

And in the `if s.session_id.is_empty()` block (line 123-126), add:
```rust
                s.conversation_id = conversation_id;
```

- [ ] **Step 3: Persist observation as message in D1**

In `generate_observation()`, restructure the observation creation to extract the ID before borrowing mutably. Replace the current observation storage block (lines 710-724):

```rust
        // Generate observation ID and build record
        let obs_id = crate::services::ask::generate_uuid();
        let observation_text = inner_resp.observation_text.clone();
        let obs_dimension = inner_resp.dimension.clone();
        let obs_framing = inner_resp.framing.clone();
        let components_json = inner_resp.components_json.clone();

        // Store in session state
        {
            let mut s = self.inner.borrow_mut();
            s.observations.push(ObservationRecord {
                id: obs_id.clone(),
                text: observation_text.clone(),
                dimension: obs_dimension.clone(),
                framing: obs_framing.clone(),
                chunk_index: moment.chunk_index,
                score: moment.score,
                baseline: moment.baseline,
                reasoning_trace: inner_resp.reasoning_trace.clone(),
                is_fallback: inner_resp.is_fallback,
                components_json: components_json.clone(),
            });
            s.last_observation_at = Some(js_sys::Date::now() as u64);
        }

        // Persist as message in conversation (awaited to prevent data loss)
        let conv_id = self.inner.borrow().conversation_id.clone();
        if let Some(ref conv_id) = conv_id {
            let msg_id = crate::services::ask::generate_uuid();
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
            if let Ok(db) = self.env.d1("DB") {
                let bind_result = db.prepare(
                    "INSERT INTO messages (id, conversation_id, role, content, message_type, \
                     dimension, framing, components_json, session_id, observation_id, created_at) \
                     VALUES (?1, ?2, 'assistant', ?3, 'observation', ?4, ?5, ?6, ?7, ?8, ?9)"
                )
                .bind(&[
                    JsValue::from_str(&msg_id),
                    JsValue::from_str(conv_id),
                    JsValue::from_str(&observation_text),
                    JsValue::from_str(&obs_dimension),
                    JsValue::from_str(&obs_framing),
                    match components_json.as_deref() {
                        Some(c) => JsValue::from_str(c),
                        None => JsValue::NULL,
                    },
                    JsValue::from_str(&session_id),
                    JsValue::from_str(&obs_id),
                    JsValue::from_str(&now),
                ]);
                if let Ok(q) = bind_result {
                    if let Err(e) = q.run().await {
                        console_error!("Failed to persist observation message: {:?}", e);
                    }
                }
            }
        }
```

- [ ] **Step 3: Include components in WebSocket observation event**

Update the `obs_event` JSON at line 677:

```rust
let mut obs_event = serde_json::json!({
    "type": "observation",
    "text": inner_resp.observation_text,
    "dimension": inner_resp.dimension,
    "framing": inner_resp.framing,
});
if let Some(br) = chunk_analysis.and_then(|a| a.bar_range.as_ref()) {
    obs_event["barRange"] = serde_json::json!(br);
}
if let Some(ref components) = inner_resp.components_json {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(components) {
        obs_event["components"] = parsed;
    }
}
```

- [ ] **Step 4: Persist summary and session_end as messages in `finalize_session`**

In `finalize_session()` (line 977), after the summary WebSocket send (around line 1052), persist messages. **All D1 writes must be `.await`ed** to prevent data loss before the Worker terminates:

```rust
        // Persist summary + session_end as messages in conversation
        let conv_id = self.inner.borrow().conversation_id.clone();
        if let Some(ref conv_id) = conv_id {
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();

            if !summary_text.is_empty() {
                if let Ok(db) = self.env.d1("DB") {
                    let msg_id = crate::services::ask::generate_uuid();
                    if let Ok(q) = db.prepare(
                        "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
                         VALUES (?1, ?2, 'assistant', ?3, 'summary', ?4, ?5)"
                    )
                    .bind(&[
                        JsValue::from_str(&msg_id),
                        JsValue::from_str(conv_id),
                        JsValue::from_str(&summary_text),
                        JsValue::from_str(&session_id),
                        JsValue::from_str(&now),
                    ]) {
                        if let Err(e) = q.run().await {
                            console_error!("Failed to persist summary message: {:?}", e);
                        }
                    }
                }
            }

            // Insert session_end message
            if let Ok(db) = self.env.d1("DB") {
                let end_msg_id = crate::services::ask::generate_uuid();
                if let Ok(q) = db.prepare(
                    "INSERT INTO messages (id, conversation_id, role, content, message_type, session_id, created_at) \
                     VALUES (?1, ?2, 'assistant', 'Recording ended', 'session_end', ?3, ?4)"
                )
                .bind(&[
                    JsValue::from_str(&end_msg_id),
                    JsValue::from_str(conv_id),
                    JsValue::from_str(&session_id),
                    JsValue::from_str(&now),
                ]) {
                    if let Err(e) = q.run().await {
                        console_error!("Failed to persist session_end message: {:?}", e);
                    }
                }
            }
        }
```

- [ ] **Step 5: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat: persist observations and summaries as messages, send components via WS"
```

---

## Task 7: Chat Handler Loads All Message Types

**Files:**
- Modify: `apps/api/src/services/chat.rs`

- [ ] **Step 1: Read current chat.rs**

Read `apps/api/src/services/chat.rs` completely to understand the current message loading. Identify:
- The `MessageRow` struct (needs new fields)
- All places where messages are queried (at least 3: `fetch_messages()`, `handle_get_conversation()`, streaming chat history)
- The `ConversationWithMessages` response struct (needs new fields for client)

- [ ] **Step 2: Update `MessageRow` struct**

Add the new columns to the deserialization struct:
```rust
#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct MessageRow {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
    #[serde(default)]
    pub message_type: Option<String>,
    #[serde(default)]
    pub dimension: Option<String>,
    #[serde(default)]
    pub framing: Option<String>,
    #[serde(default)]
    pub components_json: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}
```

- [ ] **Step 3: Update ALL conversation history queries**

Find every query that loads from `messages` table and extend the SELECT clause:

```sql
SELECT id, role, content, created_at, message_type, dimension, framing, components_json, session_id
FROM messages
WHERE conversation_id = ?1
ORDER BY created_at ASC
```

This affects:
- `fetch_messages()` (or equivalent function)
- `handle_get_conversation()` (API response)
- The streaming chat handler's history fetch (builds LLM context)

- [ ] **Step 4: Transform observation messages for LLM context**

In the streaming chat handler's message-to-LLM loop (where `MessageRow` becomes `LlmMessage`), prefix observation content so the teacher knows what happened during practice:

```rust
let llm_content = match msg.message_type.as_deref() {
    Some("observation") => {
        let dim = msg.dimension.as_deref().unwrap_or("unknown");
        format!("[Practice observation on {}]: {}", dim, msg.content)
    }
    Some("session_start") | Some("session_end") => {
        format!("[{}]", msg.content)  // Minimal marker
    }
    Some("summary") | Some("chat") | None => msg.content.clone(),
    Some(_) => msg.content.clone(),
};
```

- [ ] **Step 5: Update the API response**

Ensure `handle_get_conversation()` returns the full `MessageRow` (including `message_type`, `dimension`, `components_json`) so the web client can render observation messages with appropriate styling and artifacts.

- [ ] **Step 6: Verify compilation**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/services/chat.rs
git commit -m "feat: load all message types in chat conversation history"
```

---

## Task 8: Web Types + Practice API Updates

**Files:**
- Modify: `apps/web/src/lib/types.ts`
- Modify: `apps/web/src/lib/practice-api.ts`

- [ ] **Step 1: Extend RichMessage type**

In `types.ts`, `RichMessage` already has `components?: InlineComponent[]` and `dimension?: string`. Add only the NEW fields:

```typescript
export interface RichMessage {
	id: string;
	role: "user" | "assistant";
	content: string;
	created_at: string;
	streaming?: boolean;
	components?: InlineComponent[];      // EXISTING
	dimension?: string;                  // EXISTING
	message_type?: "chat" | "observation" | "session_start" | "session_end" | "summary";  // NEW
	session_id?: string;                 // NEW
	framing?: string;                    // NEW
}
```

- [ ] **Step 2: Extend PracticeWsEvent and ObservationEvent**

In `practice-api.ts`, add `components` to the observation event type and `conversationId` to start:

```typescript
export interface ObservationEvent {
	text: string;
	dimension: string;
	framing: string;
	barRange?: [number, number];
	components?: InlineComponent[];
}

export type PracticeWsEvent =
	| { type: "connected" }
	| { type: "chunk_processed"; index: number; scores: DimScores }
	| {
			type: "observation";
			text: string;
			dimension: string;
			framing: string;
			barRange?: string;
			components?: InlineComponent[];
	  }
	// ... rest unchanged
```

Update `practiceApi.start()` to accept and return `conversationId`:

```typescript
export interface PracticeStartResponse {
	sessionId: string;
	conversationId: string;
}

export const practiceApi = {
	async start(conversationId?: string): Promise<PracticeStartResponse> {
		const res = await fetch(`${API_BASE}/api/practice/start`, {
			method: "POST",
			credentials: "include",
			headers: conversationId ? { "Content-Type": "application/json" } : undefined,
			body: conversationId ? JSON.stringify({ conversation_id: conversationId }) : undefined,
		});
		if (!res.ok) throw new Error(`Failed to start session: ${res.status}`);
		return res.json();
	},
	// ... rest unchanged
```

- [ ] **Step 3: Add InlineComponent import to practice-api.ts**

```typescript
import type { InlineComponent } from "./types";
```

- [ ] **Step 4: Verify build**

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds (types are additive, no breaking changes)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/types.ts apps/web/src/lib/practice-api.ts
git commit -m "feat: extend web types for unified messages and components"
```

---

## Task 9: Update usePracticeSession Hook

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`

- [ ] **Step 1: Read current hook completely**

Read `apps/web/src/hooks/usePracticeSession.ts` to understand the full recording state machine.

- [ ] **Step 2: Pass conversationId through the hook**

The hook should accept an optional `conversationId` and pass it to `practiceApi.start()`. Update the `start()` function inside the hook:

```typescript
const start = useCallback(async (conversationId?: string) => {
    // ... existing mic setup ...
    const { sessionId, conversationId: returnedConvId } = await practiceApi.start(conversationId);
    // Store both IDs in refs
```

- [ ] **Step 3: Parse components from WebSocket observations**

In the `handleWsMessage` observation case (currently around line 215):

```typescript
case "observation": {
    const obs: ObservationEvent = {
        text: data.text,
        dimension: data.dimension,
        framing: data.framing,
        components: data.components,
    };
    const immediate = throttleRef.current.enqueue(obs);
    setObservations((prev) => [...prev, immediate]);
    break;
}
```

- [ ] **Step 4: Capture timestamps on observation arrival**

In the `handleWsMessage` observation case, capture the arrival time (not render time):

```typescript
case "observation": {
    const obs: ObservationEvent & { arrived_at: string } = {
        text: data.text,
        dimension: data.dimension,
        framing: data.framing,
        components: data.components,
        arrived_at: new Date().toISOString(),
    };
    // ...
```

- [ ] **Step 5: Expose observations as RichMessages**

Add a derived state that converts observations into `RichMessage[]` for direct chat integration. Use the captured `arrived_at` timestamp, NOT the current render time:

```typescript
const observationMessages: RichMessage[] = observations.map((obs, i) => ({
    id: `obs-${sessionIdRef.current}-${i}`,
    role: "assistant" as const,
    content: obs.text,
    created_at: (obs as any).arrived_at || new Date().toISOString(),
    message_type: "observation" as const,
    dimension: obs.dimension,
    framing: obs.framing,
    components: obs.components,
}));
```

- [ ] **Step 6: Update `UsePracticeSessionReturn` type**

Update the hook's return type to include the new fields:

```typescript
interface UsePracticeSessionReturn {
    start: (conversationId?: string) => Promise<void>;  // was: () => Promise<void>
    // ... existing fields ...
    observationMessages: RichMessage[];  // NEW
    conversationId: string | null;       // NEW -- the conversation linked to this session
}
```

Return `observationMessages` and `conversationId` from the hook.

- [ ] **Step 7: Update WebSocket URL to include conversationId**

In the hook's WebSocket connection (where `practiceApi.connectWebSocket(sessionId)` is called), pass the `conversationId` as a query param so the DO receives it:

```typescript
// Update practiceApi.connectWebSocket to accept conversationId
connectWebSocket(sessionId: string, conversationId?: string): WebSocket {
    const url = conversationId
        ? `${WS_BASE}/api/practice/ws/${sessionId}?conversationId=${conversationId}`
        : `${WS_BASE}/api/practice/ws/${sessionId}`;
    return new WebSocket(url);
},
```

- [ ] **Step 8: Verify build**

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 9: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/lib/practice-api.ts
git commit -m "feat: pass conversationId, parse components, expose observation messages"
```

---

## Task 10: Inline Recording Banner (Replace Overlay)

**Files:**
- Create: `apps/web/src/components/RecordingBanner.tsx`
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Create RecordingBanner component**

A compact recording indicator that sits above the chat input. Shows waveform visualization, elapsed time, and stop button. Does NOT take over the viewport.

```typescript
// RecordingBanner.tsx
interface RecordingBannerProps {
    elapsedSeconds: number;
    analyserNode: AnalyserNode | null;
    onStop: () => void;
    latestScores?: DimScores;
}
```

Key design: horizontal bar (h-16) with:
- Small waveform visualization (reuse WaveformVisualizer at reduced size)
- Timer display (MM:SS)
- Stop button (red circle)
- Optional: latest dimension scores as small pills

- [ ] **Step 2: Update AppChat to use RecordingBanner instead of ListeningMode**

In `AppChat.tsx`:
- Remove `showListeningMode` state, `handleExitListeningMode`, and the `<ListeningMode>` overlay render (currently at lines ~723-738)
- Remove the `ListeningMode` import (line ~33)
- **Do NOT delete `ListeningMode.tsx`** yet -- deprecate it (can be removed in a follow-up)
- Add `RecordingBanner` above `ChatInput` when `practice.isRecording` is true
- Pass the active `conversationId` to `practice.start(conversationId)` in `handleRecord`
- Merge `practice.observationMessages` into the main messages array
- **Deduplication:** When conversation is reloaded from D1 (e.g., on page refresh), observations are already in the message list from the server. Only merge `practice.observationMessages` for messages NOT yet in the server-loaded messages array (deduplicate by `observation_id` or message content + timestamp)

The chat area stays fully visible during recording. Observations appear as chat messages in real-time.

- [ ] **Step 3: Render observation messages with appropriate styling**

In `ChatMessages.tsx`, handle `message_type === 'observation'` with a dimension badge and optional artifact:

```typescript
{message.message_type === 'observation' && message.dimension && (
    <span className="text-xs px-2 py-0.5 rounded-full bg-surface-2 text-secondary">
        {message.dimension}
    </span>
)}
```

Session start/end messages render as subtle dividers:
```typescript
{(message.message_type === 'session_start' || message.message_type === 'session_end') && (
    <div className="text-center text-xs text-tertiary py-2">
        {message.content}
    </div>
)}
```

- [ ] **Step 4: Verify build and test**

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds

Run: `cd apps/web && bun run test`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/RecordingBanner.tsx apps/web/src/components/AppChat.tsx apps/web/src/components/ChatMessages.tsx
git commit -m "feat: replace ListeningMode overlay with inline RecordingBanner, render observation messages in chat"
```

---

## Task 11: Integration Test

**Files:**
- Modify: `apps/web/src/stores/artifact.test.ts` (verify existing tests still pass)
- Manual testing with local wrangler dev

- [ ] **Step 1: Run all web tests**

Run: `cd apps/web && bun run test`
Expected: All tests pass (artifact store + throttle)

- [ ] **Step 2: Run cargo check**

Run: `cd apps/api && cargo check 2>&1 | tail -5`
Expected: `Finished` with no errors

- [ ] **Step 3: Local smoke test**

Start the API:
```bash
cd apps/api && npx wrangler dev
```

Start the web app:
```bash
cd apps/web && bun run dev
```

Test flow:
1. Open `localhost:3000`, sign in
2. Start a conversation (type a message)
3. Click record -- banner should appear above input, chat stays visible
4. Play piano for 15+ seconds
5. Observation should appear as a chat message (not a toast)
6. If teacher creates an exercise, artifact card should render inline
7. Stop recording -- summary appears as a message
8. Type a follow-up -- teacher should reference what it said during practice

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration test fixes for unified chat"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `docs/apps/00-status.md`
- Modify: `docs/apps/05-ui-system.md`

- [ ] **Step 1: Update 00-status.md**

Mark as COMPLETE:
- Unified chat + practice session
- Teacher tool use (create_exercise)
- Exercise artifact rendering

Update the artifact system status from "pattern TBD" to "Anthropic tool_use, teacher autonomous, hybrid catalog".

- [ ] **Step 2: Update 05-ui-system.md**

Resolve the "Open Decision" (tool_use vs MCP vs structured output):
- Decision: Anthropic native tool_use with `tool_choice: "auto"`
- Teacher decides autonomously when to create artifacts
- Hybrid: catalog lookup + generated fallback
- Separate tools per artifact type (Phase 3: `create_score_highlight`, etc.)

- [ ] **Step 3: Commit**

```bash
git add docs/apps/00-status.md docs/apps/05-ui-system.md
git commit -m "docs: update status and UI system docs for unified chat + tool use"
```
