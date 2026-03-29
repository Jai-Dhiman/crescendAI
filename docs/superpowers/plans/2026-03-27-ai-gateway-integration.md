# AI Gateway Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route all LLM calls through Cloudflare AI Gateway for observability, caching, rate limiting, and automatic Groq->Workers AI fallback; migrate Workers AI background tasks to cheaper Qwen3-30b model; add shadow benchmarking to compare Groq vs Workers AI latency.

**Architecture:** Thin `AiGateway` struct in `llm.rs` constructs gateway URLs, attaches caching/observability headers, and handles response metadata logging. Existing `call_*` functions swap their base URL from direct provider endpoints to gateway endpoints. No request/response format changes -- Anthropic keeps native format, Groq keeps OpenAI-compatible format. Workers AI migrates from `env.ai()` binding to HTTP through the gateway for unified observability.

**Tech Stack:** Rust (WASM target), Cloudflare Workers, AI Gateway, `worker::Fetch`, `getrandom` (already in deps)

**Spec:** `docs/superpowers/specs/2026-03-27-ai-gateway-integration-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `apps/api/wrangler.toml` | Modify | Add `CF_ACCOUNT_ID`, `SHADOW_BENCHMARK_ENABLED`, `SHADOW_BENCHMARK_PCT` vars |
| `apps/api/src/services/llm.rs` | Modify | `AiGateway` struct, route all `call_*` functions through gateway, shadow benchmarking |
| `apps/api/src/services/chat.rs` | Modify | Pass model constant to `call_workers_ai` (title generation) |
| `apps/api/src/services/goals.rs` | Modify | Pass model constant to `call_workers_ai` (goal extraction) |
| `apps/api/src/services/memory.rs` | Modify | Pass model constant to `call_workers_ai` (3 call sites) |

No new files. No test files (WASM target -- verification is compile + deploy + manual spot-check).

---

### Task 1: Add Gateway Config to wrangler.toml

**Files:**
- Modify: `apps/api/wrangler.toml:19-24`

- [ ] **Step 1: Add new vars to wrangler.toml**

Add the three new config vars to the existing `[vars]` section:

```toml
[vars]
APPLE_WEB_SERVICES_ID = "ai.crescend.web"
ENVIRONMENT = "production"
GOOGLE_CLIENT_ID = "726399246264-lkq6jl4khal8gagn65j9t2a8bg52i7ie.apps.googleusercontent.com"
HF_INFERENCE_ENDPOINT = "https://mxcyiqltad84v9w1.us-east4.gcp.endpoints.huggingface.cloud"
HF_AMT_ENDPOINT = ""  # Set after deploying Aria-AMT HF endpoint
CF_ACCOUNT_ID = ""  # Set via `wrangler secret put` or CF dashboard
SHADOW_BENCHMARK_ENABLED = "false"
SHADOW_BENCHMARK_PCT = "10"
```

Note: `CF_ACCOUNT_ID` should be set as a secret in production (it's not sensitive but keeps config clean). `SHADOW_BENCHMARK_ENABLED` starts as `"false"` -- enable after gateway setup is verified.

- [ ] **Step 2: Commit**

```bash
git add apps/api/wrangler.toml
git commit -m "feat: add AI Gateway config vars to wrangler.toml"
```

---

### Task 2: Create AiGateway Struct and Helpers

**Files:**
- Modify: `apps/api/src/services/llm.rs:1-6` (add imports, constants, struct)

- [ ] **Step 1: Add gateway constants and model constants at top of llm.rs**

Add after the existing imports (line 6):

```rust
use worker::{console_log, Env, Fetch, Headers, Method, Request, RequestInit, Url};

// --- AI Gateway ---

/// Gateway IDs (created in CF dashboard)
const TEACHER_GATEWAY: &str = "crescendai-teacher";
const BACKGROUND_GATEWAY: &str = "crescendai-background";

/// Default cache TTL in seconds for all gateway requests
const GATEWAY_CACHE_TTL: u32 = 60;

/// Workers AI model for cheap background tasks (titles, goals, memory extraction)
pub const WORKERS_AI_CHEAP_MODEL: &str = "@cf/qwen/qwen3-30b-a3b-fp8";

/// Workers AI model matching Groq's Llama 3.3 70B (used as Groq fallback + shadow benchmark)
const WORKERS_AI_GROQ_FALLBACK_MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
```

- [ ] **Step 2: Add AiGateway struct and constructor**

Add after the constants:

```rust
/// Thin client for routing LLM calls through Cloudflare AI Gateway.
/// Constructs gateway URLs, attaches caching headers, logs response metadata.
struct AiGateway {
    account_id: String,
    gateway_id: &'static str,
}

impl AiGateway {
    /// Create a gateway client for the teacher path (Anthropic only).
    fn teacher(env: &Env) -> Result<Self, String> {
        let account_id = env
            .var("CF_ACCOUNT_ID")
            .map_err(|_| "CF_ACCOUNT_ID not configured".to_string())?
            .to_string();
        Ok(Self {
            account_id,
            gateway_id: TEACHER_GATEWAY,
        })
    }

    /// Create a gateway client for background tasks (Groq + Workers AI).
    fn background(env: &Env) -> Result<Self, String> {
        let account_id = env
            .var("CF_ACCOUNT_ID")
            .map_err(|_| "CF_ACCOUNT_ID not configured".to_string())?
            .to_string();
        Ok(Self {
            account_id,
            gateway_id: BACKGROUND_GATEWAY,
        })
    }

    /// Build a provider-specific gateway URL.
    /// Example: https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/anthropic/v1/messages
    fn provider_url(&self, provider: &str, path: &str) -> Result<Url, String> {
        let url_str = format!(
            "https://gateway.ai.cloudflare.com/v1/{}/{}/{}/{}",
            self.account_id, self.gateway_id, provider, path
        );
        url_str
            .parse()
            .map_err(|e| format!("Invalid gateway URL: {:?}", e))
    }

    /// Attach gateway-specific headers (caching) to an existing Headers object.
    fn attach_headers(&self, headers: &Headers) -> Result<(), String> {
        headers
            .set("cf-aig-cache-ttl", &GATEWAY_CACHE_TTL.to_string())
            .map_err(|e| format!("Failed to set cache TTL header: {:?}", e))?;
        Ok(())
    }

    /// Log gateway response metadata (cache status, provider step).
    fn log_response_metadata(&self, response: &worker::Response, label: &str) {
        let cache_status = response
            .headers()
            .get("cf-aig-cache-status")
            .ok()
            .flatten()
            .unwrap_or_else(|| "unknown".to_string());
        let step = response
            .headers()
            .get("cf-aig-step")
            .ok()
            .flatten()
            .unwrap_or_else(|| "0".to_string());
        console_log!(
            "gateway[{}] {}: cache={}, provider_step={}",
            self.gateway_id,
            label,
            cache_status,
            step
        );
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: compiles with no errors related to `AiGateway` (existing `Result<T, String>` warnings are expected and pre-existing)

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: add AiGateway struct with URL construction and header helpers"
```

---

### Task 3: Route Anthropic Calls Through Gateway

**Files:**
- Modify: `apps/api/src/services/llm.rs` (3 functions: `call_anthropic`, `call_anthropic_with_tools`, `call_anthropic_stream`)

The change in each function is identical: replace the hardcoded `https://api.anthropic.com/v1/messages` URL with the gateway URL, attach caching headers, and log response metadata. No request/response format changes.

- [ ] **Step 1: Update `call_anthropic` (line ~140)**

Replace the URL construction and add gateway:

```rust
pub async fn call_anthropic(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let gateway = AiGateway::teacher(env)?;
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
        .to_string();

    let request_body = AnthropicRequest {
        model: "claude-sonnet-4-6".to_string(),
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
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("anthropic", "v1/messages")?;

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

    gateway.log_response_metadata(&response, "anthropic");

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

    let anthropic_response: AnthropicResponse =
        serde_json::from_str(&response_text).map_err(|e| {
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
```

- [ ] **Step 2: Update `call_anthropic_with_tools` (line ~272)**

Same pattern -- add `let gateway = AiGateway::teacher(env)?;` at the top, replace URL with `gateway.provider_url("anthropic", "v1/messages")?`, add `gateway.attach_headers(&headers)?` after setting Content-Type, add `gateway.log_response_metadata(&response, "anthropic-tools")` after the fetch.

The full function with changes (replacing the entire function):

```rust
pub async fn call_anthropic_with_tools(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    max_tokens: u32,
    tools: Option<Vec<AnthropicTool>>,
) -> Result<AnthropicToolResult, String> {
    let gateway = AiGateway::teacher(env)?;
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
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("anthropic", "v1/messages")?;

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

    gateway.log_response_metadata(&response, "anthropic-tools");

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

    let parsed: AnthropicToolResponse = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Failed to parse Anthropic tool response: {:?} - body: {}",
            e, response_text
        )
    })?;

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in parsed.content {
        match block {
            AnthropicContentBlock::Text { text } => text_parts.push(text),
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

- [ ] **Step 3: Update `call_anthropic_stream` (line ~470)**

Same pattern. The streaming response passes through the gateway transparently:

```rust
pub async fn call_anthropic_stream(
    env: &Env,
    system_prompt: &str,
    messages: Vec<LlmMessage>,
    max_tokens: u32,
) -> Result<worker::Response, String> {
    let gateway = AiGateway::teacher(env)?;
    let api_key = env
        .secret("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not configured".to_string())?
        .to_string();

    let request_body = AnthropicStreamRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens,
        system: vec![AnthropicSystemBlock {
            block_type: "text".to_string(),
            text: system_prompt.to_string(),
            cache_control: Some(CacheControl {
                control_type: "ephemeral".to_string(),
            }),
        }],
        messages,
        stream: true,
    };

    let body_json = serde_json::to_string(&request_body)
        .map_err(|e| format!("Failed to serialize Anthropic stream request: {:?}", e))?;

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
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("anthropic", "v1/messages")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Anthropic stream request: {:?}", e))?;

    let response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Anthropic stream request failed: {:?}", e))?;

    gateway.log_response_metadata(&response, "anthropic-stream");

    let status = response.status_code();
    if status != 200 {
        return Err(format!("Anthropic streaming returned status {}", status));
    }

    Ok(response)
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -10`
Expected: compiles with no new errors

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: route Anthropic calls through AI Gateway (teacher)"
```

---

### Task 4: Route Groq Through Gateway with Fallback

**Files:**
- Modify: `apps/api/src/services/llm.rs` (`call_groq` function, line ~35)

The Groq call uses the gateway's provider-specific endpoint (not the universal endpoint). Fallback is configured at the gateway level in the CF dashboard, not in the request body. This keeps the code change minimal -- the gateway itself handles trying Workers AI when Groq fails.

- [ ] **Step 1: Update `call_groq` to route through background gateway**

Replace the entire `call_groq` function:

```rust
/// Call the Groq API via AI Gateway with a system prompt and user prompt.
/// Gateway is configured with automatic fallback to Workers AI (Llama 3.3 70B FP8).
/// Returns the assistant's response text.
pub async fn call_groq(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String, String> {
    let gateway = AiGateway::background(env)?;
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
    gateway.attach_headers(&headers)?;

    let url = gateway.provider_url("groq", "openai/v1/chat/completions")?;

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

    gateway.log_response_metadata(&response, "groq");

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Groq response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Groq returned status {}: {}",
            status, response_text
        ));
    }

    let groq_response: GroqResponse = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Failed to parse Groq response: {:?} - body: {}",
            e, response_text
        )
    })?;

    groq_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| "No choices in Groq response".to_string())
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: compiles with no new errors

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: route Groq calls through AI Gateway (background) with fallback"
```

---

### Task 5: Migrate Workers AI from Binding to HTTP Gateway

**Files:**
- Modify: `apps/api/src/services/llm.rs` (`call_workers_ai` function, line ~385)

This is the biggest change. The current `call_workers_ai` uses the `env.ai("AI")` binding which calls Workers AI directly (no gateway). We replace this with an HTTP call through the background gateway so Workers AI gets the same observability as Groq and Anthropic. The function gains a `model: &str` parameter.

- [ ] **Step 1: Replace `call_workers_ai` with HTTP-based gateway version**

Replace the entire `call_workers_ai` function:

```rust
/// Call Cloudflare Workers AI via AI Gateway.
/// Routes through the background gateway for unified observability.
/// Caller specifies the model (use WORKERS_AI_CHEAP_MODEL for background tasks).
pub async fn call_workers_ai(
    env: &Env,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> Result<String, String> {
    let gateway = AiGateway::background(env)?;

    // Workers AI through the gateway uses OpenAI-compatible format
    let request_body = GroqRequest {
        model: model.to_string(),
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
        .map_err(|e| format!("Failed to serialize Workers AI request: {:?}", e))?;

    let headers = Headers::new();
    headers
        .set("Content-Type", "application/json")
        .map_err(|e| format!("Failed to set content-type: {:?}", e))?;
    gateway.attach_headers(&headers)?;

    // Workers AI through the gateway: /workers-ai/v1/chat/completions
    let url = gateway.provider_url("workers-ai", "v1/chat/completions")?;

    let mut init = RequestInit::new();
    init.with_method(Method::Post);
    init.with_headers(headers);
    init.with_body(Some(body_json.into()));

    let request = Request::new_with_init(url.as_str(), &init)
        .map_err(|e| format!("Failed to create Workers AI request: {:?}", e))?;

    let mut response = Fetch::Request(request)
        .send()
        .await
        .map_err(|e| format!("Workers AI request failed: {:?}", e))?;

    gateway.log_response_metadata(&response, "workers-ai");

    let status = response.status_code();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Workers AI response: {:?}", e))?;

    if status != 200 {
        return Err(format!(
            "Workers AI returned status {}: {}",
            status, response_text
        ));
    }

    // Workers AI through the gateway returns OpenAI-compatible format:
    // {"choices": [{"message": {"content": "..."}}]}
    let parsed: GroqResponse = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Failed to parse Workers AI response: {:?} - body: {}",
            e, response_text
        )
    })?;

    parsed
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| "No choices in Workers AI response".to_string())
}
```

Note: We reuse `GroqRequest` and `GroqResponse` types since Workers AI through the gateway uses the same OpenAI-compatible format. The old `WorkersAiRequest`, `WorkersAiMessage`, and `env.ai()` code can be removed.

- [ ] **Step 2: Remove the old Workers AI request types**

Delete the now-unused `WorkersAiRequest` and `WorkersAiMessage` structs (previously at lines ~369-380).

- [ ] **Step 3: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -10`
Expected: compilation errors in `chat.rs`, `goals.rs`, `memory.rs` because `call_workers_ai` signature changed (added `model` param). This is expected -- we fix these in Task 6.

- [ ] **Step 4: Commit (will not compile yet -- call sites updated in Task 6)**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: migrate Workers AI from binding to HTTP gateway, add model param"
```

---

### Task 6: Update All Workers AI Call Sites

**Files:**
- Modify: `apps/api/src/services/chat.rs:675`
- Modify: `apps/api/src/services/goals.rs:178`
- Modify: `apps/api/src/services/memory.rs:422,1033,1518`

Every call to `call_workers_ai` needs the new `model` parameter inserted as the second argument. All background tasks use `WORKERS_AI_CHEAP_MODEL` (Qwen3-30b).

- [ ] **Step 1: Update `chat.rs:675` (title generation)**

Change:
```rust
match llm::call_workers_ai(env, "Generate a short title.", &prompt, 0.3, 30).await {
```
To:
```rust
match llm::call_workers_ai(env, llm::WORKERS_AI_CHEAP_MODEL, "Generate a short title.", &prompt, 0.3, 30).await {
```

- [ ] **Step 2: Update `goals.rs:178` (goal extraction)**

Change:
```rust
let response = crate::services::llm::call_workers_ai(
    env,
    "You extract structured data from pianist messages. Return only valid JSON.",
    &prompt,
    0.1,
    500,
)
```
To:
```rust
let response = crate::services::llm::call_workers_ai(
    env,
    crate::services::llm::WORKERS_AI_CHEAP_MODEL,
    "You extract structured data from pianist messages. Return only valid JSON.",
    &prompt,
    0.1,
    500,
)
```

- [ ] **Step 3: Update `memory.rs:422` (chat extraction)**

Change:
```rust
let output = llm::call_workers_ai(
    env,
    prompts::CHAT_EXTRACTION_SYSTEM,
    &user_prompt,
    0.0,
    500,
)
```
To:
```rust
let output = llm::call_workers_ai(
    env,
    llm::WORKERS_AI_CHEAP_MODEL,
    prompts::CHAT_EXTRACTION_SYSTEM,
    &user_prompt,
    0.0,
    500,
)
```

- [ ] **Step 4: Update `memory.rs:1033` (extract-chat endpoint)**

Change:
```rust
let output = match crate::services::llm::call_workers_ai(
    env,
    crate::services::prompts::CHAT_EXTRACTION_SYSTEM,
    &user_prompt,
    0.0,
    500,
)
```
To:
```rust
let output = match crate::services::llm::call_workers_ai(
    env,
    crate::services::llm::WORKERS_AI_CHEAP_MODEL,
    crate::services::prompts::CHAT_EXTRACTION_SYSTEM,
    &user_prompt,
    0.0,
    500,
)
```

- [ ] **Step 5: Update `memory.rs:1518` (entity extraction)**

Change:
```rust
let output = match crate::services::llm::call_workers_ai(
    env,
    "You extract entities from text. Return only a JSON array of strings.",
    &prompt,
    0.0,
    100,
)
```
To:
```rust
let output = match crate::services::llm::call_workers_ai(
    env,
    crate::services::llm::WORKERS_AI_CHEAP_MODEL,
    "You extract entities from text. Return only a JSON array of strings.",
    &prompt,
    0.0,
    100,
)
```

- [ ] **Step 6: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: clean compilation (all call sites now match the new signature)

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/services/chat.rs apps/api/src/services/goals.rs apps/api/src/services/memory.rs
git commit -m "feat: update Workers AI call sites to use Qwen3-30b via gateway"
```

---

### Task 7: Add Shadow Benchmarking

**Files:**
- Modify: `apps/api/src/services/llm.rs` (add shadow benchmark helper, modify `call_groq`)

- [ ] **Step 1: Add shadow benchmark helper function**

Add after the `AiGateway` impl block:

```rust
/// Check if shadow benchmarking should fire for this request.
/// Returns true with probability = SHADOW_BENCHMARK_PCT / 100.
fn should_shadow_benchmark(env: &Env) -> bool {
    let enabled = env
        .var("SHADOW_BENCHMARK_ENABLED")
        .ok()
        .map(|v| v.to_string() == "true")
        .unwrap_or(false);
    if !enabled {
        return false;
    }
    let pct: u32 = env
        .var("SHADOW_BENCHMARK_PCT")
        .ok()
        .and_then(|v| v.to_string().parse().ok())
        .unwrap_or(10);
    // Use getrandom for a random byte (0-255), scale to 0-99
    let mut buf = [0u8; 1];
    if getrandom::getrandom(&mut buf).is_err() {
        return false;
    }
    let roll = (buf[0] as u32 * 100) / 256;
    roll < pct
}

/// Fire a shadow request to Workers AI and log the latency.
/// Result is discarded -- only timing matters.
async fn shadow_benchmark_workers_ai(
    env: &Env,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f64,
    max_tokens: u32,
) -> f64 {
    let start = js_sys::Date::now();
    let _ = call_workers_ai(
        env,
        WORKERS_AI_GROQ_FALLBACK_MODEL,
        system_prompt,
        user_prompt,
        temperature,
        max_tokens,
    )
    .await;
    let elapsed = js_sys::Date::now() - start;
    console_log!("shadow_benchmark: workers_ai_latency_ms={:.0}", elapsed);
    elapsed
}
```

- [ ] **Step 2: Modify `call_groq` to run shadow benchmark concurrently**

Update the `call_groq` function body. After building the `body_json` string but before creating the fetch request, capture the system/user prompts for the shadow. After the Groq fetch returns, run the shadow if needed.

Add at the very top of `call_groq`, after `let gateway = ...`:

```rust
    let run_shadow = should_shadow_benchmark(env);
```

Then, after the Groq fetch completes and the response is fully processed (just before the final `groq_response.choices...` return), add:

```rust
    // Fire shadow benchmark if enabled (after Groq result is captured)
    if run_shadow {
        let shadow_start = js_sys::Date::now();
        let _ = call_workers_ai(
            env,
            WORKERS_AI_GROQ_FALLBACK_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
        )
        .await;
        let shadow_elapsed = js_sys::Date::now() - shadow_start;
        console_log!("shadow_benchmark: workers_ai_latency_ms={:.0}", shadow_elapsed);
    }
```

Note: In WASM single-threaded async, `futures::join!` would run both concurrently on the event loop. However, running the shadow *after* the Groq response ensures the shadow doesn't delay the student-facing response. The shadow latency is still a valid standalone measurement of Workers AI speed.

- [ ] **Step 3: Add `getrandom` import**

At the top of `llm.rs`, add:

```rust
use js_sys;
```

(`js_sys` is already in `Cargo.toml` deps, and `getrandom` is called via its function directly -- no `use` needed as it's a free function.)

- [ ] **Step 4: Verify it compiles**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: clean compilation

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/llm.rs
git commit -m "feat: add shadow benchmarking for Groq vs Workers AI latency comparison"
```

---

### Task 8: Final Verification and Cleanup

**Files:**
- Review: `apps/api/src/services/llm.rs` (full file)

- [ ] **Step 1: Full compile check**

Run: `cd apps/api && cargo check --target wasm32-unknown-unknown 2>&1`
Expected: clean compilation, no warnings related to new code

- [ ] **Step 2: Build release**

Run: `cd apps/api && cargo install -q worker-build && worker-build --release 2>&1 | tail -5`
Expected: successful WASM build

- [ ] **Step 3: Verify the old Workers AI types are removed**

Search for any remaining references to `WorkersAiRequest` or `WorkersAiMessage`:

Run: `grep -rn "WorkersAiRequest\|WorkersAiMessage" apps/api/src/`
Expected: no matches

- [ ] **Step 4: Verify all LLM calls route through gateway**

Search for any remaining direct provider URLs:

Run: `grep -n "api.groq.com\|api.anthropic.com" apps/api/src/services/llm.rs`
Expected: no matches (all replaced with gateway URLs)

- [ ] **Step 5: Verify no remaining `env.ai("AI")` calls**

Run: `grep -rn 'env.ai\|\.ai("AI")' apps/api/src/`
Expected: no matches in `llm.rs` (the `[ai]` binding in `wrangler.toml` remains for backward compatibility but is no longer called)

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "chore: cleanup old Workers AI types and verify gateway migration"
```

---

## Deferred: Gateway-Down Fallback

The spec mentions retrying against the native provider URL if the gateway itself is unreachable. This is not implemented because the gateway and Worker run on the same CF infrastructure -- if `gateway.ai.cloudflare.com` is down, the Worker is almost certainly down too. If real-world experience proves otherwise, add a direct-to-provider retry as a future task.

---

## Post-Implementation: Gateway Setup (Manual)

These steps happen in the Cloudflare dashboard, not in code:

1. **Create `crescendai-teacher` gateway** in the AI Gateway section of the CF dashboard
   - Rate limit: 100 req/min (sliding window)
   - Caching: enabled
   - Logging: enabled

2. **Create `crescendai-background` gateway**
   - Rate limit: 100 req/min (sliding window)
   - Caching: enabled
   - Logging: enabled
   - Fallback: configure Groq -> Workers AI fallback (if supported at gateway level; otherwise the gateway-level fallback is automatic when the provider returns 5xx)

3. **Set `CF_ACCOUNT_ID`** via `wrangler secret put CF_ACCOUNT_ID` or in the CF dashboard

4. **Deploy** via `just deploy-api`

5. **Verify** by triggering a chat message and checking AI Gateway logs in the dashboard

6. **Enable shadow benchmarking** by setting `SHADOW_BENCHMARK_ENABLED = "true"` in wrangler.toml and redeploying
