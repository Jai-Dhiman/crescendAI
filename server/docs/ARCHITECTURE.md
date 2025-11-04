# CrescendAI Server - Technical Architecture

**Version:** 2.0
**Last Updated:** 2025-01-30
**Status:** Planning

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Integration Patterns](#integration-patterns)
7. [Database Schema](#database-schema)
8. [Caching Strategy](#caching-strategy)
9. [Security & Authentication](#security--authentication)
10. [Error Handling](#error-handling)
11. [Observability](#observability)
12. [Deployment](#deployment)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Web App (SvelteKit)                          │
│                      https://crescend.ai                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Cloudflare Workers (Rust)                          │
│                   wasm32-unknown-unknown                             │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  API Routes:                                                │    │
│  │  • POST /api/upload          → R2 storage                  │    │
│  │  • POST /api/chat            → SSE streaming               │    │
│  │  • POST /api/feedback/:id    → Structured response         │    │
│  │  • GET  /api/chat/sessions   → D1 query                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Core Modules:                                              │    │
│  │  • dedalus_client.rs   → HTTP client for Dedalus API       │    │
│  │  • rag_tools.rs        → Tool definitions for RAG          │    │
│  │  • knowledge_base.rs   → Vectorize + D1 search             │    │
│  │  • db/*.rs             → D1 query helpers                   │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────┬───────────────┬────────────────┬────────────────┬────┘
              │               │                │                │
              ▼               ▼                ▼                ▼
       ┌────────────┐  ┌──────────┐   ┌──────────────┐  ┌─────────┐
       │     D1     │  │ Vectorize│   │      R2      │  │   KV    │
       │  (SQLite)  │  │ (Vectors)│   │  (Objects)   │  │ (Cache) │
       └────────────┘  └──────────┘   └──────────────┘  └─────────┘
              │               │                │
              │               │                │
              │               └────────┬───────┘
              │                        │
              ▼                        ▼
       ┌──────────────────────────────────────────┐
       │      Cloudflare Workers AI               │
       │      (BGE-base, BGE-reranker)            │
       └──────────────────────────────────────────┘

              │
              │ HTTPS (External)
              ▼
       ┌─────────────────────┐          ┌──────────────────┐
       │   Dedalus API       │          │  Modal (Future)  │
       │   GPT-5-nano        │          │  AST Model       │
       │   Tool Orchestration│          │  16D Analysis    │
       └─────────────────────┘          └──────────────────┘
```

---

## Architecture Principles

### 1. Edge-First Computing

- **All compute runs on Cloudflare Workers** (V8 isolates, near-instant cold starts)
- **Data stored at the edge** (D1, KV, R2 all Cloudflare-native)
- **Global distribution** with automatic routing to nearest data center

### 2. Rust + WASM for Performance

- **Rust compiled to wasm32-unknown-unknown** for Workers runtime
- **Type safety** prevents runtime errors
- **Zero-cost abstractions** for high performance
- **Small bundle size** (<1MB WASM)

### 3. AI Orchestration via Dedalus

- **Single API** for multi-model routing (GPT-5-nano primary)
- **Tool calling** for RAG search and data retrieval
- **Streaming responses** for real-time UX
- **Retry logic** built into Dedalus SDK

### 4. Data Locality

- **D1 for structured data** (chat sessions, messages, metadata)
- **Vectorize for embeddings** (semantic search)
- **R2 for blobs** (audio files, PDFs)
- **KV for caching** (embeddings, LLM responses)

### 5. Separation of Concerns

- **Workers** = routing, auth, caching
- **Dedalus** = AI orchestration, tool calling
- **D1** = data persistence
- **External services** (Modal) = heavy compute

---

## Component Breakdown

### Cloudflare Workers (Rust)

**Responsibilities:**

- HTTP request routing
- Authentication & rate limiting
- Input validation
- Response caching
- SSE streaming
- Error handling

**Key Files:**

- `src/lib.rs` - Main router and CORS
- `src/handlers/upload.rs` - Audio upload to R2
- `src/handlers/chat.rs` - Streaming chat with SSE
- `src/handlers/feedback.rs` - Structured feedback generation
- `src/dedalus_client.rs` - HTTP client for Dedalus API
- `src/rag_tools.rs` - Tool definitions
- `src/knowledge_base.rs` - RAG search logic
- `src/db/*.rs` - D1 query helpers

---

### Dedalus API (External)

**Responsibilities:**

- LLM orchestration (GPT-5-nano)
- Tool calling (RAG search, model retrieval)
- Streaming token generation
- Multi-model routing
- Retry and error handling

**Integration:**

- REST API: `https://api.dedaluslabs.ai/v1/chat/completions`
- Authentication: Bearer token
- Streaming: SSE over HTTP

**Tool Definitions:**

```rust
pub struct DedalusTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// Example tools:
// - search_knowledge_base(query: String, top_k: u32)
// - get_performance_analysis(recording_id: String)
// - get_user_context(session_id: String)
```

---

### D1 Database (SQLite)

**Responsibilities:**

- Structured data storage
- Chat session persistence
- Message history
- Recording metadata
- User context
- Knowledge chunk metadata

**Schema Highlights:**

- **chat_sessions** - Session metadata
- **chat_messages** - Full message history
- **recordings** - Audio file metadata
- **analysis_results** - AST model outputs
- **knowledge_chunks** - RAG chunk text + metadata
- **knowledge_documents** - Document metadata

See [D1_SCHEMA.md](./D1_SCHEMA.md) for full schema.

---

### Vectorize (Vector Search)

**Responsibilities:**

- Store document embeddings
- Semantic similarity search
- Fast nearest-neighbor retrieval

**Configuration:**

- **Index:** `crescendai-piano-pedagogy`
- **Dimensions:** 768 (BGE-base-en-v1.5)
- **Metric:** Cosine similarity
- **Max results:** 100

**Workflow:**

1. Generate embedding via Workers AI
2. Query Vectorize for top-k similar vectors
3. Retrieve chunk metadata from D1
4. Re-rank using BGE reranker

---

### R2 Object Storage

**Responsibilities:**

- Audio file storage (WAV, MP3, M4A)
- PDF document storage
- Presigned URLs for direct upload

**Buckets:**

- **audio-recordings** - User-uploaded audio
- **knowledge-documents** - Pedagogy PDFs

**Key Format:**

- Audio: `recordings/{recording_id}.{ext}`
- PDFs: `documents/{doc_id}/{filename}.pdf`

---

### Workers KV (Cache)

**Responsibilities:**

- Embedding cache (70% hit rate target)
- LLM response cache (40% hit rate target)
- Metadata cache

**Keys:**

- `embedding:{text_hash}` → `[f32; 768]`
- `llm:{prompt_hash}` → Response JSON
- `metadata:{recording_id}` → Metadata JSON

**TTL:**

- Embeddings: 7 days
- LLM responses: 24 hours
- Metadata: 1 hour

---

### Workers AI

**Responsibilities:**

- Text embedding generation (BGE-base-en-v1.5)
- Re-ranking (BGE reranker)

**Models:**

- **Embeddings:** `@cf/baai/bge-base-en-v1.5` (768 dims)
- **Reranker:** `@cf/baai/bge-reranker-base` (0-1 score)

---

### Modal (Future - AST Model)

**Responsibilities:**

- Audio analysis (16-dimensional scores)
- Temporal segmentation (3-second windows)
- Performance scoring

**Interface:**

- **Endpoint:** `POST https://modal.example.com/analyze`
- **Input:** Audio file URL or base64
- **Output:** JSON with 16D scores + temporal segments

**Mock Implementation (Phase 1):**

```rust
pub async fn mock_ast_analysis(recording_id: &str) -> AnalysisResult {
    AnalysisResult {
        timing_stable_unstable: 0.45,
        articulation_short_long: 0.62,
        // ... 16D scores
        temporal_segments: vec![
            TemporalSegment { timestamp: "0:00-0:03", scores: [...] },
            // ...
        ]
    }
}
```

---

## Data Flow

### Flow 1: Recording Upload

```
User                Workers               R2                  D1
 │                     │                   │                   │
 │─POST /api/upload───>│                   │                   │
 │  (multipart form)   │                   │                   │
 │                     │──PUT audio───────>│                   │
 │                     │                   │                   │
 │                     │──INSERT metadata─────────────────────>│
 │                     │                                       │
 │<─{recording_id}────│                                       │
```

### Flow 2: Streaming Chat

```
User              Workers           Dedalus API         D1          Vectorize
 │                   │                   │              │               │
 │─POST /api/chat───>│                   │              │               │
 │  {message}        │                   │              │               │
 │                   │──INSERT message──────────────────>│              │
 │                   │                   │              │               │
 │                   │──POST /chat/completions          │               │
 │                   │   {messages, tools}              │               │
 │                   │                   │              │               │
 │                   │                   │──[tool: search_kb]           │
 │                   │                   │              │               │
 │                   │<──tool_call───────│              │               │
 │                   │                   │              │               │
 │                   │──embed(query)─────────────────────────────────> │
 │                   │                                  │               │
 │                   │──query vectors───────────────────────────────>  │
 │                   │<──vector IDs─────────────────────────────────── │
 │                   │                                  │               │
 │                   │──SELECT chunks──────────────────>│               │
 │                   │<──chunk text─────────────────────│               │
 │                   │                   │              │               │
 │                   │──tool_response───>│              │               │
 │                   │                   │              │               │
 │                   │<──SSE stream──────│              │               │
 │<─SSE: token──────│                   │              │               │
 │<─SSE: token──────│                   │              │               │
 │<─SSE: [DONE]─────│                   │              │               │
 │                   │                   │              │               │
 │                   │──INSERT assistant msg────────────>│              │
```

### Flow 3: Structured Feedback Generation

```
User          Workers        Dedalus      D1        Vectorize     Modal (Future)
 │               │              │          │            │              │
 │─POST /feedback/:id          │          │            │              │
 │               │              │          │            │              │
 │               │──SELECT analysis─────> │            │              │
 │               │<──16D scores───────────│            │              │
 │               │              │          │            │              │
 │               │──POST /chat──>         │            │              │
 │               │  {scores, tools}       │            │              │
 │               │              │          │            │              │
 │               │              │──[tool: search_kb]───────────────> │
 │               │              │<──chunks─────────────────────────── │
 │               │              │          │            │              │
 │               │<──response───│          │            │              │
 │               │              │          │            │              │
 │               │──INSERT feedback─────> │            │              │
 │<──JSON─────────│              │          │            │              │
```

---

## Technology Stack

### Runtime & Languages

- **Rust 1.70+** - Systems programming language
- **wasm32-unknown-unknown** - WASM target for Workers
- **worker-rs 0.6+** - Rust bindings for Cloudflare Workers

### Cloudflare Services

- **Workers** - Edge compute (V8 isolates)
- **D1** - Distributed SQLite database
- **Vectorize** - Vector search
- **R2** - Object storage
- **KV** - Key-value cache
- **Workers AI** - ML inference (embeddings)

### External APIs

- **Dedalus API** - AI orchestration, GPT-5-nano
- **Modal** (future) - AST model hosting

### Development Tools

- **wrangler** - Cloudflare CLI
- **cargo** - Rust build tool
- **sqlx** (future) - SQL query builder for D1

---

## Integration Patterns

### Pattern 1: Dedalus HTTP Client (Rust)

```rust
pub struct DedalusClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl DedalusClient {
    pub async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<DedalusTool>,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "model": "openai/gpt-5-nano",
                "messages": messages,
                "tools": tools,
                "stream": true
            }))
            .send()
            .await?;

        Ok(parse_sse_stream(response))
    }
}
```

### Pattern 2: RAG Tool Execution

```rust
pub async fn execute_search_kb_tool(
    env: &Env,
    query: &str,
    top_k: usize,
) -> Result<Vec<KnowledgeChunk>> {
    // 1. Check embedding cache
    let cache_key = format!("embedding:{}", hash(query));
    let embedding = match env.kv("CACHE_KV")?.get(&cache_key).await? {
        Some(cached) => serde_json::from_str(&cached)?,
        None => {
            // 2. Generate embedding
            let emb = embed_text(env, query).await?;
            // 3. Cache embedding
            env.kv("CACHE_KV")?
                .put(&cache_key, &serde_json::to_string(&emb)?)?
                .expiration_ttl(604800) // 7 days
                .execute().await?;
            emb
        }
    };

    // 4. Query Vectorize
    let vector_ids = query_vectorize(env, &embedding, top_k * 3).await?;

    // 5. Fetch chunks from D1
    let chunks = fetch_chunks_by_vector_ids(env, &vector_ids).await?;

    // 6. Re-rank using BGE reranker
    let reranked = rerank_chunks(env, query, chunks, top_k).await?;

    Ok(reranked)
}
```

### Pattern 3: SSE Streaming

```rust
pub async fn stream_chat_response(
    dedalus_stream: impl Stream<Item = Result<String>>,
) -> Response {
    let stream = async_stream::stream! {
        pin_mut!(dedalus_stream);

        while let Some(result) = dedalus_stream.next().await {
            match result {
                Ok(token) => {
                    yield Ok(format!("data: {}\n\n", token));
                }
                Err(e) => {
                    yield Ok(format!("data: {{\"error\": \"{}\"}}\n\n", e));
                    break;
                }
            }
        }

        yield Ok("data: [DONE]\n\n".to_string());
    };

    Response::from_stream(stream)
        .with_headers(vec![
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
        ])
}
```

### Pattern 4: D1 Query Helper

```rust
pub async fn insert_chat_message(
    env: &Env,
    session_id: &str,
    role: &str,
    content: &str,
) -> Result<String> {
    let db = env.d1("DB")?;
    let message_id = Uuid::new_v4().to_string();

    db.prepare("
        INSERT INTO chat_messages (id, session_id, role, content, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
    ")
    .bind(&[
        message_id.clone().into(),
        session_id.into(),
        role.into(),
        content.into(),
        (js_sys::Date::now() as i64).into(),
    ])?
    .run()
    .await?;

    Ok(message_id)
}
```

---

## Database Schema

See [D1_SCHEMA.md](./D1_SCHEMA.md) for complete schema definitions.

**Key Tables:**

- `chat_sessions` - Session metadata
- `chat_messages` - Message history
- `recordings` - Audio metadata
- `analysis_results` - AST scores
- `knowledge_documents` - Document metadata
- `knowledge_chunks` - Chunk text + Vectorize IDs
- `user_contexts` - User goals/constraints

---

## Caching Strategy

### Layer 1: Workers KV (Edge Cache)

**Embedding Cache:**

- **Key:** `embedding:{sha256(text)}`
- **Value:** `[f32; 768]` (JSON)
- **TTL:** 7 days
- **Expected hit rate:** 70%

**LLM Response Cache:**

- **Key:** `llm:{sha256(prompt+context)}`
- **Value:** Response JSON
- **TTL:** 24 hours
- **Expected hit rate:** 40%

### Layer 2: D1 Query Cache

- D1 automatically caches query results
- No explicit caching needed
- Invalidate on writes

### Layer 3: HTTP Cache (Client-side)

- ETag-based caching for GET endpoints
- `Cache-Control` headers for recording metadata
- Presigned R2 URLs cached client-side

---

## Security & Authentication

### Authentication (Phase 1: Development)

- **API Key in header:** `X-API-Key: secret`
- **Validate in middleware** before routing
- **Store keys in Workers secrets**

### Authentication (Phase 2: Production)

- **Supabase Auth** (JWT tokens)
- **RS256 signature verification**
- **User ID extraction from JWT**

### Rate Limiting

```rust
pub struct RateLimiter {
    kv: KvStore,
    requests_per_minute: u32,
}

impl RateLimiter {
    pub async fn check(&self, client_ip: &str) -> Result<()> {
        let key = format!("rate:{}", client_ip);
        let count: u32 = self.kv.get(&key)
            .await?
            .unwrap_or(0);

        if count >= self.requests_per_minute {
            return Err(Error::RateLimitExceeded);
        }

        self.kv.put(&key, &(count + 1).to_string())?
            .expiration_ttl(60)
            .execute()
            .await?;

        Ok(())
    }
}
```

### Input Validation

- **File size:** Max 50MB
- **File types:** WAV, MP3, M4A only (magic bytes check)
- **JSON schemas:** Validate all request bodies
- **UUID format:** Validate all IDs

### CORS

- **Allowed origins:** `https://crescend.ai`, `http://localhost:3000`
- **Credentials:** Allow
- **Methods:** GET, POST, PUT, DELETE, OPTIONS

---

## Error Handling

### Error Types

```rust
pub enum AppError {
    InvalidInput(String),
    Unauthorized,
    RateLimitExceeded,
    DedalusApiError(String),
    D1Error(String),
    VectorizeError(String),
    R2Error(String),
    InternalError(String),
}

impl AppError {
    pub fn to_response(&self, is_dev: bool) -> Response {
        let (status, message) = match self {
            AppError::InvalidInput(msg) => (400, msg.clone()),
            AppError::Unauthorized => (401, "Unauthorized".to_string()),
            AppError::RateLimitExceeded => (429, "Rate limit exceeded".to_string()),
            AppError::DedalusApiError(msg) if is_dev => (502, format!("Dedalus: {}", msg)),
            AppError::DedalusApiError(_) => (502, "AI service unavailable".to_string()),
            _ => (500, "Internal server error".to_string()),
        };

        Response::error(message, status).unwrap()
    }
}
```

### Retry Logic

```rust
pub async fn with_retry<F, T>(
    f: F,
    max_retries: u32,
) -> Result<T>
where
    F: Fn() -> Future<Output = Result<T>>,
{
    let mut attempts = 0;
    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempts < max_retries => {
                attempts += 1;
                let backoff = Duration::from_millis(100 * 2_u64.pow(attempts));
                sleep(backoff).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

## Observability

### Logging

```rust
macro_rules! log_request {
    ($method:expr, $path:expr, $duration_ms:expr, $status:expr) => {
        console_log!(
            "[{}] {} - {}ms - {}",
            $method,
            $path,
            $duration_ms,
            $status
        );
    };
}
```

### Metrics

Track via D1 analytics table:

- Request count by endpoint
- Latency (P50, P95, P99)
- Error rate by error type
- Cache hit/miss rates
- D1 query times
- Vectorize search times

### Alerting

- 5xx errors >1% for 5 minutes
- P95 latency >1s for 5 minutes
- Dedalus API errors >10% for 5 minutes
- D1 unavailability

---

## Deployment

### Development

```bash
wrangler dev
```

### Production

```bash
# Run migrations
wrangler d1 migrations apply DB --remote

# Deploy Workers
wrangler deploy

# Verify deployment
curl https://api.crescend.ai/api/health
```

### Environment Variables

```toml
# wrangler.toml
[vars]
DEDALUS_API_KEY = "xxx"
ENVIRONMENT = "production"
ALLOWED_ORIGINS = "https://crescend.ai"
```

### Secrets

```bash
wrangler secret put DEDALUS_API_KEY
wrangler secret put MODAL_API_KEY
```

---

## Future Enhancements

1. **Multi-tenancy** - User workspaces, organizations
2. **Real-time collaboration** - Durable Objects for shared sessions
3. **Advanced analytics** - D1 analytics, dashboards
4. **A/B testing** - Model comparison, feature flags
5. **Internationalization** - Multi-language support
6. **Mobile SDK** - Native iOS/Android clients
