# System Architecture

## Piano Education Platform - Backend MVP

**Version:** 3.1
**Last Updated:** January 2025

---

## Development Phases

**Phase 1 (This Architecture)**: RAG system for PDF-based piano pedagogy

**Phase 2 (Future)**: Audio performance analysis using 16-dimensional AST model (code preserved in `model/` folder)

This architecture document describes Phase 1 only.

---

## Architecture Principles

1. **Performance + Accuracy**: Sub-200ms latency with 99% vector search recall
2. **Hybrid Edge/Cloud**: Cloudflare edge for caching/embeddings, Supabase for accuracy, GCP for compute
3. **Simplicity for MVP**: Single-region deployment, managed services, minimal ops overhead
4. **Observable by Default**: Metrics, logs, traces built into every layer
5. **Horizontal Scalability**: All services designed to scale out when needed

---

## Technology Stack

### Edge Layer (Cloudflare)

- **Storage**: R2 (PDFs, zero egress fees)
- **Cache**: Workers KV (embeddings, search results, LLM responses, <5ms P99 latency)
- **Compute**: Workers AI (BGE-base-v1.5 embeddings, cross-encoder re-ranking, 2x faster than self-hosted)
- **CDN**: Global edge network (300+ locations)
- **Rate Limiting**: WAF rules (100 req/min per IP)

### Database Layer (Supabase)

- **Primary Database**: PostgreSQL 16+ with pgvector 0.7+
- **Vector Indexing**: HNSW (99% recall, <8ms search)
- **Authentication**: Supabase Auth (JWT RS256, automatic refresh)
- **Hosting**: Managed PostgreSQL with daily backups
- **Connection Pooling**: Built-in PgBouncer

### Compute Layer (GCP)

- **API Servers**: Compute Engine (Rust + Axum framework)
- **Instance Type**: n2-standard-4 (4 vCPU, 16GB RAM)
- **Scaling**: Manual for MVP, auto-scaling for Phase 2
- **LLM Integration**: Claude 4.5 Haiku via Vertex AI (same region, low latency)
- **Background Workers**: Cloud Run Jobs (PDF processing, embedding batch jobs)
- **Future**: GPU instances (A100/H100) for AST model

### Observability Stack

- **Metrics**: Prometheus (self-hosted on GCP or Cloud Monitoring)
- **Logging**: Structured JSON → Cloud Logging
- **Tracing**: OpenTelemetry → Cloud Trace
- **Dashboards**: Grafana (hosted on GCP or Grafana Cloud)
- **Alerting**: Cloud Monitoring + PagerDuty

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Cloudflare Global Edge                      │
│  ┌────────────┬───────────────┬──────────────┐           │
│  │ R2 Storage │  Workers KV   │  Workers AI  │           │
│  │ (PDFs)     │  (Cache)      │ (Embeddings) │           │
│  └────────────┴───────────────┴──────────────┘           │
│  - WAF, DDoS protection, rate limiting                   │
│  - TLS termination, CDN (300+ edge locations)            │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              GCP Compute (Regional)                      │
│  ┌──────────────────────────────────────┐               │
│  │   Rust API Servers (Axum)            │               │
│  │   - n2-standard-4 (4 vCPU, 16GB)     │               │
│  │   - 2-3 instances (manual scaling)   │               │
│  │   - Stateless, horizontal scaling    │               │
│  └──────────────────────────────────────┘               │
│                    ↓                 ↓                  │
│         ┌─────────────────┐   ┌─────────────┐           │
│         │  Vertex AI      │   │  Cloud Run  │           │
│         │  (Claude 4.5)   │   │  Jobs       │           │
│         └─────────────────┘   └─────────────┘           │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                  Supabase (Regional)                     │
│  ┌───────────────────────────────────────┐              │
│  │  PostgreSQL 16 + pgvector             │              │
│  │  - 4XL Compute Addon (4 vCPU, 16GB)   │              │
│  │  - HNSW index (99% recall, <8ms)      │              │
│  │  - Automatic backups, HA failover     │              │
│  └───────────────────────────────────────┘              │
│  ┌───────────────────────────────────────┐              │
│  │  Supabase Auth (JWT management)       │              │
│  └───────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Cloudflare Edge Layer

#### R2 Object Storage

**Bucket Structure:**

```
piano-pdfs/
  ├── projects/{project_id}/{filename}.pdf
  └── temp-uploads/{upload_id}/     (24hr expiry)

piano-knowledge/
  ├── base/{doc_id}/                (platform content)
  └── teachers/{teacher_id}/{doc_id}/
```

**Features:**

- Zero egress fees (massive cost savings)
- Global CDN distribution (300+ edge locations)
- Sub-100ms TTFB worldwide
- Presigned URLs for secure uploads (10-min expiry)
- Multipart upload support for files >100MB

**Upload Flow:**

1. Client → API: Request upload
2. API → R2: Generate presigned URL
3. API → Client: Return presigned URL
4. Client → R2: Direct upload (bypasses API)
5. Client → API: Confirm upload
6. API → Cloud Run Job: Trigger processing

#### Workers KV (Cache)

**3-Layer Caching Strategy:**

**Layer 1: Embedding Cache**

- Key format: `embed:v1:{sha256(query_text)}`
- Value: Binary-serialized float32 array (768 dimensions, ~3KB)
- TTL: 24 hours
- Max keys: 100K
- Hit rate target: >70%
- Performance: <5ms P99 read latency

**Layer 2: Search Result Cache**

- Key format: `search:v1:{sha256(query + filters)}`
- Value: JSON array of top-3 chunks (id, content, score, metadata)
- TTL: 1 hour
- Max keys: 50K
- Hit rate target: >60%
- Invalidation: On teacher content upload

**Layer 3: LLM Response Cache**

- Key format: `llm:v1:{sha256(query + context)}`
- Value: JSON (answer, sources, confidence, timestamp)
- TTL: 24 hours
- Max keys: 10K
- Hit rate target: >40%
- Invalidation: Manual purge endpoint for admins

**Cost (100 users):**

- Storage: 50GB × $0.50/GB = $25
- Reads: 500K/mo (mostly cached) = included in Workers plan
- Total: ~$5/mo (under Workers paid plan)

#### Workers AI

**Embedding Generation:**

- Model: `@cf/baai/bge-base-en-v1.5` (768 dimensions)
- Performance: ~50ms per query (2x faster than self-hosted CPU)
- Batch API: For background processing (embeddings for uploaded docs)
- Cost: $0.011 per 1K queries

**Re-ranking:**

- Model: `@cf/baai/bge-reranker-base` (cross-encoder)
- Input: Top-20 chunks from hybrid search
- Output: Top-3 re-ranked by relevance
- Performance: ~20ms for 10 candidates
- Improves final accuracy to 99.5%+

**Deployment:**

- Cloudflare Workers (globally distributed)
- Auto-scaling (no config needed)
- Cold start: <10ms (minimal impact)

---

### 2. Supabase Database Layer

#### PostgreSQL Configuration

**Instance Specs:**

- Version: PostgreSQL 16.3
- Compute: 4XL addon (4 vCPU, 16GB RAM, 128GB disk)
- Connection limit: 200 concurrent connections
- Pooling: PgBouncer (transaction mode, built-in)
- Backup: Daily automated snapshots, 7-day retention
- High Availability: Automatic failover to standby replica

**pgvector Configuration:**

- Extension version: 0.7.0+
- Vector dimensions: 768 (BGE-base-v1.5)
- Distance metric: Cosine similarity (`<=>` operator)
- Index type: HNSW (Hierarchical Navigable Small World)

**HNSW Index Parameters:**

```sql
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (
  m = 16,                 -- Max connections per node (higher = better recall)
  ef_construction = 64    -- Build-time search depth (higher = better quality)
);

-- Query-time optimization
SET hnsw.ef_search = 40;  -- Runtime search depth (balance speed/recall)
```

**Memory Optimization:**

- **Critical**: HNSW index MUST fit in RAM for <10ms queries
- Calculation: 10K vectors × 768 dims × 4 bytes = ~30MB (easily fits in 16GB)
- At 1M vectors: ~3GB index size (still fits comfortably)
- Monitor: `pg_stat_user_indexes` for index cache hit rates (target: >99%)

**Performance Configuration:**

```sql
-- PostgreSQL tuning for pgvector
ALTER SYSTEM SET shared_buffers = '4GB';              -- 25% of RAM
ALTER SYSTEM SET effective_cache_size = '12GB';       -- 75% of RAM
ALTER SYSTEM SET maintenance_work_mem = '2GB';        -- For index builds
ALTER SYSTEM SET work_mem = '128MB';                  -- Per query operation
ALTER SYSTEM SET max_parallel_workers_per_gather = 4; -- Parallel query execution
```

#### Schema Design

```sql
-- Core tables (from PRD, optimized)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('teacher', 'student', 'admin')),
    full_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Vector table (optimized for pgvector)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES knowledge_base_docs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),  -- BGE-base-v1.5
    metadata JSONB,          -- {page, start_char, end_char, teacher_id}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index (99% recall, <8ms search)
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- BM25 full-text search index
CREATE INDEX idx_chunks_content_fts ON document_chunks
USING gin(to_tsvector('english', content));

-- Composite index for teacher-specific filtering
CREATE INDEX idx_chunks_teacher_metadata ON document_chunks
USING gin((metadata -> 'teacher_id'));

CREATE INDEX idx_chunks_doc ON document_chunks(doc_id, chunk_index);
```

**Query Patterns:**

**Hybrid Search (Vector + BM25):**

```sql
-- Vector similarity search (Top-20, ~8ms)
WITH vector_results AS (
  SELECT id, content, metadata,
         1 - (embedding <=> $query_embedding) AS score,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $query_embedding) AS rank
  FROM document_chunks
  WHERE metadata->>'teacher_id' = $teacher_id OR metadata->>'is_public' = 'true'
  ORDER BY embedding <=> $query_embedding
  LIMIT 20
),
-- BM25 keyword search (Top-20, ~3ms)
bm25_results AS (
  SELECT id, content, metadata,
         ts_rank(to_tsvector('english', content), plainto_tsquery($query)) AS score,
         ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) AS rank
  FROM document_chunks
  WHERE to_tsvector('english', content) @@ plainto_tsquery($query)
    AND (metadata->>'teacher_id' = $teacher_id OR metadata->>'is_public' = 'true')
  ORDER BY score DESC
  LIMIT 20
),
-- Reciprocal Rank Fusion (merge, ~2ms)
fused AS (
  SELECT id, content, metadata,
         SUM(1.0 / (rank + 60)) AS rrf_score
  FROM (
    SELECT * FROM vector_results
    UNION ALL
    SELECT * FROM bm25_results
  ) combined
  GROUP BY id, content, metadata
  ORDER BY rrf_score DESC
  LIMIT 10
)
SELECT * FROM fused;
```

**Connection Pooling:**

- Supabase provides PgBouncer automatically
- Pool mode: Transaction (best for serverless, low connection count)
- Max client connections: 200
- Pool size per API instance: 10 connections
- Reserve pool: 5 (for admin queries, backups)

#### Supabase Auth

**Features:**

- JWT-based authentication (RS256 signing)
- Automatic token refresh
- Row-level security (RLS) policies
- OAuth providers: Google, GitHub (future)

**JWT Configuration:**

- Access token expiry: 3600s (1 hour)
- Refresh token expiry: 604800s (7 days)
- Token rotation: Automatic on refresh
- Claims: `{sub, role, email, exp, iat}`

**Row-Level Security (RLS) Policies:**

```sql
-- Enable RLS on all tables
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Students can only access public content + their teacher's content
CREATE POLICY student_access ON document_chunks
FOR SELECT
USING (
  auth.jwt() ->> 'role' = 'student'
  AND (
    metadata->>'is_public' = 'true'
    OR metadata->>'teacher_id' IN (
      SELECT teacher_id FROM teacher_student_relationships
      WHERE student_id = auth.uid()
    )
  )
);

-- Teachers can access public content + their own content
CREATE POLICY teacher_access ON document_chunks
FOR SELECT
USING (
  auth.jwt() ->> 'role' = 'teacher'
  AND (
    metadata->>'is_public' = 'true'
    OR metadata->>'teacher_id' = auth.uid()
  )
);
```

---

### 3. GCP Compute Layer

#### API Servers (Rust + Axum)

**Instance Configuration:**

- Type: n2-standard-4 (4 vCPU, 16GB RAM)
- Instances: 2-3 (manual scaling for MVP)
- OS: Ubuntu 24.04 LTS
- Networking: Private VPC with public load balancer

**Rust API Stack:**

```toml
# Cargo.toml dependencies
[dependencies]
axum = "0.7"               # Web framework
tokio = "1.35"             # Async runtime
serde = "1.0"              # Serialization
sqlx = "0.7"               # PostgreSQL client (async, type-safe)
redis = "0.24"             # For future local caching
tower = "0.4"              # Middleware (tracing, timeouts)
tower-http = "0.5"         # HTTP middleware (CORS, compression)
tracing = "0.1"            # Logging/tracing
tracing-subscriber = "0.3" # Log aggregation
jsonwebtoken = "9.2"       # JWT validation
reqwest = "0.11"           # HTTP client (for Claude API via Vertex AI)
```

**Performance Optimizations:**

- Async/await for non-blocking I/O
- Connection pooling to Supabase (10 connections per instance)
- Request timeouts: 5s (standard), 30s (RAG queries)
- Response compression: gzip/brotli (tower-http)
- Prepared statements: All SQL queries pre-compiled at startup
- Zero-copy deserialization: `serde_json` with `#[serde(borrow)]`

**Deployment:**

1. Build Rust binary (release mode with LTO)
2. Docker image (Alpine Linux, ~50MB)
3. Deploy to GCP Compute Engine (systemd service)
4. Health checks: `GET /api/health` every 30s
5. Graceful shutdown: 30s drain period (SIGTERM handling)

**Scaling Strategy (Future):**

- Managed Instance Groups (MIGs)
- Auto-scaling rules:
  - CPU >70% for 5 minutes → scale up
  - P95 latency >100ms for 5 minutes → scale up
  - Min instances: 2, Max: 10

#### Load Balancer

**Configuration:**

- Type: GCP HTTP(S) Load Balancer (global)
- TLS: Managed certificate (auto-renewal)
- Backend: Instance group with health checks
- Timeout: 60s (for streaming RAG responses)
- CDN: Disabled (Cloudflare already handles this)

**Health Check:**

- Path: `GET /api/health`
- Interval: 30s
- Timeout: 10s
- Healthy threshold: 2 consecutive successes
- Unhealthy threshold: 3 consecutive failures

#### Gemini 2.0 Flash (Vertex AI)

**Integration:**

- API: Vertex AI (same GCP region as compute)
- Model: `gemini-2.0-flash-exp` (experimental, fastest)
- Streaming: Server-Sent Events (SSE)
- TTFT: <350ms (time to first token)
- Token rate: ~280 tokens/sec

**Configuration:**

```rust
// Gemini API client (Rust)
use reqwest::Client;

struct GeminiClient {
    client: Client,
    api_key: String,
    project_id: String,
}

async fn query_rag(
    &self,
    context: &str,
    query: &str
) -> Result<Stream<String>> {
    let prompt = format!(
        "Context: {}\n\nQuestion: {}\n\nProvide a concise answer with source citations.",
        context, query
    );

    let response = self.client
        .post(format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/anthropic/models/claude-haiku-4-5:streamGenerateContent",
            self.region, self.project_id, self.region
        ))
        .bearer_auth(&self.api_key)
        .json(&json!({
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.2,      // Low temp for factual accuracy
            "stream": true
        }))
        .send()
        .await?;

    Ok(response.bytes_stream())
}
```

**Cost Optimization:**

- Cache LLM responses in Workers KV (40% hit rate = 40% cost savings)
- Use Gemini Flash (cheapest tier, still high quality)
- Set `max_output_tokens`: 1024 (prevent runaway costs)
- Monitor token usage via Cloud Monitoring

#### Background Workers (Cloud Run Jobs)

**PDF Processing Worker:**

- Trigger: Pub/Sub message on R2 upload
- Task: Extract text, chunk, generate embeddings
- Concurrency: 5 workers
- Timeout: 10 minutes
- Retry: 3 attempts with exponential backoff

**Embedding Batch Worker:**

- Schedule: Nightly (for base knowledge base updates)
- Task: Generate embeddings for new content
- Batch size: 32 chunks per API call (Workers AI batch API)
- Performance: ~32 embeddings/sec

---

## RAG Pipeline Architecture

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER QUERY                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CLOUDFLARE EDGE (nearest edge location)                 │
│    - Cloudflare Worker receives request                     │
│    - Check Workers KV for cached embedding (70% hit rate)   │
│      ✓ HIT: Return cached embedding (<5ms)                  │
│      ✗ MISS: Call Workers AI BGE-base (50ms)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. GCP COMPUTE ENGINE (Rust API)                           │
│    - Receive query + embedding from edge                    │
│    - Execute hybrid search on Supabase                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SUPABASE (PostgreSQL + pgvector)                        │
│    - Vector search: HNSW index (Top-20, 99% recall, 8ms)   │
│    - BM25 search: GIN index (Top-20, 3ms)                   │
│    - Reciprocal Rank Fusion: Merge results (2ms)            │
│    - Return Top-10 chunks to API                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RE-RANKING (Cloudflare Workers AI)                      │
│    - API calls Workers AI cross-encoder                     │
│    - Input: Top-10 chunks                                   │
│    - Output: Top-3 most relevant (20ms)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. RESPONSE CACHING CHECK (Workers KV)                     │
│    - Check if LLM response cached (40% hit rate)            │
│      ✓ HIT: Stream cached response (<5ms)                   │
│      ✗ MISS: Proceed to LLM                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. LLM SYNTHESIS (Claude 4.5 Haiku via Vertex AI)         │
│    - Assemble context from Top-3 chunks                     │
│    - Stream prompt to Claude 4.5 Haiku                      │
│    - TTFT: <100ms, Token rate: Very fast                    │
│    - Stream response back through API                       │
│    - Cache response in Workers KV (24hr TTL)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. RESPONSE TO USER                                         │
│    - Stream response via SSE (Server-Sent Events)           │
│    - Include source citations + confidence score            │
└─────────────────────────────────────────────────────────────┘
```

### Performance Breakdown

| Stage | Latency (Cold) | Latency (Cached) | Notes |
|-------|----------------|------------------|-------|
| Edge routing | 5ms | 5ms | Cloudflare Worker |
| Embedding generation | 50ms | 5ms | 70% cache hit rate |
| Vector search (HNSW) | 8ms | - | 99% recall |
| BM25 search | 3ms | - | Keyword matching |
| RRF merge | 2ms | - | In-memory |
| Re-ranking | 20ms | - | Cross-encoder |
| LLM response cache | - | 5ms | 40% hit rate |
| Claude TTFT | 100ms | - | First token (Haiku) |
| **Total (Cold)** | **188ms** | - | Well under 200ms target! |
| **Total (Embed Cached)** | **143ms** | - | 70% of queries |
| **Total (Fully Cached)** | **10ms** | **10ms** | 40% of queries (LLM cache hit) |
| **Weighted Average** | **~130ms** | - | Accounting for cache hits |

---

## Deployment Architecture

### Region Selection

- **Primary Region**: `us-west2` (Los Angeles) or `us-central1` (Iowa)
- Rationale: Low latency to US users, good GPU availability for future AST model
- Supabase: Closest region to GCP deployment
- Cloudflare: Automatically routes to nearest edge (global)

### Networking

```
Internet
    ↓
Cloudflare Edge (Global Anycast)
    ├─ TLS termination
    ├─ DDoS protection
    ├─ WAF rules (rate limiting)
    └─ CDN (R2 assets)
    ↓
GCP HTTP(S) Load Balancer (Regional)
    ├─ Health checks
    ├─ SSL certificate (backup)
    └─ Backend: Compute Engine MIG
        ↓
    GCP VPC (Private)
        ├─ API Servers (private IPs, NAT for outbound)
        ├─ Cloud Run Jobs (private, VPC connector)
        └─ Vertex AI (private endpoint)
    ↓
Supabase (External, TLS-encrypted)
    ├─ PostgreSQL (pooled connection)
    └─ Auth API (JWT validation)
```

**Firewall Rules:**

- Allow: HTTPS (443) from Cloudflare IPs → Load Balancer
- Allow: Load Balancer → Compute instances (health checks)
- Allow: Compute instances → Supabase (PostgreSQL port 5432, HTTPS 443)
- Allow: Compute instances → Vertex AI (HTTPS 443)
- Deny: All other traffic

---

## Cost Analysis

### MVP (100 users, 50K RAG queries/month)

| Service | Component | Specs | Cost |
|---------|-----------|-------|------|
| **Cloudflare** | | | |
| | R2 Storage | 2.5TB storage, 200K reads | $38 |
| | Workers (Paid Plan) | 10M requests | $5 |
| | Workers AI | 50K embeddings + 50K rerank | $20 |
| | Workers KV | 50GB storage, 500K reads | $5 |
| **Supabase** | | | |
| | Pro Plan | PostgreSQL + Auth + Backups | $25 |
| | Compute Addon (4XL) | 4 vCPU, 16GB RAM, 128GB disk | $100 |
| **GCP** | | | |
| | Compute Engine | 2x n2-standard-4 (24/7) | $240 |
| | Claude 4.5 Haiku | 50K queries × ~1K tokens | $25 |
| | Cloud Logging | 50GB/month | $10 |
| | Cloud Monitoring | Metrics + alerting | $10 |
| | Load Balancer | Minimal traffic | $20 |
| **Total** | | | **$523/month** |

### At Scale (5K users, 2M queries/month)

| Service | Component | Cost |
|---------|-----------|------|
| Cloudflare | R2 + Workers + KV + AI | $180 |
| Supabase | Pro + 8XL compute | $250 |
| GCP | Compute (4x n2-standard-4) + Claude 4.5 Haiku + Monitoring | $550 |
| **Total** | | **$1,030/month** |

**Cost per Query (at scale):**

- $1,030 / 2M queries = **$0.0005/query** (0.05 cents)

---

## Performance Testing Strategy

### Test Scenarios

**1. RAG Query Load Test (k6)**

```javascript
// rag-load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 50 },  // Ramp to 50 users
    { duration: '5m', target: 50 },  // Hold 50 users
    { duration: '2m', target: 100 }, // Ramp to 100 users
    { duration: '3m', target: 100 }, // Hold 100 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    'http_req_duration{type:rag}': ['p(95)<200', 'p(99)<500'],
    'http_req_failed': ['rate<0.01'],  // <1% error rate
  },
};

export default function () {
  const queries = [
    'How do I improve finger independence?',
    'What is the correct hand position for scales?',
    'Explain legato vs staccato articulation',
  ];

  const query = queries[Math.floor(Math.random() * queries.length)];

  const res = http.post(
    'https://api.piano-platform.com/api/chat/query',
    JSON.stringify({ query }),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${__ENV.AUTH_TOKEN}`
      },
      tags: { type: 'rag' },
    }
  );

  check(res, {
    'status 200': (r) => r.status === 200,
    'has answer': (r) => JSON.parse(r.body).answer !== undefined,
    'has sources': (r) => JSON.parse(r.body).sources.length > 0,
  });
}
```

Run: `k6 run --vus 100 --duration 10m rag-load-test.js`

**2. Vector Search Recall Test**

- Pre-load 10K test chunks with known relevance labels
- Query 1K test questions
- Measure recall@10, recall@20, precision@3
- Target: >99% recall@10

**3. Cache Hit Rate Validation**

- Pre-warm cache with 1K common queries
- Execute 5K queries (70% repeats, 30% new)
- Measure hit rates per cache layer
- Target: Embedding cache >70%, LLM cache >40%

### Monitoring Dashboards

**RAG Performance Dashboard (Grafana):**

- Query latency (P50, P95, P99) - line graph
- Latency breakdown (embedding, search, rerank, LLM) - stacked area
- Cache hit rates (embedding, search, LLM) - gauge
- Vector search recall (rolling average) - line graph
- Queries per second - counter
- Error rate - gauge (alert if >0.1%)

**Database Dashboard:**

- Connection pool utilization - gauge
- Query latency by type (SELECT, INSERT, UPDATE) - histogram
- Slow queries (>100ms) - table
- Index hit rate (target: >99%) - gauge
- HNSW index memory usage - gauge

**Edge Performance Dashboard:**

- Workers KV latency (P95, P99) - line graph
- Workers AI latency (embedding, rerank) - histogram
- R2 request rate + bandwidth - area chart
- Cache hit rates by layer - stacked bar

---

## Security Architecture

### Authentication Flow

```
User → Frontend → Cloudflare Worker
    ↓
Supabase Auth (login/register)
    ↓
Return JWT (access + refresh tokens)
    ↓
Frontend stores tokens
    ↓
Subsequent requests → Cloudflare Worker
    ↓
Worker validates JWT signature (cached public key)
    ↓
Forward to GCP API with user context
    ↓
API enforces authorization (relationship-based access)
```

### Data Encryption

**In Transit:**

- TLS 1.3 everywhere (Cloudflare → GCP → Supabase)
- Certificate pinning for Supabase API calls

**At Rest:**

- R2: AES-256 encryption (Cloudflare-managed keys)
- Supabase: AES-256 encryption (Supabase-managed keys)
- GCP: Disk encryption enabled by default

### Secrets Management

**GCP Secret Manager:**

- Supabase API keys
- Supabase JWT secret (for validation)
- Claude API key (Vertex AI)
- R2 access keys
- Workers AI API token

**Access Control:**

- API servers: Service account with least privilege
- Secrets rotated every 90 days (automated)

---

## Disaster Recovery

### Backup Strategy

- **Supabase**: Daily automated backups, 7-day retention, point-in-time recovery (last 7 days)
- **R2**: 11-nines durability, no manual backups needed
- **Database snapshots**: Weekly full backup to GCS (90-day retention)

### Recovery Procedures

- **Database failure**: Supabase automatic failover (<2 min RTO)
- **API server failure**: Load balancer auto-routes to healthy instances
- **Region failure**: Manual restore from Supabase backup (4-hour RTO acceptable for MVP)

### Testing

- Quarterly DR drill (restore Supabase backup to staging environment)
- Monthly failover test (kill one API instance, verify load balancer routing)

---

## Migration Path (Post-MVP)

### Scaling to 5K Users

- Increase Supabase compute: 4XL → 8XL
- Add GCP instances: 2 → 4 (Managed Instance Group)
- Implement auto-scaling rules
- Add Supabase read replicas for analytics queries

### Scaling to 50K Users

- Multi-region deployment: US-West + US-East
- Supabase regional replicas (read-only)
- Cloudflare Vectorize for 80% of queries (edge), pgvector for 20% (accuracy)
- Self-hosted LLM (if needed) or upgrade to Claude Sonnet for more complex tasks

### Phase 2: Audio Spectrogram Transformer Integration

- Existing AST model preserved in `model/` folder (PyTorch Lightning, 16 dimensions)
- Provision GPU instances (4x A100 40GB in us-west2)
- Deploy Triton Inference Server or Modal GPU integration
- Re-enable audio upload endpoints (currently disabled in server code)
- Audio upload to R2, processing on GCP GPUs
- Integrate AST analysis results with RAG tutor for personalized feedback
- Architecture already optimized for GPU workloads via GCP

---

## Summary

This hybrid architecture delivers:

✅ **Performance**: <200ms P95 RAG queries (weighted avg ~250ms with caching)
✅ **Accuracy**: 99% vector search recall (pgvector HNSW)
✅ **Simplicity**: Managed services (Cloudflare, Supabase, GCP)
✅ **Cost-Effectiveness**: $523/month (100 users), $1,030/month (5K users)
✅ **Scalability**: Horizontal scaling to 50K+ users
✅ **Global Performance**: Cloudflare edge caching (300+ locations)
✅ **Future-Proof**: GCP for GPU support, clear migration path

**Key Trade-offs:**

- Hybrid complexity (3 platforms) vs pure single-cloud
- Slightly higher latency than pure edge (Vectorize) but 99% accuracy vs 80%
- Manual scaling for MVP vs auto-scaling (cost optimization)

**Next Steps:**

1. Provision Cloudflare account (R2, Workers, KV)
2. Set up Supabase project, enable pgvector, configure HNSW index
3. Deploy GCP Compute instances, configure networking
4. Implement Rust API with Axum framework
5. Integrate Cloudflare Workers AI for embeddings
6. Performance testing (k6 load tests, recall validation)
7. Validate <200ms P95 latency + 99% recall before frontend development
