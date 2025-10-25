# Product Requirements Document
## Piano Education Platform - Backend MVP
**Version:** 3.1
**Last Updated:** January 2025
**Performance Priority:** Critical - All decisions optimized for sub-200ms latency with 99% accuracy

---

## Development Phases

**Phase 1 (This PRD)**: RAG system for PDF-based piano pedagogy with teacher-student workflows.

**Phase 2 (Future)**: Audio performance analysis integration using 16-dimensional AST model (existing code in `model/` folder preserved).

This PRD focuses exclusively on Phase 1 RAG implementation.

---

## Overview

High-performance backend for piano education platform with NotebookLM-style RAG capabilities for intelligent Q&A about sheet music, piano techniques, and pedagogy. This MVP focuses on backend validation with performance testing before frontend development.

**Hybrid Architecture**: Cloudflare Edge (global caching, storage, embeddings) + Supabase (PostgreSQL + pgvector) + GCP Compute Engine (core API logic)

---

## Core Principles

- **Performance + Accuracy**: Sub-200ms RAG queries with 99% recall accuracy
- **Backend Validation**: Prove architecture can scale to 100+ users before building frontend
- **Teacher-Centric**: All workflows prioritize teacher needs and content
- **PDF-Focused**: Sheet music PDFs are the central artifact
- **Hybrid Edge/Cloud**: Leverage edge for speed, cloud for accuracy

---

## MVP Scope

### In Scope

**Core Backend Systems:**
1. High-performance RAG pipeline for piano pedagogy Q&A (99% recall)
2. PDF project management (upload, storage, metadata)
3. Annotation system (CRUD API for PDF annotations)
4. User authentication and teacher-student relationships
5. Knowledge base management (base content + teacher uploads)
6. Comprehensive performance testing and benchmarking

**Target Users:**
- 50-100 beta users (mix of teachers and students)
- Single region deployment (nearest to beta users)
- In-person lessons (no real-time video/audio streaming)

### Out of Scope for Phase 1

- Frontend/UI development (backend API validation only)
- Real-time collaborative editing (async annotations only)
- Video/audio streaming sessions
- Payment processing
- Mobile apps
- **Audio Spectrogram Transformer (Phase 2 - code preserved in `model/` folder)**
- Multi-region deployment
- Scheduling/calendar features

---

## Technical Requirements

### Performance Targets

**API Latency:**
- P50: <20ms
- P95: <50ms
- P99: <100ms

**RAG Query Performance:**
- Cold query (full pipeline): <200ms P95
- Cached query (edge): <50ms P95
- Streaming TTFT (Time-to-First-Token): <350ms

**Database Queries:**
- Simple queries (user lookup, project fetch): <5ms P95
- Vector similarity search (pgvector HNSW): <8ms P95
- Complex joins (projects with access): <15ms P95

**PDF Operations:**
- Upload to R2 (10MB file): <3s
- Metadata extraction: <500ms
- Presigned URL generation: <10ms

**Annotation Operations:**
- Create/Update/Delete: <10ms P95
- Fetch page annotations: <15ms P95

### Accuracy Requirements

**Vector Search Recall:**
- Target: 99% recall@10 (critical for teacher-uploaded content)
- Method: pgvector with HNSW indexing
- Validation: Re-ranking with cross-encoder for top-3 results

**RAG Answer Quality:**
- Source citations for all answers
- Confidence scores for LLM responses
- Fallback to "I don't know" when confidence <70%

### Document Specifications

**PDF Support:**
- Max file size: 50MB
- Formats: PDF only for MVP
- Storage: Cloudflare R2 (zero egress fees, global CDN)
- Processing: Metadata extraction, page count, text extraction for RAG

**Knowledge Base Content:**
- Types: PDF, video (YouTube URLs), text, web pages
- Base content: Curated piano pedagogy materials (public)
- Teacher content: Teacher-specific uploads (private)
- Processing: Chunking (512 tokens), embedding generation (BGE-base-v1.5), vector indexing

### Scalability Targets

- **MVP (Beta)**: 50-100 users, ~500 PDF projects, ~10K knowledge base chunks
- **Post-MVP**: 500 users, ~5K projects, ~100K chunks
- **Phase 2**: 5K users, ~50K projects, ~1M chunks (pgvector supports up to 5M)

---

## Technology Stack

### Edge Layer (Cloudflare)
- **Storage**: R2 (PDFs, processed assets)
- **Cache**: Workers KV (embeddings, search results, LLM responses)
- **Compute**: Workers AI (BGE-base-v1.5 embeddings, cross-encoder re-ranking)
- **CDN**: Global edge network (300+ locations)

### Database Layer (Supabase)
- **Primary Database**: PostgreSQL 16+ with pgvector extension
- **Vector Indexing**: HNSW (99% recall, <8ms search)
- **Authentication**: Supabase Auth (JWT-based)
- **Realtime**: (Future) WebSocket support for collaborative features

### Compute Layer (GCP)
- **API Servers**: Compute Engine (Rust + Axum)
- **LLM Integration**: Claude 4.5 Haiku via Vertex AI
- **Background Workers**: Cloud Run Jobs (PDF processing, embedding generation)
- **Future**: GPU instances for Audio Spectrogram Transformer

### Observability
- **Metrics**: Prometheus + Grafana
- **Logging**: Structured JSON logs (Cloud Logging)
- **Tracing**: OpenTelemetry → Cloud Trace
- **Alerting**: Cloud Monitoring alerts

---

## Success Metrics

### Performance KPIs

**Must Achieve Before Frontend Development:**
- RAG query P95 latency: <200ms (target: <180ms)
- Vector search recall: >99% (critical for accuracy)
- API endpoint P95 latency: <50ms (target: <40ms)
- Database query P95 latency: <8ms (target: <6ms)
- Edge cache hit rate: >70% (embeddings, search results)

### Functional Validation

- Successfully ingest and query 1000+ knowledge base documents
- Achieve 99% recall on teacher-uploaded content queries
- Handle 100 concurrent RAG queries without degradation
- Store and retrieve 500+ PDF projects with annotations
- Manage 50 teacher-student relationships
- Process 10MB PDF uploads reliably

### Load Testing Targets

- Simulate 100 concurrent users
- 500 requests/second sustained for 5 minutes
- No degradation under 2x expected load
- Graceful degradation at 5x load

---

## Non-Functional Requirements

### Security

- **Authentication**: JWT tokens via Supabase Auth (RS256), 15-min access tokens, 7-day refresh
- **Authorization**: Relationship-based access (students see only their teacher's content)
- **Encryption**: TLS 1.3 in transit, AES-256 at rest (Supabase + R2)
- **Data Privacy**: GDPR-compliant data handling, user deletion support
- **Rate Limiting**: Cloudflare (100 req/min per IP), API-level (1K req/hr per user)

### Reliability

- **Uptime**: 99.9% (acceptable for MVP)
- **Data Durability**: 99.999999999% (11 nines via R2 + Supabase backups)
- **Backups**: Daily PostgreSQL snapshots (Supabase), 7-day retention
- **Error Handling**: Graceful degradation, circuit breakers on external APIs
- **Failover**: Supabase automatic failover, GCP instance auto-restart

### Observability

- **Metrics**: Prometheus (1s granularity for critical paths)
- **Logging**: Structured JSON logs (searchable via Cloud Logging)
- **Tracing**: OpenTelemetry distributed traces (RAG pipeline, DB queries, edge requests)
- **Dashboards**: Grafana (API latency, database performance, cache hit rates, RAG metrics, vector search recall)
- **Alerting**: <5min detection for P95 latency violations or recall degradation

---

## User Roles & Workflows

### Teacher Workflow

1. Upload piano pedagogy content to knowledge base (PDFs, videos)
2. Create PDF projects (sheet music assignments)
3. Grant students access to specific projects
4. View student annotations on PDFs
5. Query RAG system about techniques, pieces, pedagogy
6. Add annotations/feedback to student work
7. Verify RAG answers cite their uploaded content (accuracy validation)

### Student Workflow

1. Access projects shared by teacher
2. View PDF sheet music (served from R2 via Cloudflare CDN)
3. Add annotations (questions, struggles, notes)
4. Query RAG system for help with techniques, theory
5. View teacher feedback/annotations
6. Trust RAG answers for technique-critical questions (requires 99% accuracy)

---

## Data Model

### Core Entities

**Users:**
- Roles: teacher, student, admin
- Auth via Supabase Auth (JWT)
- Profile: email, name, role

**Teacher-Student Relationships:**
- Many-to-many (teacher can have multiple students, student can have multiple teachers)

**Projects:**
- PDF-centric (each project = 1 PDF)
- Owner: teacher or student
- Storage: R2 bucket (presigned URL access)
- Access control: explicit grants (view/edit permissions)

**Annotations:**
- Per-page, per-project
- Types: highlight, note, drawing
- JSONB payload for flexibility (coordinates, text, styling)

**Knowledge Base Documents:**
- Base content (public, platform-provided)
- Teacher content (private, teacher-specific)
- Chunked and embedded for RAG
- Stored in Supabase with pgvector embeddings

**Chat Sessions:**
- Per-project chat history
- User queries + AI responses with source citations
- Confidence scores for answer quality

---

## RAG Pipeline Details

### Content Ingestion

1. Upload document (PDF to R2, video URL to DB)
2. Extract text:
   - PDFs: Apache Tika or pdfium-render
   - Videos: YouTube transcript API or Whisper
3. Chunk content:
   - 512 tokens per chunk
   - 128 token overlap (context continuity)
   - Preserve metadata (page number, timestamp, source)
4. Generate embeddings:
   - Model: BGE-base-v1.5 (768 dimensions)
   - Service: Cloudflare Workers AI (50ms per embedding)
   - Batch processing for background ingestion
5. Store in Supabase:
   - Table: `document_chunks`
   - Columns: `id`, `doc_id`, `content`, `embedding` (vector[768]), `metadata` (JSONB)
6. Index for hybrid search:
   - HNSW index on `embedding` (99% recall)
   - GIN index on `content` for BM25 full-text search

### Query Pipeline

```
User Query
    ↓
Cloudflare Worker (edge routing)
    ↓
Check KV cache for embedding (70% hit rate)
    ├─ HIT: Use cached embedding (<5ms)
    └─ MISS: Generate via Workers AI (50ms)
    ↓
GCP Compute Engine (Rust API)
    ↓
Hybrid Search (Supabase pgvector)
    ├─ Vector similarity (HNSW, Top-20, 99% recall, 8ms)
    └─ BM25 keyword search (GIN index, Top-20, 3ms)
    ↓
Reciprocal Rank Fusion (merge results, in-memory, 2ms)
    ↓
Re-ranking (Cloudflare Workers AI: cross-encoder, 20ms)
    ↓ Top-3 most relevant chunks
Check KV cache for LLM response (40% hit rate)
    ├─ HIT: Stream cached response (<5ms)
    └─ MISS: Continue to LLM
        ↓
    Claude 4.5 Haiku (Vertex AI, 100ms TTFT)
        ↓
    Stream response to user + cache in KV
    ↓
Return with source citations + confidence score
```

**Total Latency:**
- Cold path: 50ms (embed) + 8ms (vector) + 3ms (BM25) + 2ms (RRF) + 20ms (rerank) + 100ms (LLM) = **183ms P95**
- Cached embedding: 5ms + 8ms + 3ms + 2ms + 20ms + 100ms = **138ms P95**
- Fully cached: **<10ms** (edge KV)

**Accuracy:**
- Vector search recall: 99%
- After re-ranking: 99.5%+
- Source citations: 100% (always provided)

### Caching Strategy

**3-Layer Cache (Cloudflare Workers KV):**

1. **Query Embedding Cache**
   - Key: `embed:query:{sha256(query_text)}`
   - Value: Float array (768 dimensions, binary serialized)
   - TTL: 24 hours
   - Hit rate target: >70%
   - Saves: ~45ms per hit

2. **Search Result Cache**
   - Key: `search:{sha256(query_text)}:{user_filters}`
   - Value: JSON array of top-3 chunks with metadata
   - TTL: 1 hour (teacher content changes infrequently)
   - Hit rate target: >60%
   - Saves: ~40ms per hit

3. **LLM Response Cache**
   - Key: `llm:{sha256(query + context)}`
   - Value: Full response text + sources + confidence
   - TTL: 24 hours
   - Hit rate target: >40% (common student questions repeat)
   - Saves: ~350ms per hit

**Cache Invalidation:**
- Teacher uploads new content → invalidate search cache for that teacher
- Knowledge base updated → invalidate related search caches
- Manual purge endpoint for admins

---

## API Overview

### Core Endpoints

**Authentication (Supabase Auth):**
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Authenticate
- `POST /api/auth/refresh` - Refresh JWT token

**Projects:**
- `POST /api/projects` - Create project, get presigned R2 URL
- `GET /api/projects` - List user's accessible projects
- `GET /api/projects/:id` - Get project details + R2 presigned URL
- `POST /api/projects/:id/access` - Grant user access
- `DELETE /api/projects/:id` - Delete project

**Annotations:**
- `POST /api/annotations` - Create annotation
- `GET /api/annotations?project_id=&page=` - Get page annotations
- `PATCH /api/annotations/:id` - Update annotation
- `DELETE /api/annotations/:id` - Delete annotation

**RAG:**
- `POST /api/chat/query` - Query RAG (streaming SSE response)
- `GET /api/chat/sessions` - List user's chat sessions
- `GET /api/chat/sessions/:id` - Get session messages
- `POST /api/chat/sessions` - Create new session
- `DELETE /api/chat/sessions/:id` - Delete session

**Knowledge Base:**
- `POST /api/knowledge/upload` - Upload content (get presigned R2 URL)
- `POST /api/knowledge/:id/process` - Trigger embedding generation (async job)
- `GET /api/knowledge/:id/status` - Check processing status
- `GET /api/knowledge` - List available content

**Performance Testing:**
- `GET /api/health` - Health check (API + DB + edge connectivity)
- `GET /api/metrics` - Prometheus metrics
- `POST /api/admin/cache/purge` - Purge edge cache (admin only)

---

## Development Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

- Set up Cloudflare account, configure R2 buckets
- Set up Supabase project, enable pgvector extension
- Set up GCP project, create Compute Engine instances
- Configure networking (VPC, firewall rules)
- Implement authentication (Supabase Auth + JWT middleware)
- Basic API framework (Rust + Axum on GCP Compute)

### Phase 2: PDF & Annotation System (Weeks 3-4)

- PDF upload/storage (presigned R2 URLs)
- Cloudflare CDN configuration for R2
- Project CRUD APIs
- Annotation CRUD APIs
- Access control logic (relationship-based)

### Phase 3: RAG Pipeline (Weeks 5-8)

- Embedding service (Cloudflare Workers AI integration)
- Knowledge base ingestion pipeline (background workers)
- pgvector HNSW index configuration and optimization
- Hybrid search implementation (vector + BM25)
- Re-ranking with cross-encoder (Workers AI)
- LLM integration (Claude 4.5 Haiku via Vertex AI)
- Edge caching layer (Cloudflare Workers KV)

### Phase 4: Performance Testing & Optimization (Weeks 9-10)

- Load testing (k6 scripts for 100 concurrent users)
- Database query optimization (EXPLAIN ANALYZE, index tuning)
- pgvector memory configuration (keep HNSW in RAM)
- Cache tuning (TTL optimization, hit rate monitoring)
- Benchmark validation (P95 latency, recall accuracy)
- Documentation (architecture diagrams, runbooks)

---

## Cost Estimation

### MVP (100 users, 50K RAG queries/month)

| Component | Cost |
|-----------|------|
| **Cloudflare** | |
| - R2 Storage (2.5TB) | $38 |
| - Workers AI (embeddings + reranking) | $25 |
| - Workers KV (cache) | $5 |
| **Supabase** | |
| - Pro plan (PostgreSQL + Auth) | $25 |
| - Compute addon (4XL for pgvector) | $100 |
| **GCP** | |
| - Compute Engine (n2-standard-4) | $120 |
| - Claude 4.5 Haiku API | $25 |
| - Cloud Logging/Monitoring | $20 |
| **Total** | **~$358/month** |

### At Scale (5K users, 2M queries/month)

| Component | Cost |
|-----------|------|
| Cloudflare (Workers + R2 + KV) | $180 |
| Supabase (Pro + 8XL compute) | $250 |
| GCP (Compute + Claude + Monitoring) | $400 |
| **Total** | **~$830/month** |

---

## Future Considerations

**Post-MVP Enhancements:**
- Frontend development (once backend validated)
- Real-time collaborative annotations (Supabase Realtime)
- Audio recording upload and storage (R2)
- Multi-region deployment (Supabase regional read replicas)
- Advanced analytics dashboards (student progress tracking)

**Long-Term Vision:**
- Audio Spectrogram Transformer for performance feedback (GCP GPU instances)
- Real-time AI tutoring during practice sessions
- Multi-instrument support
- Mobile native apps
- Migration to Cloudflare Vectorize for global edge queries (once recall improves to 95%+)
