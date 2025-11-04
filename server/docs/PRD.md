# CrescendAI Server - Product Requirements Document

**Version:** 2.0
**Last Updated:** 2025-01-30
**Status:** Planning
**Author:** CrescendAI Team

---

## Executive Summary

CrescendAI Server is a piano education platform backend that combines AI-powered performance analysis with a comprehensive RAG (Retrieval-Augmented Generation) system. Students upload piano recordings and receive personalized, actionable feedback by combining:

1. **16-dimensional performance analysis** from an AST (Audio Spectrogram Transformer) model
2. **Piano pedagogy knowledge base** with RAG search over expert teaching materials
3. **Conversational AI tutoring** powered by Dedalus orchestration and GPT-5-nano

This document defines the product vision, requirements, and success criteria for the Cloudflare Workers-based backend redesign.

---

## Product Vision

**Mission:** Democratize expert piano pedagogy by providing instant, personalized feedback that combines quantitative performance analysis with qualitative pedagogical expertise.

**Vision:** Every piano student, regardless of location or access to expert teachers, receives world-class feedback on their playing through AI-powered analysis grounded in proven teaching methods.

---

## Target Users

### Primary Users

- **Piano students** (beginner to advanced) seeking feedback on recordings
- **Self-taught pianists** lacking access to regular teachers
- **Students between lessons** wanting interim guidance

### Secondary Users

- **Piano teachers** using the platform to augment their teaching
- **Music schools** integrating automated feedback into curricula

---

## User Stories

### Core User Journey

**As a piano student, I want to:**

1. **Upload a recording** of my playing
   - Accept WAV/MP3/M4A formats
   - Validate file size (max 50MB) and audio quality
   - Receive immediate confirmation of successful upload

2. **Ask questions** about my performance in natural language
   - "How can I improve my pedaling in this piece?"
   - "What exercises should I practice for better timing?"
   - "Is my interpretation of this Chopin piece convincing?"

3. **Receive structured feedback** that includes:
   - **Technical observations** based on 16D performance scores
   - **Specific practice exercises** referenced from pedagogy literature
   - **Temporal feedback** highlighting specific moments in the recording
   - **Citations** to authoritative teaching materials

4. **Have a conversation** with the AI tutor
   - Ask follow-up questions
   - Get clarification on recommendations
   - Discuss practice strategies
   - Receive real-time streaming responses

5. **Track progress** over multiple recordings
   - See improvement in performance dimensions
   - Build a practice history
   - Get adaptive recommendations based on past sessions

---

## Functional Requirements

### FR-1: Recording Upload & Storage

**Priority:** P0 (Critical)

**Requirements:**

- Accept audio files via `POST /api/upload`
- Supported formats: WAV, MP3, M4A
- Max file size: 50MB
- Store in Cloudflare R2 with unique ID
- Return recording ID immediately
- Metadata stored in D1: upload timestamp, file size, user context

**Acceptance Criteria:**

- Upload completes in <2 seconds for 10MB file
- Files are accessible via presigned R2 URLs
- Duplicate detection based on audio hash

---

### FR-2: Performance Analysis (AST Model)

**Priority:** P0 (Critical)

**Requirements:**

- Analyze recording using 16-dimensional AST model
- Model deployed on Modal (external HTTP endpoint)
- Return scores for:
  - Timing stability
  - Articulation (length, hardness)
  - Pedaling (density, clarity)
  - Timbre (color, richness, brightness, dynamics)
  - Musical interpretation (phrasing, expression, emotion)
- Store results in D1 for retrieval
- **Mock implementation initially** (return dummy scores)

**Acceptance Criteria:**

- Analysis completes in <10 seconds (when model integrated)
- Scores are normalized 0.0-1.0
- Temporal segments (3-second windows) with per-segment scores
- Mock returns realistic dummy data for development

---

### FR-3: RAG Knowledge Base

**Priority:** P0 (Critical)

**Requirements:**

- Ingest piano pedagogy documents (PDFs, text)
- Chunk documents (1000 chars, 20% overlap)
- Generate embeddings using Cloudflare Workers AI (BGE-base)
- Store embeddings in Vectorize
- Store chunk metadata in D1
- Hybrid search: semantic (Vectorize) + full-text (D1)
- Re-rank top results using BGE reranker

**Acceptance Criteria:**

- Search latency <100ms P95
- Top-3 recall >90% for known queries
- Citations include source, page, and relevance score
- Support for 100+ documents (expandable to 1000+)

---

### FR-4: Conversational Chat Interface

**Priority:** P0 (Critical)

**Requirements:**

- `POST /api/chat` endpoint for streaming chat
- Server-Sent Events (SSE) for real-time token streaming
- Full message history stored in D1
- User messages, assistant responses, and tool calls persisted
- Session context includes:
  - Recording ID
  - User goals and constraints
  - Repertoire information
  - Chat history

**Acceptance Criteria:**

- First token latency <1 second
- Streaming updates every 50-100ms
- Chat sessions persist across page reloads
- Message history retrievable via `GET /api/chat/sessions/:id`

---

### FR-5: Structured Feedback Generation

**Priority:** P0 (Critical)

**Requirements:**

- `POST /api/feedback/:recording_id` endpoint
- Combine AST scores + RAG search results
- Generate structured feedback:
  - Overall assessment (strengths, priority areas, character)
  - Temporal feedback (timestamp-based insights)
  - Practice recommendations (immediate priorities, long-term goals)
  - Encouragement message
- Use Dedalus for LLM orchestration
- GPT-5-nano as primary model

**Acceptance Criteria:**

- Feedback generation <5 seconds
- 3-5 temporal segments per recording
- 3-5 immediate practice priorities
- 2-3 long-term development goals
- All recommendations cite pedagogy sources

---

### FR-6: Dedalus AI Orchestration

**Priority:** P0 (Critical)

**Requirements:**

- Call Dedalus HTTP API from Rust Workers
- Support tool calling for:
  - `search_knowledge_base(query, top_k)`
  - `get_performance_analysis(recording_id)`
  - `get_user_context(session_id)`
- Handle streaming responses
- Implement retry logic with exponential backoff
- Cache LLM responses in KV

**Acceptance Criteria:**

- HTTP client with proper error handling
- Tool calls execute correctly
- Streaming tokens parsed from SSE
- Cache hit rate >40% for common queries

---

### FR-7: User Context Management

**Priority:** P1 (High)

**Requirements:**

- Store user context in D1:
  - Goals (e.g., "improve timing", "prepare for recital")
  - Practice time per day
  - Constraints (e.g., "limited wrist mobility")
  - Repertoire info (composer, piece, difficulty)
- Update context via `PUT /api/context`
- Include context in all AI calls

**Acceptance Criteria:**

- Context persists across sessions
- Feedback adapts to user goals
- Constraints reflected in recommendations

---

## Non-Functional Requirements

### NFR-1: Performance

- **Upload latency:** <2s for 10MB file
- **Analysis latency:** <10s (model inference)
- **RAG search latency:** <100ms P95
- **Chat first token:** <1s
- **Feedback generation:** <5s
- **API P95 latency:** <500ms (excluding LLM)

### NFR-2: Scalability

- Support 1000 concurrent users
- Handle 10K recordings/day
- Scale Vectorize to 100K+ document chunks
- D1 supports 10M+ messages

### NFR-3: Reliability

- 99.9% uptime for API endpoints
- Graceful degradation if Modal model unavailable
- Retry logic for transient failures
- Data persistence guaranteed (D1 + R2)

### NFR-4: Security

- API key authentication (development)
- Rate limiting: 100 req/min per user
- Input validation (file types, sizes, JSON schemas)
- CORS with origin whitelisting
- No PII in logs

### NFR-5: Observability

- Request logging with timing
- Error tracking with stack traces
- Cache hit/miss metrics
- Model inference metrics
- D1 query performance monitoring

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Priority |
|--------|----------|-------------|----------|
| POST | /api/upload | Upload audio recording | P0 |
| POST | /api/chat | Start streaming chat session | P0 |
| GET | /api/chat/sessions/:id | Retrieve session history | P0 |
| POST | /api/feedback/:recording_id | Generate structured feedback | P0 |
| GET | /api/recordings/:id | Get recording metadata | P1 |
| PUT | /api/context | Update user context | P1 |
| POST | /api/knowledge/ingest | Ingest documents to KB | P1 |
| GET | /api/knowledge/search | Direct RAG search | P2 (debug) |
| GET | /api/health | Health check | P0 |

---

## Success Metrics

### Product Metrics

- **User Engagement:**
  - 80% of users ask ≥1 follow-up question
  - Average 5+ messages per chat session
  - 60% of users upload multiple recordings

- **Quality Metrics:**
  - >4.0/5 feedback quality rating
  - >90% citation accuracy (sources exist and relevant)
  - <5% user-reported hallucinations

- **Performance Metrics:**
  - P95 feedback latency <5s
  - P95 chat first token <1s
  - Cache hit rate >40%

### Technical Metrics

- **Reliability:**
  - 99.9% API uptime
  - <0.1% error rate
  - <1% failed uploads

- **Efficiency:**
  - D1 query time <10ms P95
  - Vectorize search <50ms P95
  - KV cache hit rate >40%

---

## Out of Scope (V2.0)

The following features are **not** included in this redesign:

- ❌ User authentication (beyond API keys)
- ❌ Payment/subscription management
- ❌ Teacher dashboard
- ❌ Real-time collaborative features
- ❌ Mobile app (API-only backend)
- ❌ Video upload/analysis
- ❌ MIDI file analysis (future)
- ❌ Social features (sharing, comments)
- ❌ Multi-language support

---

## Dependencies

### External Services

1. **Dedalus API** - AI orchestration and LLM calls
2. **Modal** - AST model hosting (future integration)
3. **Cloudflare Platform:**
   - Workers (compute)
   - D1 (database)
   - Vectorize (vector search)
   - R2 (object storage)
   - KV (caching)
   - Workers AI (embeddings)

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Dedalus API downtime | High | Low | Fallback to direct OpenAI API, retry logic |
| Modal model latency >10s | Medium | Medium | Mock with dummy scores, async processing |
| Vectorize limitations | Medium | Low | Fallback to KV-only RAG, external vector DB |
| D1 query performance | High | Medium | Indexing strategy, caching, query optimization |
| GPT-5-nano quality issues | High | Low | A/B test models, allow model switching |

---

## Timeline & Phases

### Phase 1: Foundation (Week 1-2)

- D1 schema design
- Dedalus HTTP client
- Basic RAG with Vectorize

### Phase 2: Core Features (Week 2-3)

- Streaming chat handler
- Structured feedback generation
- User context management

### Phase 3: Polish & Integration (Week 3-4)

- Modal model integration (or keep mock)
- Advanced RAG (reranker, hybrid search)
- Testing & optimization

### Phase 4: Documentation & Deployment (Week 4)

- API documentation
- Deployment automation
- Monitoring setup

---

## Appendix

### Performance Dimensions (16D AST Model)

1. **Timing:** stable ↔ unstable
2. **Articulation (length):** short ↔ long
3. **Articulation (hardness):** soft ↔ hard
4. **Pedaling (density):** sparse ↔ saturated
5. **Pedaling (clarity):** clean ↔ blurred
6. **Timbre (color):** even ↔ colorful
7. **Timbre (richness):** shallow ↔ rich
8. **Timbre (brightness):** bright ↔ dark
9. **Timbre (dynamics):** soft ↔ loud
10. **Dynamic control:** sophisticated ↔ raw
11. **Dynamic range:** little ↔ large
12. **Tempo:** fast ↔ slow
13. **Phrasing:** flat ↔ spacious
14. **Balance:** disproportioned ↔ balanced
15. **Expression:** pure ↔ dramatic
16. **Mood:** optimistic ↔ dark
17. **Energy:** low ↔ high
18. **Imagination:** honest ↔ imaginative
19. **Interpretation:** unsatisfactory ↔ convincing

### Glossary

- **AST:** Audio Spectrogram Transformer (performance analysis model)
- **RAG:** Retrieval-Augmented Generation (knowledge base + LLM)
- **D1:** Cloudflare's distributed SQLite database
- **Vectorize:** Cloudflare's vector search service
- **SSE:** Server-Sent Events (streaming protocol)
- **KV:** Cloudflare Workers KV (key-value store)
- **BGE:** BAAI General Embedding (embedding model)
