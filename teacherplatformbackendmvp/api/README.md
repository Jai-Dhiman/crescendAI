# Piano API - Teacher Platform Backend

Rust + Axum API server for the CrescendAI piano education platform with high-performance RAG, teacher-student workflows, and PDF project management.

## API Endpoints

### Authentication

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user

### Knowledge Base

- `POST /api/knowledge` - Create knowledge base document
- `GET /api/knowledge` - List accessible documents
- `GET /api/knowledge/:id` - Get document details
- `DELETE /api/knowledge/:id` - Delete document
- `POST /api/knowledge/:id/process` - Trigger PDF processing
- `GET /api/knowledge/:id/status` - Get processing status

### RAG Chat

- `POST /api/chat/query` - RAG query (streaming SSE response)
- `POST /api/chat/sessions` - Create chat session
- `GET /api/chat/sessions` - List user's sessions
- `GET /api/chat/sessions/:id` - Get session + messages
- `DELETE /api/chat/sessions/:id` - Delete session

### Relationships

- `POST /api/relationships` - Create teacher-student relationship
- `GET /api/relationships` - List relationships
- `DELETE /api/relationships/:id` - Remove relationship

### Projects (Coming Soon)

- `POST /api/projects` - Create project + presigned upload URL
- `GET /api/projects` - List accessible projects
- `GET /api/projects/:id` - Get project details
- `PUT /api/projects/:id` - Update project metadata
- `DELETE /api/projects/:id` - Delete project

### Annotations (Coming Soon)

- `POST /api/annotations` - Create annotation
- `GET /api/annotations?project_id=&page=` - Get page annotations
- `PATCH /api/annotations/:id` - Update annotation
- `DELETE /api/annotations/:id` - Delete annotation

### Health

- `GET /api/health` - Health check

## Architecture

### Tech Stack

- **Framework**: Axum (Rust async web framework)
- **Database**: PostgreSQL 16 + pgvector 0.8.0 (via Supabase)
- **AI/ML**: Cloudflare Workers AI
  - Embeddings: BGE-base-en-v1.5 (768-dim)
  - Reranking: BGE-reranker-base
  - LLM: Llama 4 Scout 17B
- **Storage**: Cloudflare R2 (S3-compatible)
- **Caching**: Cloudflare Workers KV (3-layer cache)

### RAG Pipeline

```
User Query
    ↓
1. Generate Embedding (Workers AI + KV cache)
    ↓
2. Hybrid Search (pgvector HNSW + BM25 full-text)
    ↓
3. Reciprocal Rank Fusion (merge results)
    ↓
4. Re-ranking (Workers AI cross-encoder)
    ↓
5. LLM Generation (Llama 4 Scout, streaming)
    ↓
6. Return with sources + confidence
```

### Knowledge Base Ingestion

```
PDF Upload
    ↓
1. Extract text (pdf-extract crate)
    ↓
2. Chunk (512 tokens, 128 overlap, tiktoken-rs)
    ↓
3. Generate embeddings (Workers AI, batches of 50)
    ↓
4. Store chunks + embeddings (PostgreSQL pgvector)
    ↓
5. Create HNSW index (m=16, ef_construction=64)
```

### Caching Strategy (3-layer)

1. **Embedding Cache** (24hr TTL)
   - Caches query embeddings to avoid redundant generation
   - Target: 70% hit rate
   - Reduces latency: ~100ms → ~10ms

2. **Search Cache** (1hr TTL)
   - Caches search results (chunk IDs + scores)
   - Target: 60% hit rate
   - Reduces latency: ~50ms → ~5ms

3. **LLM Cache** (disabled for streaming)
   - Could cache full responses for deterministic queries
   - Currently disabled to support streaming

## Documentation

- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) - Full system architecture
- [`../docs/PRD.md`](../docs/PRD.md) - Product requirements
- [`../docs/TASKS.md`](../docs/TASKS.md) - Implementation tasks & timeline
- [`../docs/WORKERS_AI_SETUP.md`](../docs/WORKERS_AI_SETUP.md) - Workers AI setup guide

## Contributing

See [`../CLAUDE.md`](../CLAUDE.md) for development guidelines.

## License

Proprietary - CrescendAI Platform
