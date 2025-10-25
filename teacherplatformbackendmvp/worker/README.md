# Cloudflare Worker - Edge Layer

Edge routing and caching layer for Piano Platform. Deployed globally to 300+ locations.

## Architecture

```
User Request
    ↓
Cloudflare Worker (Edge - this code)
    ├─ Check KV Cache (embedding/search/LLM)
    ├─ If HIT: Return immediately (<5ms)
    └─ If MISS: Call GCP API → Cache → Return
         ↓
GCP Rust API (../api/)
    ├─ Database queries (Supabase)
    ├─ RAG pipeline
    └─ LLM calls (Vertex AI)
```

## Features

- **3-Layer KV Caching**: Embeddings (24hr), Search (1hr), LLM responses (24hr)
- **R2 Bindings**: Zero-latency PDF access
- **Workers AI Bindings**: Embeddings + re-ranking at edge
- **Smart Proxying**: Complex operations forwarded to GCP API

## Setup

### 1. Install worker-build

```bash
cargo install worker-build
```

### 2. Create KV Namespaces

```bash
# Embedding cache
wrangler kv:namespace create "EMBEDDING_CACHE"
# Copy the ID and update wrangler.toml

# Search cache
wrangler kv:namespace create "SEARCH_CACHE"
# Copy the ID and update wrangler.toml

# LLM response cache
wrangler kv:namespace create "LLM_CACHE"
# Copy the ID and update wrangler.toml
```

### 3. Create R2 Buckets

```bash
wrangler r2 bucket create piano-pdfs
wrangler r2 bucket create piano-knowledge
```

### 4. Set Secrets

```bash
wrangler secret put SUPABASE_JWT_SECRET
# Paste your JWT secret

wrangler secret put GCP_API_TOKEN
# Optional: Token for Worker -> GCP API authentication
```

### 5. Update wrangler.toml

- Replace KV namespace IDs with the ones you created
- Update `GCP_API_URL` to your deployed GCP API endpoint

### 6. Deploy

```bash
# Development (local)
wrangler dev

# Production
wrangler deploy
```

## Local Development

```bash
# Terminal 1: Run GCP API
cd ../api
cargo run

# Terminal 2: Run Worker locally
cd ../worker
wrangler dev

# Test
curl http://localhost:8787/health
```

## Performance Targets

- **Cached queries**: <5ms P95
- **Cache miss (edge → GCP)**: <200ms P95
- **Cache hit rate**: >70% (embeddings), >60% (search), >40% (LLM)

## Best Practices

1. **Bindings > REST APIs**: Use R2, KV, AI bindings (zero latency) instead of REST calls
2. **Keep Workers thin**: Complex logic stays in GCP API
3. **Cache aggressively**: Edge caching is your performance superpower
4. **Monitor cache hit rates**: Should be >70% for embeddings

## Routes

- `GET /health` - Health check
- `POST /api/chat/query` - RAG query (with caching)
- `POST /api/embeddings/generate` - Generate embeddings (with caching)
- `POST /api/projects/upload-url` - Generate R2 presigned upload URL
- `GET /api/projects/:id/download-url` - Generate R2 presigned download URL
- `* /api/*` - Proxy to GCP API for complex operations

## Monitoring

View Worker metrics in Cloudflare dashboard:
- Requests per second
- P50/P95/P99 latency
- Cache hit rates
- Error rates
- CPU time

## Cost

Workers:
- First 100K requests/day: FREE
- After: $0.30 per million requests

R2:
- Storage: $0.015/GB/month
- Zero egress fees (huge saving!)

KV:
- First 100K reads/day: FREE
- Writes: $0.50 per million
- Storage: $0.50/GB/month

Workers AI:
- Embeddings: ~$0.001 per 1K requests
- Re-ranking: ~$0.001 per 1K requests
