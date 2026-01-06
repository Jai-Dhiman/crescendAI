# Piano Performance Feedback - System Architecture

## System Overview

A three-tier architecture leveraging Cloudflare's edge platform for the frontend and API layer, with GPU inference offloaded to RunPod serverless.

```jsonl
User Browser <-> Cloudflare Edge <-> RunPod GPU
                      |
                      +-> Vectorize (RAG)
                      +-> Workers AI (LLM)
```

## Technology Stack

### Frontend

- **Framework**: Leptos (Rust)
- **Compilation**: WebAssembly (WASM)
- **Hosting**: Cloudflare Pages
- **Styling**: Tailwind CSS (or minimal custom CSS)

### API Layer

- **Runtime**: Cloudflare Workers
- **Language**: Rust (workers-rs)
- **Storage**: Cloudflare R2 (audio files, precomputed features)
- **Cache**: Cloudflare KV (analysis results)

### AI/ML Layer

- **Model Inference**: RunPod Serverless (Python + PyTorch)
- **LLM Feedback**: Cloudflare Workers AI (Llama 3.3 70B)
- **RAG Retrieval**: Cloudflare Vectorize + BGE embeddings

### Data Pipeline (Offline)

- **Feature Extraction**: Python (VirtuosoNet/pyScoreParser)
- **Audio Transcription**: Python (Basic Pitch or Onsets and Frames)
- **Embedding Generation**: Python (sentence-transformers)

## Component Architecture

### Cloudflare Pages (Frontend)

Leptos application compiled to WASM, served statically.

**Responsibilities**:

- Render performance gallery
- Audio playback with waveform visualization
- Display analysis results (radar chart, feedback text)
- Handle loading states and error display

**Key Dependencies**:

- leptos (reactive UI framework)
- gloo-net (HTTP requests)
- web-sys (Web APIs for audio)

### Cloudflare Workers (API)

Rust-based serverless functions handling request orchestration.

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/performances` | GET | List available demo performances |
| `/api/performances/:id` | GET | Get performance metadata |
| `/api/analyze/:id` | POST | Trigger analysis pipeline |

**Responsibilities**:

- Serve preloaded performance metadata
- Orchestrate RunPod inference call
- Query Vectorize for RAG context
- Call Workers AI for feedback generation
- Cache results to avoid redundant computation

**Key Dependencies**:

- workers-rs (Cloudflare Workers bindings)
- serde/serde_json (serialization)
- reqwest or fetch API (HTTP to RunPod)

### RunPod Serverless (GPU Inference)

Python-based serverless GPU endpoint running the PercePiano model.

**Responsibilities**:

- Load precomputed VirtuosoNet features from bundled storage
- Run HAN model inference (PyTorch)
- Return 19-dimensional prediction vector

**Container Contents**:

- PyTorch runtime with CUDA support
- PercePiano model checkpoint (best.pt)
- Precomputed feature files for all demo performances
- RunPod handler script

**Hardware**: NVIDIA T4 GPU (cost-effective for inference)

### Cloudflare Vectorize (RAG)

Vector database storing piano knowledge embeddings.

**Content Categories**:

- Piano pedagogy (technique, practice methods)
- Performance practice (historical context)
- Masterclass transcripts
- Competition judging criteria

**Configuration**:

- Embedding model: @cf/baai/bge-base-en-v1.5 (768 dimensions)
- Index size: ~5,000 chunks
- Query: Top-5 retrieval per analysis

### Cloudflare Workers AI (LLM)

Serverless LLM inference for generating teacher feedback.

**Model**: @cf/meta/llama-3.3-70b-instruct-fp8-fast

**Input**: Structured prompt containing:

- 19 dimension scores
- Retrieved RAG context
- Tone/style instructions

**Output**: 150-200 word encouraging feedback narrative

## Data Flow

### Analysis Request Flow

```
1. User clicks "Analyze" on performance X
   |
2. Leptos frontend -> POST /api/analyze/X
   |
3. Cloudflare Worker receives request
   |
4. Worker checks KV cache for existing result
   |  (if cached, skip to step 8)
   |
5. Worker calls RunPod endpoint with performance_id
   |  - RunPod loads precomputed features
   |  - RunPod runs HAN inference
   |  - RunPod returns 19 predictions
   |
6. Worker queries Vectorize with prediction context
   |  - "timing=0.7, articulation=0.85, Chopin Ballade"
   |  - Returns top-5 relevant chunks
   |
7. Worker calls Workers AI with predictions + RAG context
   |  - LLM generates teacher feedback
   |
8. Worker caches result in KV, returns to frontend
   |
9. Leptos renders radar chart + feedback
```

### Precomputation Pipeline (Offline)

```
For each demo performance:
   |
1. Obtain audio file (WAV) + MusicXML score
   |
2. Transcribe audio to MIDI (Basic Pitch)
   |
3. Align MIDI to score (pyScoreParser)
   |
4. Extract VirtuosoNet features (79 dimensions per note)
   |
5. Apply z-score normalization (using training stats)
   |
6. Save as .pt file with note locations
   |
7. Bundle into RunPod container
```

## Rust/Python Boundary

### Rust Components

| Component | Justification |
|-----------|---------------|
| Frontend (Leptos) | Learning goal, type safety, WASM performance |
| API Worker | Learning goal, Cloudflare native support |
| MIDI parsing (midly) | Rust crate available, simple task |

### Python Components

| Component | Justification |
|-----------|---------------|
| VirtuosoNet features | Requires pyScoreParser (no Rust equivalent) |
| HAN model inference | PyTorch model, complex architecture |
| Audio transcription | TensorFlow models, no ONNX export ready |
| Precomputation scripts | One-time offline task, Python ecosystem |

### Future Rust Migration Path

1. MIDI parsing: Already possible with `midly`
2. Model inference: Convert to ONNX, use `ort` crate
3. Feature extraction: Would require porting pyScoreParser (~3 months)

## Performance Considerations

### Latency Budget (Target: <15s)

| Stage | Target | Mitigation |
|-------|--------|------------|
| Frontend -> Worker | <100ms | Edge-local |
| Worker -> RunPod | <5s | Keep warm worker |
| RunPod inference | <500ms | Precomputed features |
| Worker -> Vectorize | <100ms | Edge-local |
| Worker -> Workers AI | <3s | Streaming if needed |
| Worker -> Frontend | <100ms | Edge-local |
| **Total** | **<9s warm** | Cache results |

### Cold Start Mitigation

- RunPod: Configure min_workers=1 during demo periods
- Workers: No cold start (always warm)
- Vectorize: No cold start (managed service)

### Caching Strategy

- **KV Cache**: Store analysis results by performance_id
- **TTL**: 24 hours (results don't change for preloaded content)
- **Cache key**: `analysis:{performance_id}`

## Cost Estimate

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Cloudflare Pages | Static hosting | $0 (free tier) |
| Cloudflare Workers | API requests | $0 (free tier, <100k req) |
| Cloudflare KV | Result cache | $0 (free tier) |
| Cloudflare Vectorize | 5k vectors, queries | $0 (free tier) |
| Cloudflare Workers AI | LLM inference | ~$5-10 |
| Cloudflare R2 | Audio/feature storage | $0 (<10GB) |
| RunPod Serverless | GPU inference | ~$5-10 |
| **Total** | | **~$15/month** |

## Security Considerations

- No user data collected or stored
- No authentication required (public demo)
- RunPod endpoint protected by API key (stored in Workers secrets)
- Rate limiting on /api/analyze to prevent abuse
- CORS configured to allow only the demo domain

## Monitoring and Observability

- Cloudflare Analytics: Request volume, latency, errors
- RunPod Dashboard: GPU utilization, cold starts
- Workers AI Dashboard: Token usage, model performance
- Manual testing: Pre-demo verification script

## Failure Modes and Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|----------|
| RunPod timeout | HTTP 504 | Show cached result or error message |
| Workers AI error | HTTP 5xx | Return predictions without LLM feedback |
| Vectorize empty | Empty results | Skip RAG context in prompt |
| Invalid performance_id | 404 | Show "Performance not found" |

## Development and Deployment

### Local Development

- Frontend: `trunk serve` (Leptos dev server)
- Worker: `wrangler dev` (local Workers runtime)
- RunPod: Local Python with mock handler

### Deployment Pipeline

1. Frontend: Push to GitHub -> Cloudflare Pages auto-deploy
2. Worker: `wrangler deploy`
3. RunPod: Docker build -> Push to registry -> Update endpoint
4. Vectorize: One-time setup via Wrangler CLI

### Environment Configuration

- **Development**: Local services, mock RunPod
- **Staging**: Cloudflare preview URLs, RunPod dev endpoint
- **Production**: Custom domain, RunPod prod endpoint
