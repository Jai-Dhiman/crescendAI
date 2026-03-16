# Local Inference Server for Dev Testing

> **Status:** Design spec. Enables zero-cost self-testing of the full practice pipeline by running MuQ + AMT inference locally on Mac (M4, 32GB) instead of paying for the HF endpoint.

## Problem

The HF inference endpoint is the largest recurring cost in the CrescendAI stack. For self-testing (single user validating feedback quality before packaging for beta testers), this cost is avoidable -- the M4 Mac with 32GB unified memory can run MuQ (~300M params) and ByteDance AMT locally via MPS.

The existing `eval_runner.py` proves local inference works but is a batch tool. We need a live HTTP server that the Cloudflare Worker can call during practice sessions, matching the HF endpoint's API contract.

## Goals

1. Run MuQ + AMT inference locally on Mac during practice sessions
2. Zero changes to the Cloudflare Worker's inference client code -- just a URL swap
3. Keep production path (HF endpoint) unchanged and easily switchable

## Non-Goals

- No new UI work (use web app as-is)
- No iOS support
- No inference caching or optimization
- No auth on the local server
- No production deployment of the local server
- No LLM cost optimization (Groq + Anthropic stay as-is)

---

## Design

### Component 1: Local Inference Server

**File:** `apps/inference/local_server.py`

A FastAPI server that wraps the existing `ModelCache` and `TranscriptionModel` infrastructure.

**Endpoints:**
- `POST /` -- inference (matches HF endpoint contract)
- `GET /health` -- returns `{"status": "ok"}` for connectivity checks

**Request format (POST /):** Raw audio bytes in the request body with `Content-Type: audio/webm;codecs=opus` (or any audio format). This matches what the Cloudflare Worker sends to the HF endpoint. The `Authorization` header is accepted but ignored.

**How raw bytes are received:** The HF Inference Endpoints framework automatically wraps raw bytes into `{"inputs": <bytes>}` before calling `EndpointHandler.__call__`. The local server replicates this: the FastAPI endpoint reads the raw body via `await request.body()` (not JSON parsing), then passes the bytes to the same processing pipeline. The response is a plain JSON dict returned directly -- no wrapper envelope.

**Processing:**
1. Read raw bytes via `await request.body()`
2. Pass to `preprocess_audio_from_bytes()` (existing function, handles WebM/Opus decoding)
3. Extract MuQ embeddings via `extract_muq_embeddings()` (existing)
4. Run A1-Max ensemble prediction via `predict_with_ensemble()` (existing)
5. Run AMT transcription via `TranscriptionModel.transcribe()` (existing)
6. Return JSON response matching the HF endpoint's response shape

**Response format:** Identical to `EndpointHandler.__call__` output:

```json
{
    "predictions": {"dynamics": 0.45, "timing": 0.62, ...},
    "midi_notes": [...],
    "pedal_events": [...],
    "transcription_info": {"note_count": 42, ...},
    "model_info": {"name": "A1-Max", ...},
    "audio_duration_seconds": 15.0,
    "processing_time_ms": 2500
}
```

**Error handling:** Exceptions during inference are caught and returned in the same format as `handler.py`: `{"error": {"code": "...", "message": "..."}}` with HTTP 200 (matching HF behavior). This keeps the Worker's response parsing consistent between local and production.

**Startup:** Loads all models into memory once on launch. Uses `CRESCEND_DEVICE=auto` (resolves to MPS on M4). Checkpoint directory resolved relative to `__file__` (not CWD): `Path(__file__).parents[1].parent / "model" / "data" / "checkpoints" / "model_improvement" / "A1"`, matching the approach in `eval_runner.py`.

**Invocation:** `cd apps/inference && uv run python local_server.py`

(`CRESCEND_DEVICE` defaults to `auto` via `_resolve_device()` in `models/loader.py`, no need to set explicitly.)

**Port:** `localhost:8000` (configurable via `--port` flag).

**Concurrency:** Runs with a single uvicorn worker. Inference is synchronous PyTorch code, so chunks serialize naturally. For single-user testing, chunks arrive every 15s and processing takes 3-5s -- no overlap concern. If two chunks do overlap (e.g., quick pause/resume), the second waits for the first to finish, which is acceptable.

**Dependencies:** `local_server.py` uses uv inline script metadata (`# /// script` header) to declare `fastapi` and `uvicorn` as dependencies. This avoids modifying `requirements.txt`, which is read by HF Inference Endpoints and should not include dev-only dependencies.

### Component 2: Worker URL Toggle

**Existing config:** `wrangler.toml` has `HF_INFERENCE_ENDPOINT` in `[vars]`.

**Change:** Add to `.dev.vars` (already exists, already gitignored):

```
HF_INFERENCE_ENDPOINT=http://localhost:8000
```

This overrides the production URL in `wrangler.toml` when running `wrangler dev`. No Rust code changes needed -- the Worker already reads `env.var("HF_INFERENCE_ENDPOINT")` and sends requests to whatever URL it finds.

**Prewarm behavior:** `prewarm_hf_endpoint()` in `start.rs` fires a background POST with `Content-Type: application/json` and body `"{}"` to `HF_INFERENCE_ENDPOINT` when a session starts. Against localhost, this hits `POST /` with non-audio content, which will return an error response. This is harmless (the prewarm result is ignored) but will produce error log noise in the local server console. Expected behavior -- not a bug.

**HF_TOKEN:** The Worker sends `Authorization: Bearer {HF_TOKEN}` on every inference request. The local server ignores this header. No change needed.

### Component 3: Dev Session Workflow

**Three terminals:**

1. `cd apps/inference && CRESCEND_DEVICE=auto uv run python local_server.py` -- models load in ~10-15s
2. `cd apps/api && npx wrangler dev` -- Worker on localhost:8787, reads `.dev.vars`
3. `cd apps/web && bun run dev` -- web app on localhost:5173

**Practice flow:**

1. Open localhost:5173, sign in via debug auth
2. Set piece context
3. Hit record, play piano
4. Browser sends 15s audio chunks to Worker (localhost:8787)
5. Worker forwards each chunk to local inference server (localhost:8000)
6. Worker runs STOP classifier + teaching moment selection on results
7. Hit stop -- triggers "how was that?" pipeline
8. Worker picks top teaching moment, calls Groq subagent + Anthropic teacher
9. Observation appears in chat

**Cost per session:** $0 inference. ~$0.01-0.02 per observation for LLM calls. ~$0.05-0.10 total for a 30-minute session with 3-5 observations.

### Durable Objects Consideration

Durable Objects run within `wrangler dev` using the miniflare simulator (local mode, the default). The DO-based practice session (`PracticeSession`) manages chunk accumulation and observation delivery. Miniflare DO support is mature; this should work. If it does not, that is a separate bug to fix -- do not use `wrangler dev --remote` since a remote Worker cannot reach `localhost:8000` on your Mac.

### Latency Budget

| Stage | Expected Latency (M4 MPS) |
|-------|--------------------------|
| Audio preprocessing (WebM decode + resample) | ~200ms |
| MuQ embedding extraction (4 folds) | ~2-3s |
| AMT transcription | ~1-2s |
| HTTP overhead | ~10ms |
| **Total per chunk** | **~3-5s** |

For the on-demand "how was that?" flow, chunks are processed as they arrive during recording. By the time the user hits stop, most chunks are already processed. The final chunk plus teaching moment selection + LLM calls add ~3-5s total wait time. Well within acceptable latency.

---

## Files Changed

| File | Change | Type |
|------|--------|------|
| `apps/inference/local_server.py` | New ~100-line FastAPI server with inline script metadata for deps | New file |
| `apps/api/.dev.vars` | Add `HF_INFERENCE_ENDPOINT=http://localhost:8000` | Edit |

No Rust code changes. No web app changes. No production config changes.

---

## Testing

1. Start local server, send a test audio file via curl, verify response shape matches HF endpoint output
2. Start full 3-terminal setup, record a short practice session, verify observations appear in chat
3. Verify production still works by removing the `.dev.vars` override and confirming the Worker calls HF

---

## Future: Scaling to Beta Testers

When ready to invite 10-20 beta testers, flip `.dev.vars` back to the HF endpoint URL (or deploy to production). The local server is a dev-only tool. For beta, consider:

- HF endpoint with scale-to-zero (pay per inference, ~$1-2/hr when active)
- Batching chunks to reduce cold-start impact
- Setting session limits to cap per-user cost

This design does not address beta scaling -- it is purely for single-user self-testing.
