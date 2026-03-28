# AMT on Cloudflare Containers

> Status: SPEC
> Date: 2026-03-28

Deploy Aria-AMT transcription inference on Cloudflare Containers instead of HuggingFace Inference Endpoints. CPU inference via ONNX Runtime, integrated with the API Worker via service binding, with a named instance pool for routing.

## Motivation

AMT inference has never been deployed to production (HF_AMT_ENDPOINT is empty). HF dedicated GPU endpoints cost ~$950/month. AMT is a 49M-param Whisper-variant where:

- Latency tolerance is high (15-30s acceptable for beta) -- AMT feeds piece identification and bar analysis, not the real-time scoring path
- The model is small enough for CPU inference with ONNX optimization
- CF Containers offer scale-to-zero billing, cutting costs to ~$55-110/month for beta usage patterns

MuQ stays on HF (GPU required for real-time 15s chunk scoring).

## Architecture

### New: `apps/inference/amt/`

CF Container project containing:

- **Container Worker (`src/index.ts`)** -- TypeScript Worker routing AMT requests to a named instance pool. Manages busy/idle state per instance via DO storage.
- **Docker image (`Dockerfile`)** -- Multi-stage build. Python 3.11 slim + ONNX Runtime (CPU) + ffmpeg + Aria-AMT tokenizer + baked-in ONNX model weights. No PyTorch. ~700MB total.
- **Inference server (`server.py`)** -- Lightweight HTTP server inside the container. Single endpoint: `POST /transcribe`. ONNX Runtime replaces PyTorch for encoder and decoder inference.
- **ONNX export script (`scripts/export_onnx.py`)** -- Converts Aria-AMT encoder and decoder from PyTorch to ONNX format. Run at image build time.
- **`wrangler.toml`** -- Container configuration (instance type, pool size, sleep timeout).
- **`package.json`** -- Worker dependencies (`@cloudflare/containers`).

### Reorganized: `apps/inference/muq/`

Move existing MuQ inference code from `apps/inference/` root:

- `handler.py` (HF endpoint handler)
- `muq_local_server.py` (local dev server)
- MuQ-specific requirements

### Kept: `apps/inference/shared/`

Shared code used by both MuQ and AMT (already partially exists):

- `constants.py` (MODEL_INFO, dimensions)
- `preprocessing/` (audio preprocessing)
- `models/` (model loading utilities)

### Modified: `apps/api/`

- **`wrangler.toml`** -- Add service binding to `crescendai-amt`. Remove `HF_AMT_ENDPOINT` env var.
- **`session_inference.rs`** -- Replace `call_amt_endpoint()` to use service binding instead of external HTTP fetch. Remove HF token auth. Keep retry logic and `AmtResponse` parsing.

## Instance Pool Design

Named instance pool with busy/idle tracking:

```
Container Worker fetch() handler
  Pool: amt-0, amt-1, ..., amt-{POOL_SIZE-1}
  For each request:
    1. Iterate instances, check DO state for first idle
    2. Mark busy, forward POST /transcribe to container
    3. On response (success or error), mark idle
    4. If all busy: return 503 (POOL_EXHAUSTED)
  sleepAfter: 5m (container sleeps when idle, saves cost)
```

Each instance's DO storage tracks:
- `busy: boolean` -- currently processing a request
- `lastUsed: number` -- timestamp of last request
- `inferenceCount: number` -- lifetime inference count (observability)

Pool size controlled by `POOL_SIZE` env var (default: 2 for beta).

## Data Flow

### Request Path

```
PracticeSession DO dispatches inference (futures::join!)
  +---> call_muq_endpoint()  --> HF Endpoint (GPU, ~1-2s)
  +---> call_amt_endpoint()  --> Service Binding --> AMT Container Worker
          --> Pool routing (find idle instance, mark busy)
          --> Forward to container HTTP server
          --> Container: base64 decode -> ffmpeg -> ONNX encode -> ONNX decode -> detokenize -> dedup
          --> Mark idle, return AmtResponse
```

### Timing Estimates

| Step | Estimated |
|---|---|
| Service binding hop | <1ms |
| Pool routing (DO read/write) | <5ms |
| ffmpeg decode (30s WebM to PCM) | 200-500ms |
| ONNX encoder (log-mel + transformer) | 1-3s |
| ONNX decoder (autoregressive tokens) | 5-12s |
| Detokenize + dedup | <50ms |
| **Total** | **~8-15s** |

### Response Contract (unchanged)

```json
{
  "midi_notes": [{"pitch": 60, "onset": 0.5, "offset": 1.2, "velocity": 80}],
  "pedal_events": [{"time": 0.3, "value": 127}],
  "transcription_info": {
    "note_count": 42,
    "pitch_range": [36, 84],
    "pedal_event_count": 5,
    "transcription_time_ms": 9500,
    "context_duration_s": 15.0,
    "chunk_duration_s": 15.0
  }
}
```

No changes to `AmtResponse` Rust struct, `process_amt_result()`, score following, or bar analysis.

## Error Handling

| Failure | Detection | Response |
|---|---|---|
| All instances busy | Pool routing finds no idle instance | Return 503 `POOL_EXHAUSTED`. Existing retry logic in `call_amt_endpoint()` backs off and retries. |
| Container cold start | First request to sleeping instance | `startAndWaitForPorts()` blocks until ready. Timeout: 60s. Caller sees higher latency, not an error. |
| Container OOM | Process killed by CF runtime | Instance restarts automatically. DO state survives. In-flight request fails, retry handles it. |
| ONNX inference error | Python server returns `{"error": {...}}` | Forwarded as-is. Rust API already parses this error shape. |
| ffmpeg decode failure | Corrupt audio | Returns `TRANSCRIPTION_ERROR`. Triggers Tier 3 degradation (MuQ scores only). |
| Instance dies mid-request | Connection reset on fetch | Container Worker catches error, marks instance unhealthy, retries on next instance. |
| Model load failure | Python server never binds port | `startAndWaitForPorts()` times out (60s). Returns 503. |

### Graceful Degradation (already built)

AMT failure triggers Tier 3 in the existing pipeline:
- Score following skips (no MIDI to align)
- Bar analysis falls back to MuQ scores only
- Piece identification defers to next successful chunk
- Practice session continues normally

## Health Check

Python inference server exposes `GET /health`:
- Model loaded status (ONNX sessions initialized)
- Current inference count and uptime
- Memory usage

Container Worker can ping via DO alarms to proactively detect and restart dead instances.

## Docker Image

Multi-stage build for minimal size:

```
Stage 1 (builder):
  - python:3.11-slim
  - Install PyTorch + aria-amt (for ONNX export only)
  - Copy safetensors checkpoint
  - Run export_onnx.py -> encoder.onnx + decoder.onnx

Stage 2 (runtime):
  - python:3.11-slim
  - Install: onnxruntime, numpy, ffmpeg, ariautils (tokenizer)
  - NO PyTorch, NO CUDA
  - Copy ONNX models from builder
  - Copy server.py
  - EXPOSE 8080
```

Estimated image size: ~700MB (vs ~3-4GB with PyTorch).

## Container Configuration

### `apps/inference/amt/wrangler.toml`

```toml
name = "crescendai-amt"
compatibility_date = "2025-01-01"

[containers]
image = "./Dockerfile"
instance_type = "standard-4"    # 4 vCPU, 12 GiB RAM
max_instances = 5
sleepAfter = "5m"

[vars]
POOL_SIZE = "2"
```

### API Worker binding (`apps/api/wrangler.toml`)

```toml
[[services]]
binding = "AMT_SERVICE"
service = "crescendai-amt"
```

## Cost

### CF Container (beta, 2 instances)

| Scenario | Monthly |
|---|---|
| Always warm (24/7) | ~$310 |
| 8 hrs/day active | ~$105 |
| 4 hrs/day active | ~$55 |
| Workers + DO overhead | ~$5-10 |

### Compared to HF

| Option | Monthly |
|---|---|
| HF dedicated T4 | ~$950 |
| CF Container (typical beta) | ~$55-110 |
| **Savings** | **~85-94%** |

## Local Development

Two paths for local AMT development:

1. **`wrangler dev`** -- Runs Container Worker locally with Docker. Full end-to-end with service binding simulation.
2. **Local server fallback** -- Keep `amt_local_server.py` pattern for fast iteration without Docker. API Worker in local dev hits `localhost:8001`.

## Deployment

```
1. docker build -t crescendai-amt apps/inference/amt/
2. wrangler deploy (from apps/inference/amt/)
   - Image pushed to CF Registry (R2-backed)
   - Rolling deploy: 10% canary, then 90%
3. Verify: wrangler containers ssh amt-0
4. just deploy-api (picks up service binding)
```

## ONNX Export Considerations

Aria-AMT has a custom encoder-decoder architecture with KV cache. The ONNX export must handle:

- **Encoder**: Straightforward -- fixed input shape (1, n_mels, time_steps), single forward pass. Export with `torch.onnx.export()` or `torch.export`.
- **Decoder**: Autoregressive with KV cache. Two ONNX models needed:
  - `decoder_prefill.onnx` -- Initial token sequence + cross-attention from encoder output
  - `decoder_step.onnx` -- Single token input with KV cache state as input/output
- **KV cache as ONNX I/O**: Each decoder layer's key/value tensors become explicit inputs and outputs of the step model. This is the standard approach for ONNX autoregressive models (used by Optimum, ONNX Runtime GenAI).
- **Validation**: Compare ONNX output vs PyTorch output on test audio. Max acceptable drift: <0.01 per note onset/offset, exact pitch match.

## Future Considerations

- **ONNX Runtime quantization**: INT8 dynamic quantization could give another 1.5-2x speedup with minimal accuracy loss. Evaluate after initial deployment.
- **CF autoscaling**: When CF ships `autoscale = true`, replace manual pool routing with native scaling. Named instances make this migration straightforward.
- **MuQ on Containers**: If CF adds GPU support, MuQ could also move to Containers for the same cost benefits.
- **Batch inference**: The container pattern works for offline batch jobs (eval harness, T5 labeling) at fraction of HF Jobs cost.
