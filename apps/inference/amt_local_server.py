# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.115.0",
#     "uvicorn>=0.34.0",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "aria",
# ]
# ///
"""Local Aria-AMT inference server for dev testing.

Piano transcription only (MIDI notes + pedal events). No quality scoring.
Mirrors the production Aria-AMT HF endpoint (amt_handler.py).

Accepts JSON with base64-encoded audio fields:
  - chunk_audio (required): current 15s chunk
  - context_audio (optional): previous chunk for 30s overlap

Usage:
    cd apps/inference && uv run python amt_local_server.py
    cd apps/inference && uv run python amt_local_server.py --port 8001
"""

from __future__ import annotations

import argparse
import base64
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CRESCEND_DEVICE", "auto")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
_handler = None


def _init_model(checkpoint_dir: str) -> None:
    global _handler

    print("[AMT] Loading Aria-AMT model...")
    from amt_handler import EndpointHandler

    _handler = EndpointHandler(path=checkpoint_dir)
    print("[AMT] Model loaded. Ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "aria-amt", "loaded": _handler is not None}


@app.post("/")
async def inference(request: Request):
    """Run Aria-AMT transcription.

    Accepts two formats:
    1. JSON with base64 fields: {"chunk_audio": "<b64>", "context_audio": "<b64 or null>"}
       (matches production HF endpoint format)
    2. Raw audio bytes in body (legacy compatibility with old pipeline)
    """
    if _handler is None:
        return JSONResponse(
            content={"error": {"code": "NOT_READY", "message": "AMT model not loaded"}},
            status_code=503,
        )

    start_time = time.time()

    try:
        content_type = request.headers.get("content-type", "")
        body = await request.body()

        if not body:
            return JSONResponse(
                content={"error": {"code": "NO_AUDIO", "message": "Empty request body"}},
                status_code=200,
            )

        # Parse input -- support both JSON (production) and raw bytes (legacy)
        if "json" in content_type or body[:1] == b"{":
            import json

            data = json.loads(body)
            # Decode base64 fields for the handler
            if "chunk_audio" in data and isinstance(data["chunk_audio"], str):
                data["chunk_audio"] = base64.b64decode(data["chunk_audio"])
            if "context_audio" in data and isinstance(data["context_audio"], str):
                data["context_audio"] = base64.b64decode(data["context_audio"])
        else:
            # Raw audio bytes -- treat as chunk_audio with no context
            data = {"chunk_audio": body}

        # Delegate to the production handler
        result = _handler(data)

        processing_time_ms = int((time.time() - start_time) * 1000)
        print(
            f"[AMT] Done in {processing_time_ms}ms "
            f"({result.get('transcription_info', {}).get('note_count', '?')} notes)"
        )

        return result

    except Exception as e:
        return JSONResponse(
            content={
                "error": {
                    "code": "TRANSCRIPTION_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            },
            status_code=200,
        )


if __name__ == "__main__":
    import uvicorn

    # Default: look for aria-amt checkpoint in model weights
    default_checkpoint = str(
        Path(__file__).parents[1].parent / "model" / "data" / "weights" / "aria-amt"
    )

    parser = argparse.ArgumentParser(description="Local Aria-AMT inference server (transcription)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--checkpoint-dir", default=default_checkpoint)
    args = parser.parse_args()

    _init_model(args.checkpoint_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
