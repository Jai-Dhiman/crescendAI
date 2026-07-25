"""Container inference server for Transkun piano transcription.

Runs inside a Cloudflare Container (linux/amd64). Transcription is delegated to
the shared Transkun-backed EndpointHandler (transcription.py). The former ONNX
encoder + PyTorch decoder split -- whose only rationale was aria-amt CPU
throughput -- is gone; Transkun manages its own weights.

Environment variables:
    CHECKPOINT_PATH: retained for call-site compatibility; ignored by Transkun.
    PORT: server port (default: 8080)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcription import (  # noqa: E402
    EndpointHandler,
    build_response,  # noqa: F401  (kept: shared response builder)
    decode_webm_to_pcm,  # noqa: F401  (kept: shared ffmpeg decode)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("amt-server")

# --- Globals set at startup ---
_handler: EndpointHandler | None = None
_inference_count: int = 0
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Resolve the Transkun transcriber once at startup."""
    global _start_time, _handler
    _start_time = time.time()
    _handler = EndpointHandler(path=os.environ.get("CHECKPOINT_PATH", ""))
    logger.info("Transkun server ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def handle_transcribe(request: Request) -> JSONResponse:
    """Transcribe audio to MIDI notes and pedal events via Transkun.

    context_audio is accepted for backward compatibility but ignored (the
    EndpointHandler transcribes the chunk whole-clip).
    """
    global _inference_count

    if _handler is None:
        return JSONResponse(
            content={"error": {"code": "NOT_READY", "message": "Transkun model not loaded"}},
            status_code=503,
        )

    try:
        body = await request.json()
        result = _handler(body)
        if "error" not in result:
            _inference_count += 1
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Transcription failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(
            content={"error": {"code": "TRANSCRIPTION_ERROR", "message": str(e)}},
            status_code=500,
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": _handler is not None,
        "inference_count": _inference_count,
        "uptime_s": round(time.time() - _start_time, 1),
    })


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
