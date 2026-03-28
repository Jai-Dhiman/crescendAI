# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.115.0",
#     "uvicorn>=0.34.0",
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "pytorch-lightning>=2.0.0",
#     "muq",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "torchaudio>=2.0.0",
#     "torchcodec>=0.1.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
# ]
# ///
"""Local MuQ inference server for dev testing.

Quality scoring only (6 dimensions). No transcription.
Mirrors the production MuQ-only HF endpoint (handler.py).

Usage:
    cd apps/inference && uv run python muq/muq_local_server.py
    cd apps/inference && uv run python muq/muq_local_server.py --port 8000
"""

from __future__ import annotations

import sys
import argparse
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CRESCEND_DEVICE", "auto")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.inference import extract_muq_embeddings, predict_with_ensemble
from models.loader import ModelCache, get_model_cache
from preprocessing.audio import preprocess_audio_from_bytes

DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).parents[1].parent
    / "model"
    / "data"
    / "checkpoints"
    / "ablation"
    / "optimized_weights"
)

app = FastAPI()
_model_cache: ModelCache | None = None


def _init_models(checkpoint_dir: str) -> None:
    global _model_cache

    device = os.environ.get("CRESCEND_DEVICE", "auto")
    print(f"[MuQ] Device: {device}")
    print(f"[MuQ] Checkpoint dir: {checkpoint_dir}")

    print("[MuQ] Loading MuQ + A1-Max prediction heads...")
    _model_cache = get_model_cache()
    _model_cache.initialize(device=device, checkpoint_dir=Path(checkpoint_dir))

    if not _model_cache.muq_heads:
        raise RuntimeError(
            f"No prediction heads loaded from {checkpoint_dir}. "
            f"Expected fold_{{0-3}}/ subdirectories with .ckpt files."
        )
    print(f"[MuQ] Loaded {len(_model_cache.muq_heads)} prediction heads. Ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "muq", "loaded": _model_cache is not None}


@app.post("/")
async def inference(request: Request):
    """Run MuQ quality scoring on raw audio bytes.

    Accepts raw audio bytes (WebM/Opus, WAV, etc.) in the request body.
    Returns 6-dimension quality scores.
    """
    if _model_cache is None:
        return JSONResponse(
            content={"error": {"code": "NOT_READY", "message": "MuQ model not loaded"}},
            status_code=503,
        )

    start_time = time.time()

    try:
        audio_bytes = await request.body()
        if not audio_bytes:
            return JSONResponse(
                content={"error": {"code": "NO_AUDIO", "message": "Empty request body"}},
                status_code=200,
            )

        audio, duration = preprocess_audio_from_bytes(audio_bytes, max_duration=300)
        print(f"[MuQ] Audio: {duration:.1f}s")

        embeddings = extract_muq_embeddings(audio, _model_cache)
        predictions = predict_with_ensemble(embeddings, _model_cache)
        pred_dict = {
            dim: float(predictions[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
        }

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "predictions": pred_dict,
            "model_info": {
                "name": MODEL_INFO["name"],
                "type": MODEL_INFO["type"],
                "pairwise": MODEL_INFO["pairwise"],
                "architecture": MODEL_INFO["architecture"],
                "ensemble_folds": len(_model_cache.muq_heads),
            },
            "audio_duration_seconds": duration,
            "processing_time_ms": processing_time_ms,
        }

        print(f"[MuQ] Done in {processing_time_ms}ms")
        return result

    except Exception as e:
        return JSONResponse(
            content={
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            },
            status_code=200,
        )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Local MuQ inference server (quality scoring)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    _init_models(args.checkpoint_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
