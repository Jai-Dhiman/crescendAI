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
#     "piano-transcription-inference",
#     "pretty-midi>=0.2.10",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
# ]
# ///
"""Local inference server for dev testing.

Wraps the same ModelCache + TranscriptionModel used by the HF endpoint
and eval_runner, exposed as a FastAPI server that the Cloudflare Worker
can call instead of the HF inference endpoint.

Usage:
    cd apps/inference && uv run python local_server.py
    cd apps/inference && uv run python local_server.py --port 9000
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

# Set device before any torch imports
os.environ.setdefault("CRESCEND_DEVICE", "auto")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.inference import extract_muq_embeddings, predict_with_ensemble
from models.loader import ModelCache, _resolve_device, get_model_cache
from models.transcription import TranscriptionError, TranscriptionModel
from preprocessing.audio import preprocess_audio_from_bytes

DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).parents[1].parent
    / "model"
    / "data"
    / "checkpoints"
    / "model_improvement"
    / "A1"
)

app = FastAPI()

# Global references set during _init_models() before server starts
_model_cache: ModelCache | None = None
_transcription: TranscriptionModel | None = None


def _init_models(checkpoint_dir: str) -> None:
    """Load MuQ + AMT models into memory."""
    global _model_cache, _transcription

    device = os.environ.get("CRESCEND_DEVICE", "auto")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # MuQ + A1-Max heads
    print("Loading MuQ + A1-Max prediction heads...")
    _model_cache = get_model_cache()
    _model_cache.initialize(device=device, checkpoint_dir=Path(checkpoint_dir))

    if not _model_cache.muq_heads:
        raise RuntimeError(
            f"No prediction heads loaded from {checkpoint_dir}. "
            f"Expected fold_{{0-3}}/ subdirectories with .ckpt files."
        )
    print(f"Loaded {len(_model_cache.muq_heads)} prediction heads")

    # Resolve "auto" to actual device for AMT (PianoTranscription doesn't understand "auto")
    resolved_device = str(_resolve_device(device))

    # AMT (with MPS fallback to CPU)
    print("Loading ByteDance AMT model...")
    try:
        _transcription = TranscriptionModel(device=resolved_device)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            print(f"AMT failed on {resolved_device}, falling back to CPU: {e}")
            _transcription = TranscriptionModel(device="cpu")
        else:
            raise

    print("Models loaded. Server ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": _model_cache is not None}


@app.post("/")
async def inference(request: Request):
    """Run MuQ + AMT inference on raw audio bytes.

    Accepts the same request format the Cloudflare Worker sends to the
    HF endpoint: raw audio bytes in the body with Content-Type audio/*.
    Returns the same JSON response shape as EndpointHandler.__call__.
    """
    if _model_cache is None or _transcription is None:
        return JSONResponse(
            content={"error": {"code": "NOT_READY", "message": "Models not loaded. Start server via __main__."}},
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

        # Preprocess audio (handles WebM/Opus, WAV, etc.)
        audio, duration = preprocess_audio_from_bytes(audio_bytes, max_duration=300)
        print(f"Audio loaded: {duration:.1f}s")

        # MuQ embeddings
        embeddings = extract_muq_embeddings(audio, _model_cache)

        # A1-Max ensemble predictions
        predictions = predict_with_ensemble(embeddings, _model_cache)
        pred_dict = {
            dim: float(predictions[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
        }

        # AMT transcription
        midi_notes = None
        pedal_events = None
        transcription_info = None
        amt_error = None

        try:
            midi_notes, pedal_events = _transcription.transcribe(audio, 24000)
            pitches = [n["pitch"] for n in midi_notes]
            transcription_info = {
                "note_count": len(midi_notes),
                "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
                "pedal_event_count": len(pedal_events),
            }
        except TranscriptionError as e:
            print(f"AMT failed (graceful degradation): {e}")
            amt_error = str(e)

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "predictions": pred_dict,
            "midi_notes": midi_notes,
            "pedal_events": pedal_events,
            "transcription_info": transcription_info,
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

        if amt_error:
            result["amt_error"] = amt_error

        print(f"Inference complete in {processing_time_ms}ms")
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

    parser = argparse.ArgumentParser(description="Local inference server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    _init_models(args.checkpoint_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
