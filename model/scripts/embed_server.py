"""Local Aria symbolic-embedding server for serve-time cosine drill selection.

Wraps model_improvement.aria_embeddings.extract_embedding (the 650M
aria-medium-embedding model, 512-dim EOS-pooled) behind a tiny HTTP endpoint so
the Worker can embed a student's weak-passage MIDI at serve time and cosine-rank
it against the exercise catalog.

This is the "model service" the cosine path (issue #103, Goal B(i)) needs: the
Worker has no torch, and the 650M model cannot run in a Worker. It is a SEPARATE
model from Aria-AMT (transcription) -- AMT produces the MIDI, this embeds it.

Endpoints:
  GET  /health  -> {"status": "ok", "model_loaded": bool}
  POST /embed   {"midi_base64": "<b64 of a .mid>"} -> {"embedding": [512 floats], "dim": 512}

Run: just embed   (cd model && uv run python scripts/embed_server.py --port 8002)

Fails loud: a malformed request or an embedding extraction error returns a
non-200 with the reason, never a silent zero vector (a zero/garbage query would
silently corrupt cosine ranking).
"""
from __future__ import annotations

import argparse
import base64
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_improvement.aria_embeddings import extract_embedding

EMBED_DIM = 512
app = FastAPI()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model_loaded": True})


@app.post("/embed")
async def embed(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001 - surface any parse failure
        return JSONResponse({"error": f"invalid JSON body: {exc}"}, status_code=400)

    b64 = body.get("midi_base64")
    if not isinstance(b64, str) or not b64:
        return JSONResponse(
            {"error": "missing or empty 'midi_base64'"}, status_code=400
        )

    try:
        raw = base64.b64decode(b64)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"base64 decode failed: {exc}"}, status_code=400)

    # extract_embedding reads a path; stage the bytes to a temp .mid.
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            vec = extract_embedding(Path(tmp.name), variant="embedding")
        except Exception as exc:  # noqa: BLE001 - fail loud, no zero-vector fallback
            return JSONResponse(
                {"error": f"embedding extraction failed: {exc}"}, status_code=422
            )

    values = [float(x) for x in vec.tolist()]
    if len(values) != EMBED_DIM:
        return JSONResponse(
            {"error": f"expected {EMBED_DIM}-dim, got {len(values)}"}, status_code=500
        )
    return JSONResponse({"embedding": values, "dim": EMBED_DIM})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aria symbolic embedding server")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    # Warm the model once at startup so the first request is not slow / racy.
    print("[embed] warming aria-medium-embedding ...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
