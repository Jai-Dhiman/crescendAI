"""Hybrid inference server for Aria-AMT piano transcription.

ONNX Runtime encoder (80% of compute) + PyTorch decoder (autoregressive).
Runs inside a Cloudflare Container (linux/amd64, 4 vCPU, 12 GiB RAM).

The encoder is exported to ONNX for graph-optimized CPU inference.
The decoder stays in PyTorch because its KV cache uses in-place ops
that prevent clean ONNX export (PyTorch tracer bakes reshape
dimensions as constants, breaking dynamic sequence lengths).

Environment variables:
    MODEL_DIR: path to ONNX encoder + export_manifest.json (default: /app/models)
    CHECKPOINT_PATH: path to .safetensors checkpoint for PyTorch decoder
    PORT: server port (default: 8080)
    ONNX_INTER_THREADS: inter-op parallelism for encoder (default: 2)
    ONNX_INTRA_THREADS: intra-op parallelism for encoder (default: 4)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -- Patch amt.config before any amt imports --
import amt.config as _amt_config

_AMT_CONFIG = {
    "tokenizer": {
        "velocity_quantization": {"step": 5, "default": 60},
        "time_quantization": {"num_steps": 3000, "step": 10},
    },
    "audio": {
        "sample_rate": 16000,
        "n_fft_large": 2048,
        "n_fft_med": 2048,
        "n_fft_small": 800,
        "hop_len": 160,
        "chunk_len": 30,
        "n_mels_large": 384,
        "n_mels_med": 256,
        "n_mels_small": 128,
    },
    "data": {"stride_factor": 15, "max_seq_len": 4096},
}

_original_load_config = _amt_config.load_config


def _patched_load_config():
    cfg = _original_load_config()
    if "audio" not in cfg or "time_quantization" not in cfg.get("tokenizer", {}):
        return _AMT_CONFIG
    return cfg


_amt_config.load_config = _patched_load_config

from amt.config import load_model_config
from amt.inference.model import AmtEncoderDecoder, ModelConfig, KVCache
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform

from transcription import _load_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("amt-server")

SAMPLE_RATE = 16000
CHUNK_LEN_S = 30
MAX_BLOCK_LEN = 4096
FFMPEG_DECODE_TIMEOUT_S = 60

# --- Globals set at startup ---

_encoder_onnx: ort.InferenceSession | None = None
_decoder: torch.nn.Module | None = None
_audio_transform: AudioTransform | None = None
_tokenizer: AmtTokenizer | None = None
_inference_count: int = 0
_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _setup_kv_cache_for_decoder(decoder: torch.nn.Module) -> None:
    """Reset KV caches and causal mask for a new sequence (CPU only)."""
    decoder.causal_mask = torch.tril(
        torch.ones(MAX_BLOCK_LEN, MAX_BLOCK_LEN, dtype=torch.bool, device="cpu")
    )
    for b in decoder.blocks:
        head_dim = decoder.n_state // decoder.n_head
        b.attn.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_length=MAX_BLOCK_LEN,
            n_heads=decoder.n_head,
            head_dim=head_dim,
            dtype=torch.float32,
        ).to("cpu")
        b.cross_attn.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_length=1500,
            n_heads=decoder.n_head,
            head_dim=head_dim,
            dtype=torch.float32,
        ).to("cpu")


def load_models() -> None:
    """Load ONNX encoder and PyTorch decoder at startup."""
    global _encoder_onnx, _decoder, _audio_transform, _tokenizer

    model_dir = os.environ.get("MODEL_DIR", "/app/models")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoint.safetensors")
    config_name = os.environ.get("AMT_MODEL_CONFIG", "medium-double")

    # Load ONNX encoder
    manifest_path = Path(model_dir) / "export_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        encoder_filename = manifest["files"]["encoder"]["filename"]
    else:
        encoder_filename = "encoder.onnx"

    encoder_path = Path(model_dir) / encoder_filename
    if not encoder_path.exists():
        raise FileNotFoundError(f"ONNX encoder not found: {encoder_path}")

    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = int(os.environ.get("ONNX_INTER_THREADS", "2"))
    sess_options.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_THREADS", "4"))
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _encoder_onnx = ort.InferenceSession(
        str(encoder_path), sess_options, providers=["CPUExecutionProvider"]
    )
    logger.info("ONNX encoder loaded: %s", encoder_path.name)

    # Load full PyTorch model (decoder weights + architecture)
    logger.info("Loading PyTorch decoder (config: %s)", config_name)
    model_config = ModelConfig(**load_model_config(config_name))
    _tokenizer = AmtTokenizer()
    model_config.set_vocab_size(_tokenizer.vocab_size)

    model = AmtEncoderDecoder(model_config)
    state_dict = _load_weight(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    # Keep only the decoder; encoder is replaced by ONNX
    _decoder = model.decoder
    _setup_kv_cache_for_decoder(_decoder)

    # Audio transform for log-mel spectrogram (PyTorch, lightweight)
    _audio_transform = AudioTransform().to("cpu")

    eos_id = _tokenizer.tok_to_id.get(_tokenizer.eos_tok)
    if eos_id is None:
        raise RuntimeError("EOS token not found in tokenizer vocabulary")

    logger.info("Hybrid server ready: ONNX encoder + PyTorch decoder")


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def decode_webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus encoded audio bytes to 16kHz mono PCM float32."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()
        tmp_in_path = tmp_in.name

        result = subprocess.run(
            [
                "ffmpeg",
                "-i", tmp_in_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-v", "error",
                "pipe:1",
            ],
            capture_output=True,
            timeout=FFMPEG_DECODE_TIMEOUT_S,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decoding failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    pcm_data = np.frombuffer(result.stdout, dtype=np.float32)
    if len(pcm_data) == 0:
        raise RuntimeError("ffmpeg produced empty output")

    return pcm_data


# ---------------------------------------------------------------------------
# Post-processing (pure Python, from transcription.py)
# ---------------------------------------------------------------------------


def midi_dict_to_notes_and_pedals(
    midi_dict: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert ariautils MidiDict to note and pedal event lists."""
    notes = []
    for msg in midi_dict.note_msgs:
        data = msg["data"]
        start_ms = midi_dict.tick_to_ms(data["start"])
        end_ms = midi_dict.tick_to_ms(data["end"])
        notes.append({
            "pitch": int(data["pitch"]),
            "onset": round(start_ms / 1000.0, 4),
            "offset": round(end_ms / 1000.0, 4),
            "velocity": int(data["velocity"]),
        })

    pedal_events = []
    for msg in midi_dict.pedal_msgs:
        tick_ms = midi_dict.tick_to_ms(msg["tick"])
        value = 127 if msg["data"] == 1 else 0
        pedal_events.append({
            "time": round(tick_ms / 1000.0, 4),
            "value": value,
        })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    pedal_events.sort(key=lambda e: e["time"])

    return notes, pedal_events


def deduplicate_notes(
    notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
    context_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter to current chunk notes only, adjust timestamps."""
    if context_duration_s <= 0:
        return notes, pedal_events

    filtered_notes = [
        {
            "pitch": n["pitch"],
            "onset": round(n["onset"] - context_duration_s, 4),
            "offset": round(n["offset"] - context_duration_s, 4),
            "velocity": n["velocity"],
        }
        for n in notes
        if n["onset"] >= context_duration_s
    ]

    filtered_pedals = [
        {
            "time": round(p["time"] - context_duration_s, 4),
            "value": p["value"],
        }
        for p in pedal_events
        if p["time"] >= context_duration_s
    ]

    return filtered_notes, filtered_pedals


# ---------------------------------------------------------------------------
# Hybrid transcription: ONNX encoder + PyTorch decoder
# ---------------------------------------------------------------------------


@torch.inference_mode()
def transcribe(combined_pcm: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Transcribe audio using ONNX encoder + PyTorch decoder.

    1. AudioTransform: PCM -> log-mel spectrogram (PyTorch, lightweight)
    2. ONNX encoder: log-mel -> audio features (graph-optimized, ~80% of compute)
    3. PyTorch decoder: autoregressive token generation with KV cache
    4. Detokenize -> MIDI notes + pedal events
    """
    # Pad or truncate to 30s
    chunk_samples = CHUNK_LEN_S * SAMPLE_RATE
    audio_tensor = torch.from_numpy(combined_pcm).float()
    if len(audio_tensor) < chunk_samples:
        audio_tensor = torch.cat([audio_tensor, torch.zeros(chunk_samples - len(audio_tensor))])
    elif len(audio_tensor) > chunk_samples:
        audio_tensor = audio_tensor[:chunk_samples]

    audio_len_samples = len(audio_tensor)

    # Step 1: Log-mel spectrogram via PyTorch AudioTransform (guaranteed correct)
    log_mels = _audio_transform.log_mel(audio_tensor.unsqueeze(0))  # (1, 512, 3000)

    # Step 2: ONNX encoder (graph-optimized CPU inference)
    log_mels_np = log_mels.numpy()
    encoder_output = _encoder_onnx.run(None, {"log_mels": log_mels_np})
    audio_features_np = encoder_output[0]  # (1, 1500, 768)

    # Convert back to torch for PyTorch decoder
    audio_features = torch.from_numpy(audio_features_np)

    # Step 3: PyTorch decoder with KV cache
    tokenizer = _tokenizer
    decoder = _decoder

    prefix = [tokenizer.bos_tok]
    seq = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long)

    # Reset KV cache
    _setup_kv_cache_for_decoder(decoder)

    generated_ids = list(seq[0].tolist())
    idx = seq.shape[1]
    xa_len = audio_features.shape[1]

    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    pedal_off_id = tokenizer.tok_to_id.get(("pedal", 0))

    # Prefill with initial tokens
    logits = decoder(
        x=seq,
        xa=audio_features,
        x_input_pos=torch.arange(0, idx),
        xa_input_pos=torch.arange(0, xa_len),
    )

    for _ in range(MAX_BLOCK_LEN - len(generated_ids)):
        next_token_logits = logits[:, -1, :]

        if pedal_off_id is not None:
            next_token_logits[:, pedal_off_id] *= 1.05

        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        generated_ids.append(next_token_id)
        idx += 1

        if next_token_id == eos_id:
            break

        next_input = torch.tensor([[next_token_id]], dtype=torch.long)
        logits = decoder(
            x=next_input,
            xa=audio_features,
            x_input_pos=torch.tensor([idx - 1], dtype=torch.int),
            xa_input_pos=torch.tensor([], dtype=torch.int),
        )

    # Step 4: Detokenize to MIDI
    decoded_seq = tokenizer.decode(generated_ids)
    last_onset_ms = 0
    for tok in decoded_seq:
        if isinstance(tok, tuple) and tok[0] == "onset":
            last_onset_ms = max(last_onset_ms, tok[1])

    total_duration_ms = int(audio_len_samples / SAMPLE_RATE * 1000)
    midi_dict = tokenizer.detokenize(
        tokenized_seq=decoded_seq,
        len_ms=max(last_onset_ms, total_duration_ms),
    )

    return midi_dict_to_notes_and_pedals(midi_dict)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load models at startup."""
    global _start_time
    _start_time = time.time()
    load_models()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def handle_transcribe(request: Request) -> JSONResponse:
    """Transcribe audio to MIDI notes and pedal events."""
    global _inference_count

    start_time = time.time()

    try:
        body = await request.json()

        chunk_audio_b64 = body.get("chunk_audio")
        if not chunk_audio_b64:
            return JSONResponse(
                content={
                    "error": {
                        "code": "MISSING_CHUNK_AUDIO",
                        "message": "chunk_audio field is required",
                    }
                },
                status_code=400,
            )

        chunk_audio_bytes = base64.b64decode(chunk_audio_b64)
        chunk_pcm = decode_webm_to_pcm(chunk_audio_bytes)
        chunk_duration_s = len(chunk_pcm) / SAMPLE_RATE

        context_audio_b64 = body.get("context_audio")
        context_duration_s = 0.0

        if context_audio_b64:
            context_audio_bytes = base64.b64decode(context_audio_b64)
            context_pcm = decode_webm_to_pcm(context_audio_bytes)
            context_duration_s = len(context_pcm) / SAMPLE_RATE
            combined_pcm = np.concatenate([context_pcm, chunk_pcm])
        else:
            combined_pcm = chunk_pcm

        logger.info(
            "Audio decoded: context=%.1fs, chunk=%.1fs, combined=%.1fs",
            context_duration_s, chunk_duration_s, len(combined_pcm) / SAMPLE_RATE,
        )

        midi_notes, pedal_events = transcribe(combined_pcm)

        midi_notes, pedal_events = deduplicate_notes(
            midi_notes, pedal_events, context_duration_s
        )

        processing_time_ms = int((time.time() - start_time) * 1000)
        pitches = [n["pitch"] for n in midi_notes]

        _inference_count += 1

        result = {
            "midi_notes": midi_notes,
            "pedal_events": pedal_events,
            "transcription_info": {
                "note_count": len(midi_notes),
                "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
                "pedal_event_count": len(pedal_events),
                "transcription_time_ms": processing_time_ms,
                "context_duration_s": round(context_duration_s, 2),
                "chunk_duration_s": round(chunk_duration_s, 2),
            },
        }

        logger.info(
            "Transcription complete: %d notes, %d pedal events in %dms",
            len(midi_notes), len(pedal_events), processing_time_ms,
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Transcription failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(
            content={
                "error": {
                    "code": "TRANSCRIPTION_ERROR",
                    "message": str(e),
                }
            },
            status_code=500,
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": _encoder_onnx is not None and _decoder is not None,
        "inference_count": _inference_count,
        "uptime_s": round(time.time() - _start_time, 1),
    })


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
