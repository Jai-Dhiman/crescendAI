"""ONNX Runtime inference server for Aria-AMT piano transcription.

Runs inside a Cloudflare Container (linux/amd64, 4 vCPU, 12 GiB RAM).
No PyTorch -- uses ONNX Runtime for CPU inference.

Accepts JSON with base64-encoded WebM/Opus audio, returns MIDI notes and pedal
events matching the Rust AmtResponse struct exactly.

Environment variables:
    MODEL_DIR: path to ONNX models and export_manifest.json (default: /app/models)
    PORT: server port (default: 8080)
    ONNX_INTER_THREADS: inter-op parallelism (default: 2)
    ONNX_INTRA_THREADS: intra-op parallelism (default: 4)
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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# aria-amt tokenizer (pure Python, no PyTorch dependency)
#
# When both aria-amt and aria packages are installed, ariautils can overwrite
# aria-amt's config. Patch load_config before importing the tokenizer.
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

from amt.tokenizer import AmtTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("amt-server")

SAMPLE_RATE = 16000
CHUNK_LEN_S = 30
MAX_BLOCK_LEN = 4096
FFMPEG_DECODE_TIMEOUT_S = 60

# --- Globals set at startup ---

_encoder_session: ort.InferenceSession | None = None
_decoder_sessions: dict[str, ort.InferenceSession] = {}
_tokenizer: AmtTokenizer | None = None
_export_manifest: dict[str, Any] = {}
_inference_count: int = 0
_start_time: float = 0.0

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ONNX models and tokenizer at server startup."""
    global _encoder_session, _decoder_sessions, _tokenizer, _export_manifest, _start_time

    _start_time = time.time()

    model_dir = os.environ.get("MODEL_DIR", "/app/models")
    logger.info("Loading models from %s", model_dir)

    _encoder_session, _decoder_sessions, _export_manifest = load_models(model_dir)
    _tokenizer = AmtTokenizer()

    logger.info(
        "Server ready (strategy=%s, encoder=%s, decoders=%s)",
        _export_manifest["strategy"],
        "loaded",
        list(_decoder_sessions.keys()) or "none (strategy C)",
    )

    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Audio decoding (ffmpeg subprocess, identical to transcription.py)
# ---------------------------------------------------------------------------


def decode_webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus encoded audio bytes to 16kHz mono PCM float32.

    Uses ffmpeg subprocess for robust decoding of WebM containers with
    independent EBML headers (each chunk from MediaRecorder has its own).

    Raises:
        RuntimeError: If ffmpeg decoding fails or produces empty output.
    """
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
# Post-processing (pure Python, copied from transcription.py)
# ---------------------------------------------------------------------------


def midi_dict_to_notes_and_pedals(
    midi_dict: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert ariautils MidiDict to note and pedal event lists.

    Returns:
        Tuple of (notes, pedal_events) in PerfNote/PerfPedalEvent format.
    """
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
        # MidiDict pedal data: 0 = off, 1 = on
        # PerfPedalEvent value: 0 = off, 127 = on (CC64 convention)
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
    """Filter notes/pedals to only those in the current chunk, adjust timestamps.

    Notes with onset >= context_duration are from the current chunk.
    Timestamps are adjusted to be relative to the chunk start.
    """
    if context_duration_s <= 0:
        return notes, pedal_events

    filtered_notes = []
    for note in notes:
        if note["onset"] >= context_duration_s:
            filtered_notes.append({
                "pitch": note["pitch"],
                "onset": round(note["onset"] - context_duration_s, 4),
                "offset": round(note["offset"] - context_duration_s, 4),
                "velocity": note["velocity"],
            })

    filtered_pedals = []
    for pedal in pedal_events:
        if pedal["time"] >= context_duration_s:
            filtered_pedals.append({
                "time": round(pedal["time"] - context_duration_s, 4),
                "value": pedal["value"],
            })

    return filtered_notes, filtered_pedals


# ---------------------------------------------------------------------------
# Log-mel spectrogram (NumPy reimplementation of amt.audio.AudioTransform)
# ---------------------------------------------------------------------------


def _build_mel_filterbank(
    sr: int, n_fft: int, n_mels: int,
) -> np.ndarray:
    """Build a Mel filterbank matrix (n_mels x (n_fft//2 + 1)).

    Reimplements librosa.filters.mel using HTK formula to avoid the
    librosa dependency.
    """
    fmin = 0.0
    fmax = sr / 2.0

    # HTK mel scale
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)

    weights = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]
        for j in range(n_bins):
            f = fft_freqs[j]
            if lower <= f <= center and center > lower:
                weights[i, j] = (f - lower) / (center - lower)
            elif center < f <= upper and upper > center:
                weights[i, j] = (upper - f) / (upper - center)

    # Slaney normalization
    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def _stft_magnitude(audio: np.ndarray, n_fft: int, hop_len: int) -> np.ndarray:
    """Compute STFT magnitude spectrogram using NumPy.

    Returns shape (n_fft//2 + 1, n_frames).
    """
    # Hann window
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    # Pad audio so we get the same number of frames as torch.stft with center=True
    pad_len = n_fft // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, n_fft),
        strides=(audio_padded.strides[0] * hop_len, audio_padded.strides[0]),
    ).copy()

    frames *= window
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    return np.abs(spectrum).T  # (n_fft//2 + 1, n_frames)


def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    """Compute 3-scale log-mel spectrogram matching AudioTransform.log_mel().

    Concatenates large (384 mels), medium (256 mels), and small (128 mels)
    mel spectrograms along the mel axis. Output shape: (1, 768, n_frames).
    """
    hop_len = 160

    configs = [
        (2048, 384),  # large
        (2048, 256),  # medium
        (800, 128),   # small
    ]

    mel_parts = []
    target_frames = None

    for n_fft, n_mels in configs:
        mag = _stft_magnitude(audio, n_fft, hop_len)
        mel_fb = _build_mel_filterbank(SAMPLE_RATE, n_fft, n_mels)
        mel_spec = mel_fb @ mag  # (n_mels, n_frames)

        if target_frames is None:
            target_frames = mel_spec.shape[1]
        elif mel_spec.shape[1] != target_frames:
            # Pad or truncate to match the first spectrogram's frame count
            if mel_spec.shape[1] < target_frames:
                pad_width = target_frames - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
            else:
                mel_spec = mel_spec[:, :target_frames]

        mel_parts.append(mel_spec)

    # Concatenate along mel axis: (768, n_frames)
    combined = np.concatenate(mel_parts, axis=0)

    # Log scale with clamp (matches torch.clamp(min=1e-5).log())
    combined = np.log(np.maximum(combined, 1e-5))

    # Add batch dimension: (1, 768, n_frames)
    return combined[np.newaxis, :, :].astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX model loading
# ---------------------------------------------------------------------------


def _create_session_options() -> ort.SessionOptions:
    """Create ONNX Runtime session options from environment variables."""
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = int(os.environ.get("ONNX_INTER_THREADS", "2"))
    sess_options.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_THREADS", "4"))
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_options


def load_models(model_dir: str) -> tuple[
    ort.InferenceSession,
    dict[str, ort.InferenceSession],
    dict[str, Any],
]:
    """Load ONNX models based on export_manifest.json.

    Returns:
        Tuple of (encoder_session, decoder_sessions_dict, manifest).

    Raises:
        FileNotFoundError: If manifest or model files are missing.
        ValueError: If strategy is unsupported.
    """
    model_path = Path(model_dir)
    manifest_path = model_path / "export_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"export_manifest.json not found in {model_dir}. "
            f"Run the ONNX export script first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    strategy = manifest["strategy"]
    files = manifest["files"]
    sess_options = _create_session_options()

    logger.info("Loading ONNX models (strategy %s) from %s", strategy, model_dir)

    # Encoder is always present
    encoder_path = model_path / files["encoder"]
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder model not found: {encoder_path}")

    encoder_session = ort.InferenceSession(
        str(encoder_path), sess_options, providers=["CPUExecutionProvider"]
    )
    logger.info("Encoder loaded: %s", encoder_path.name)

    decoder_sessions: dict[str, ort.InferenceSession] = {}

    if strategy == "A":
        # KV cache strategy: prefill + step sessions
        for key in ("decoder_prefill", "decoder_step"):
            path = model_path / files[key]
            if not path.exists():
                raise FileNotFoundError(f"Decoder model not found: {path}")
            decoder_sessions[key] = ort.InferenceSession(
                str(path), sess_options, providers=["CPUExecutionProvider"]
            )
            logger.info("Decoder loaded: %s", path.name)

    elif strategy == "B":
        # Stateless decoder: re-processes all tokens each step
        path = model_path / files["decoder_stateless"]
        if not path.exists():
            raise FileNotFoundError(f"Decoder model not found: {path}")
        decoder_sessions["decoder_stateless"] = ort.InferenceSession(
            str(path), sess_options, providers=["CPUExecutionProvider"]
        )
        logger.info("Decoder loaded: %s", path.name)

    elif strategy == "C":
        # Encoder-only ONNX; decoder requires PyTorch.
        # NOTE: This strategy requires torch in the container. The server will
        # raise an error at transcription time if torch is not available.
        logger.warning(
            "Strategy C: encoder-only ONNX. Decoder requires PyTorch in the container."
        )

    else:
        raise ValueError(f"Unknown export strategy: {strategy}")

    return encoder_session, decoder_sessions, manifest


# ---------------------------------------------------------------------------
# Autoregressive decoding (ONNX)
# ---------------------------------------------------------------------------


def _transcribe_strategy_a(
    audio_features: np.ndarray,
    tokenizer: AmtTokenizer,
    decoder_sessions: dict[str, ort.InferenceSession],
    manifest: dict[str, Any],
    audio_len_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Autoregressive decoding with KV cache (Strategy A).

    The prefill session takes the first token and audio features, producing
    initial logits and KV cache state. The step session takes one token at a
    time with the cache as both input and output.
    """
    prefill_session = decoder_sessions["decoder_prefill"]
    step_session = decoder_sessions["decoder_step"]

    n_layers = manifest["n_layers"]
    n_heads = manifest["n_heads"]
    head_dim = manifest["head_dim"]

    bos_id = tokenizer.tok_to_id[tokenizer.bos_tok]
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    pedal_off_id = tokenizer.tok_to_id.get(("pedal", 0))

    # Prefill with BOS token
    input_ids = np.array([[bos_id]], dtype=np.int64)
    x_input_pos = np.array([0], dtype=np.int64)
    xa_input_pos = np.arange(audio_features.shape[1], dtype=np.int64)

    prefill_inputs = {
        "input_ids": input_ids,
        "audio_features": audio_features,
        "x_input_pos": x_input_pos,
        "xa_input_pos": xa_input_pos,
    }

    prefill_outputs = prefill_session.run(None, prefill_inputs)
    # Output order: logits, then self_attn KV pairs, then cross_attn KV pairs
    logits = prefill_outputs[0]

    # Extract KV cache arrays from prefill output
    # Convention: outputs[1..1+2*n_layers] = self-attention K,V per layer
    #             outputs[1+2*n_layers..1+4*n_layers] = cross-attention K,V per layer
    self_kv = {}
    cross_kv = {}
    offset = 1
    for i in range(n_layers):
        self_kv[f"self_k_{i}"] = prefill_outputs[offset + 2 * i]
        self_kv[f"self_v_{i}"] = prefill_outputs[offset + 2 * i + 1]
    offset += 2 * n_layers
    for i in range(n_layers):
        cross_kv[f"cross_k_{i}"] = prefill_outputs[offset + 2 * i]
        cross_kv[f"cross_v_{i}"] = prefill_outputs[offset + 2 * i + 1]

    generated_ids = [bos_id]
    idx = 1

    for _step in range(MAX_BLOCK_LEN - 1):
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Boost pedal-off probability
        if pedal_off_id is not None:
            next_token_logits[pedal_off_id] *= 1.05

        next_token_id = int(np.argmax(next_token_logits))
        generated_ids.append(next_token_id)

        if next_token_id == eos_id:
            break

        # Step with KV cache
        step_inputs = {
            "input_ids": np.array([[next_token_id]], dtype=np.int64),
            "x_input_pos": np.array([idx], dtype=np.int64),
            "audio_features": audio_features,
        }
        # Add KV cache inputs
        step_inputs.update(self_kv)
        step_inputs.update(cross_kv)

        step_outputs = step_session.run(None, step_inputs)
        logits = step_outputs[0]

        # Update self-attention KV cache from step outputs
        offset = 1
        for i in range(n_layers):
            self_kv[f"self_k_{i}"] = step_outputs[offset + 2 * i]
            self_kv[f"self_v_{i}"] = step_outputs[offset + 2 * i + 1]
        # Cross-attention KV cache does not change after prefill

        idx += 1

    return _detokenize(tokenizer, generated_ids, audio_len_samples)


def _transcribe_strategy_b(
    audio_features: np.ndarray,
    tokenizer: AmtTokenizer,
    decoder_sessions: dict[str, ort.InferenceSession],
    audio_len_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Autoregressive decoding without KV cache (Strategy B).

    Each step passes ALL generated tokens so far. Simpler but slower
    (quadratic in sequence length).
    """
    session = decoder_sessions["decoder_stateless"]

    bos_id = tokenizer.tok_to_id[tokenizer.bos_tok]
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    pedal_off_id = tokenizer.tok_to_id.get(("pedal", 0))

    generated_ids = [bos_id]

    for _step in range(MAX_BLOCK_LEN - 1):
        input_ids = np.array([generated_ids], dtype=np.int64)

        outputs = session.run(None, {
            "input_ids": input_ids,
            "audio_features": audio_features,
        })
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]

        # Boost pedal-off probability
        if pedal_off_id is not None:
            next_token_logits[pedal_off_id] *= 1.05

        next_token_id = int(np.argmax(next_token_logits))
        generated_ids.append(next_token_id)

        if next_token_id == eos_id:
            break

    return _detokenize(tokenizer, generated_ids, audio_len_samples)


def _detokenize(
    tokenizer: AmtTokenizer,
    generated_ids: list[int],
    audio_len_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert token IDs to notes and pedal events via MidiDict."""
    decoded_seq = tokenizer.decode(generated_ids)

    # Find last onset time for len_ms parameter
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
# Main transcription entry point
# ---------------------------------------------------------------------------


def transcribe(
    combined_pcm: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ONNX transcription on combined (context + chunk) PCM audio.

    Raises:
        RuntimeError: If models are not loaded or strategy is unsupported.
    """
    if _encoder_session is None or _tokenizer is None:
        raise RuntimeError("Models not loaded")

    strategy = _export_manifest["strategy"]

    # Pad or truncate to 30s
    chunk_samples = CHUNK_LEN_S * SAMPLE_RATE
    if len(combined_pcm) < chunk_samples:
        combined_pcm = np.pad(combined_pcm, (0, chunk_samples - len(combined_pcm)))
    elif len(combined_pcm) > chunk_samples:
        combined_pcm = combined_pcm[:chunk_samples]

    audio_len_samples = len(combined_pcm)

    # Compute log-mel spectrogram: (1, 768, n_frames)
    log_mels = compute_log_mel(combined_pcm)

    # Encode audio
    encoder_outputs = _encoder_session.run(None, {"audio": log_mels})
    audio_features = encoder_outputs[0]  # (1, seq_len, d_model)

    if strategy == "A":
        return _transcribe_strategy_a(
            audio_features, _tokenizer, _decoder_sessions, _export_manifest,
            audio_len_samples,
        )
    elif strategy == "B":
        return _transcribe_strategy_b(
            audio_features, _tokenizer, _decoder_sessions,
            audio_len_samples,
        )
    elif strategy == "C":
        raise RuntimeError(
            "Strategy C requires PyTorch for the decoder. "
            "Install torch in the container or use Strategy A/B."
        )
    else:
        raise RuntimeError(f"Unknown export strategy: {strategy}")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------


@app.post("/transcribe")
async def handle_transcribe(request: Request) -> JSONResponse:
    """Transcribe audio to MIDI notes and pedal events.

    Request body (JSON):
        chunk_audio: base64-encoded WebM/Opus (required)
        context_audio: base64-encoded WebM/Opus or null (optional)

    Response matches the Rust AmtResponse struct.
    """
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

        # Decode chunk audio
        chunk_audio_bytes = base64.b64decode(chunk_audio_b64)
        chunk_pcm = decode_webm_to_pcm(chunk_audio_bytes)
        chunk_duration_s = len(chunk_pcm) / SAMPLE_RATE

        # Decode context audio (optional)
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

        # Transcribe
        midi_notes, pedal_events = transcribe(combined_pcm)

        # Deduplicate: only return notes from the current chunk
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
        "model_loaded": _encoder_session is not None,
        "inference_count": _inference_count,
        "uptime_s": round(time.time() - _start_time, 1),
    })


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
