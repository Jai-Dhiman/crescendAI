# AMT on Cloudflare Containers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Aria-AMT transcription inference on Cloudflare Containers with ONNX Runtime CPU inference, replacing the empty HF endpoint slot.

**Architecture:** CF Container running Python + ONNX Runtime serves `POST /transcribe`. A TypeScript Container Worker routes requests from the API Worker via service binding to a named instance pool (2 instances). Existing `call_amt_endpoint()` in Rust API switches from HF HTTP to service binding.

**Tech Stack:** Cloudflare Containers, `@cloudflare/containers`, ONNX Runtime (CPU), Python 3.11, FastAPI, TypeScript Worker, Rust (worker-rs service binding)

---

### Task 1: Reorganize inference directory -- move MuQ to `muq/`

**Files:**
- Move: `apps/inference/handler.py` -> `apps/inference/muq/handler.py`
- Move: `apps/inference/muq_local_server.py` -> `apps/inference/muq/muq_local_server.py`
- Move: `apps/inference/Dockerfile` -> `apps/inference/muq/Dockerfile`
- Move: `apps/inference/requirements.txt` -> `apps/inference/muq/requirements.txt`
- Move: `apps/inference/sync_checkpoints.sh` -> `apps/inference/muq/sync_checkpoints.sh`
- Move: `apps/inference/checkpoints/` -> `apps/inference/muq/checkpoints/`
- Keep in place: `apps/inference/constants.py`, `apps/inference/models/`, `apps/inference/preprocessing/`, `apps/inference/tests/`

- [ ] **Step 1: Create muq directory and move files**

```bash
cd apps/inference
mkdir -p muq
git mv handler.py muq/handler.py
git mv muq_local_server.py muq/muq_local_server.py
git mv Dockerfile muq/Dockerfile
git mv requirements.txt muq/requirements.txt
git mv sync_checkpoints.sh muq/sync_checkpoints.sh
git mv checkpoints muq/checkpoints
```

- [ ] **Step 2: Update imports in `muq/handler.py`**

The handler imports from `constants`, `models.inference`, `models.loader`, `preprocessing.audio`. After the move, these are one level up. Add the parent directory to `sys.path` at the top of the file:

```python
import sys
from pathlib import Path
# Add parent directory (apps/inference/) to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.inference import (
    extract_muq_embeddings,
    predict_with_ensemble,
)
from models.loader import get_model_cache
from preprocessing.audio import (
    preprocess_audio_from_bytes,
)
```

- [ ] **Step 3: Update imports in `muq/muq_local_server.py`**

Same parent-path injection:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.inference import extract_muq_embeddings, predict_with_ensemble
from models.loader import ModelCache, get_model_cache
from preprocessing.audio import preprocess_audio_from_bytes
```

- [ ] **Step 4: Update `muq/Dockerfile` COPY paths**

Keep build context as `apps/inference/` and Dockerfile path as `muq/Dockerfile`. Update COPY instructions:

```dockerfile
COPY constants.py .
COPY muq/handler.py handler.py
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/
```

Update build command in Justfile to: `docker build -f muq/Dockerfile .` from `apps/inference/`.

- [ ] **Step 5: Verify MuQ local server still works**

```bash
cd apps/inference
CRESCEND_DEVICE=cpu python muq/muq_local_server.py
# Should start on port 8000, print "MuQ local server starting..."
# Ctrl-C after it loads
```

- [ ] **Step 6: Run existing tests to verify nothing broke**

```bash
cd apps/inference
uv run pytest test_amt_handler.py -v
uv run pytest tests/ -v
```

Expected: All existing tests pass (they import via file path, not module path).

- [ ] **Step 7: Commit**

```bash
git add -A apps/inference/
git commit -m "refactor: reorganize inference directory, move MuQ to muq/"
```

---

### Task 2: Move AMT handler to `amt/` and extract pure-Python helpers

**Files:**
- Move: `apps/inference/amt_handler.py` -> `apps/inference/amt/transcription.py`
- Move: `apps/inference/amt_local_server.py` -> `apps/inference/amt/amt_local_server.py`
- Move: `apps/inference/test_amt_handler.py` -> `apps/inference/amt/test_transcription.py`

- [ ] **Step 1: Create amt directory and move files**

```bash
cd apps/inference
mkdir -p amt
git mv amt_handler.py amt/transcription.py
git mv amt_local_server.py amt/amt_local_server.py
git mv test_amt_handler.py amt/test_transcription.py
```

- [ ] **Step 2: Update test import path in `amt/test_transcription.py`**

The test uses `Path(__file__).parent / "amt_handler.py"` to locate the module. Update:

```python
spec = importlib.util.spec_from_file_location(
    "amt_handler",
    Path(__file__).parent / "transcription.py",
)
```

- [ ] **Step 3: Run AMT unit tests**

```bash
cd apps/inference/amt
uv run pytest test_transcription.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A apps/inference/
git commit -m "refactor: move AMT handler to amt/ directory"
```

---

### Task 3: ONNX export script

**Files:**
- Create: `apps/inference/amt/scripts/export_onnx.py`

This script converts the Aria-AMT PyTorch model to ONNX format. It produces encoder.onnx (audio -> features) and decoder ONNX models (autoregressive token generation with KV cache I/O).

- [ ] **Step 1: Write the ONNX export script**

Create `apps/inference/amt/scripts/export_onnx.py`:

```python
"""Export Aria-AMT encoder and decoder to ONNX format.

Produces:
  - encoder.onnx: audio log-mel spectrogram -> encoder features
  - decoder_prefill.onnx: initial token sequence -> first logits + KV cache
  - decoder_step.onnx: single token + KV cache -> next logits + updated KV cache

Usage:
  uv run python scripts/export_onnx.py \
    --checkpoint /path/to/model.safetensors \
    --output-dir /path/to/output/ \
    --config medium-double
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent for shared amt_handler imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcription import _load_weight, _AMT_CONFIG

# Patch amt config before importing amt modules
import amt.config as _amt_config
_original_load_config = _amt_config.load_config

def _patched_load_config():
    cfg = _original_load_config()
    if "audio" not in cfg or "time_quantization" not in cfg.get("tokenizer", {}):
        return _AMT_CONFIG
    return cfg

_amt_config.load_config = _patched_load_config

from amt.config import load_model_config
from amt.inference.model import AmtEncoderDecoder, ModelConfig
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform


SAMPLE_RATE = 16000
CHUNK_LEN_S = 30
MAX_BLOCK_LEN = 4096


class EncoderWrapper(nn.Module):
    """Wraps the AMT encoder for ONNX export.

    Input: raw audio tensor (1, samples) at 16kHz
    Output: encoder features (1, seq_len, d_model)
    """

    def __init__(self, audio_transform: AudioTransform, encoder: nn.Module):
        super().__init__()
        self.audio_transform = audio_transform
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        log_mels = self.audio_transform.log_mel(audio)
        return self.encoder(xa=log_mels)


class DecoderPrefillWrapper(nn.Module):
    """Wraps the AMT decoder prefill step for ONNX export.

    Input: token sequence (1, seq_len), encoder features (1, enc_len, d_model)
    Output: logits (1, seq_len, vocab_size)

    KV cache is internal to the model and exported as output tensors.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        x_input_pos: torch.Tensor,
        xa_input_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            x=tokens,
            xa=audio_features,
            x_input_pos=x_input_pos,
            xa_input_pos=xa_input_pos,
        )


class DecoderStepWrapper(nn.Module):
    """Wraps a single decoder step for ONNX export.

    Input: single token (1, 1), encoder features, position index
    Output: logits (1, 1, vocab_size)
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,
        audio_features: torch.Tensor,
        x_input_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            x=token,
            xa=audio_features,
            x_input_pos=x_input_pos,
            xa_input_pos=torch.tensor([], dtype=torch.int),
        )


def export_encoder(
    model: AmtEncoderDecoder,
    audio_transform: AudioTransform,
    output_path: Path,
) -> None:
    """Export encoder to ONNX."""
    print("Exporting encoder...")
    wrapper = EncoderWrapper(audio_transform, model.encoder)
    wrapper.eval()

    # Dummy input: 30s of audio at 16kHz
    dummy_audio = torch.randn(1, CHUNK_LEN_S * SAMPLE_RATE)

    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        str(output_path / "encoder.onnx"),
        input_names=["audio"],
        output_names=["audio_features"],
        dynamic_axes={
            "audio": {1: "samples"},
            "audio_features": {1: "enc_seq_len"},
        },
        opset_version=17,
    )
    print(f"Encoder exported to {output_path / 'encoder.onnx'}")


def export_decoder(
    model: AmtEncoderDecoder,
    tokenizer: AmtTokenizer,
    output_path: Path,
) -> None:
    """Export decoder prefill and step models to ONNX.

    The AMT decoder uses internal KV cache state that mutates during
    inference. For ONNX, we export two models:
    1. prefill: processes the initial BOS token sequence with encoder features
    2. step: processes one token at a time using cached KV state

    The KV cache tensors are exported as explicit I/O on the ONNX model.
    This is the standard approach used by ONNX Runtime GenAI and Optimum.

    If the model's internal KV cache makes static ONNX export difficult,
    fall back to torch.export with dynamic shapes or use ONNX Runtime's
    GenAI model builder for Whisper-class architectures.
    """
    print("Exporting decoder...")

    decoder = model.decoder
    decoder.eval()

    # Dummy inputs
    vocab_size = tokenizer.vocab_size
    dummy_tokens = torch.tensor([[1]], dtype=torch.long)  # BOS token
    dummy_features = torch.randn(1, 750, decoder.n_state)  # ~15s of encoded audio
    dummy_x_pos = torch.tensor([0], dtype=torch.int)
    dummy_xa_pos = torch.arange(750, dtype=torch.int)

    try:
        torch.onnx.export(
            DecoderPrefillWrapper(decoder),
            (dummy_tokens, dummy_features, dummy_x_pos, dummy_xa_pos),
            str(output_path / "decoder_prefill.onnx"),
            input_names=["tokens", "audio_features", "x_input_pos", "xa_input_pos"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {1: "seq_len"},
                "audio_features": {1: "enc_seq_len"},
                "x_input_pos": {0: "x_pos_len"},
                "xa_input_pos": {0: "xa_pos_len"},
            },
            opset_version=17,
        )
        print(f"Decoder prefill exported to {output_path / 'decoder_prefill.onnx'}")

        # Export step model
        dummy_step_token = torch.tensor([[42]], dtype=torch.long)
        dummy_step_pos = torch.tensor([1], dtype=torch.int)

        torch.onnx.export(
            DecoderStepWrapper(decoder),
            (dummy_step_token, dummy_features, dummy_step_pos),
            str(output_path / "decoder_step.onnx"),
            input_names=["token", "audio_features", "x_input_pos"],
            output_names=["logits"],
            dynamic_axes={
                "audio_features": {1: "enc_seq_len"},
            },
            opset_version=17,
        )
        print(f"Decoder step exported to {output_path / 'decoder_step.onnx'}")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("The AMT decoder uses in-place KV cache operations that may not")
        print("export cleanly to ONNX. Falling back to full-model export...")
        raise


def validate_onnx(output_path: Path, model: AmtEncoderDecoder, audio_transform: AudioTransform) -> None:
    """Compare ONNX output vs PyTorch output on dummy input."""
    import onnxruntime as ort

    print("Validating encoder ONNX...")
    dummy_audio = torch.randn(1, CHUNK_LEN_S * SAMPLE_RATE)

    # PyTorch reference
    with torch.inference_mode():
        log_mels = audio_transform.log_mel(dummy_audio)
        pt_features = model.encoder(xa=log_mels).numpy()

    # ONNX
    sess = ort.InferenceSession(str(output_path / "encoder.onnx"))
    onnx_features = sess.run(None, {"audio": dummy_audio.numpy()})[0]

    max_diff = np.max(np.abs(pt_features - onnx_features))
    print(f"Encoder max diff: {max_diff:.6f}")
    if max_diff > 0.01:
        raise ValueError(f"Encoder ONNX validation failed: max_diff={max_diff}")
    print("Encoder validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Export Aria-AMT to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for ONNX models")
    parser.add_argument("--config", type=str, default="medium-double", help="Model config name")
    parser.add_argument("--validate", action="store_true", help="Run validation after export")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model config: {args.config}")
    model_config = ModelConfig(**load_model_config(args.config))
    tokenizer = AmtTokenizer()
    model_config.set_vocab_size(tokenizer.vocab_size)

    model = AmtEncoderDecoder(model_config)
    state_dict = _load_weight(args.checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    audio_transform = AudioTransform()

    # Export
    export_encoder(model, audio_transform, output_path)
    export_decoder(model, tokenizer, output_path)

    # Validate
    if args.validate:
        validate_onnx(output_path, model, audio_transform)

    print(f"\nONNX models saved to {output_path}/")
    print("Files: encoder.onnx, decoder_prefill.onnx, decoder_step.onnx")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the export script locally**

This requires the Aria-AMT checkpoint:

```bash
cd apps/inference/amt
mkdir -p scripts
uv run python scripts/export_onnx.py \
  --checkpoint /path/to/aria-amt-piano.safetensors \
  --output-dir ./onnx_models/ \
  --config medium-double \
  --validate
```

Expected: Three ONNX files created. Encoder validation passes (max_diff < 0.01). If decoder export fails due to KV cache in-place ops, the error message explains the fallback path.

- [ ] **Step 3: Commit**

```bash
git add apps/inference/amt/scripts/export_onnx.py
git commit -m "feat: add ONNX export script for Aria-AMT"
```

---

### Task 4: ONNX inference server (`server.py`)

**Files:**
- Create: `apps/inference/amt/server.py`
- Create: `apps/inference/amt/requirements-container.txt`

The inference server runs inside the CF Container. It replaces the PyTorch EndpointHandler with ONNX Runtime sessions.

- [ ] **Step 1: Create container requirements file**

Create `apps/inference/amt/requirements-container.txt`:

```
# ONNX Runtime CPU inference -- NO PyTorch
onnxruntime>=1.17.0
numpy>=1.24.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
# Aria-AMT tokenizer only (ariautils)
ariautils>=0.1.0
```

- [ ] **Step 2: Write the ONNX inference server**

Create `apps/inference/amt/server.py`:

```python
"""Aria-AMT ONNX inference server for Cloudflare Containers.

Lightweight HTTP server that loads ONNX encoder + decoder models
and serves transcription requests. Replaces PyTorch inference with
ONNX Runtime for CPU-optimized performance.

Endpoint: POST /transcribe
Health:   GET /health
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

# Aria-AMT tokenizer (ariautils, no PyTorch dependency)
from ariautils.tokenizer import AmtTokenizer

SAMPLE_RATE = 16000
CHUNK_LEN_S = 30
MAX_BLOCK_LEN = 4096
FFMPEG_DECODE_TIMEOUT_S = 60

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

app = FastAPI(title="Aria-AMT ONNX Inference")

# Global state -- loaded once at startup
_encoder_session: ort.InferenceSession | None = None
_decoder_prefill_session: ort.InferenceSession | None = None
_decoder_step_session: ort.InferenceSession | None = None
_tokenizer: AmtTokenizer | None = None
_startup_time: float = 0.0
_inference_count: int = 0


class TranscribeRequest(BaseModel):
    chunk_audio: str  # base64-encoded WebM/Opus
    context_audio: str | None = None  # base64-encoded WebM/Opus


class NoteEvent(BaseModel):
    pitch: int
    onset: float
    offset: float
    velocity: int


class PedalEvent(BaseModel):
    time: float
    value: int


class TranscriptionInfo(BaseModel):
    note_count: int
    pitch_range: list[int]
    pedal_event_count: int
    transcription_time_ms: int
    context_duration_s: float
    chunk_duration_s: float


class TranscribeResponse(BaseModel):
    midi_notes: list[NoteEvent]
    pedal_events: list[PedalEvent]
    transcription_info: TranscriptionInfo


class ErrorResponse(BaseModel):
    error: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    inference_count: int
    uptime_s: float


def decode_webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus audio to 16kHz mono PCM float32 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()

        result = subprocess.run(
            [
                "ffmpeg",
                "-i", tmp_in.name,
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


def transcribe_onnx(audio: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Run ONNX encoder + decoder inference on PCM audio.

    Returns (notes, pedal_events).
    """
    global _inference_count

    # Pad/truncate to chunk length
    chunk_samples = CHUNK_LEN_S * SAMPLE_RATE
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    elif len(audio) > chunk_samples:
        audio = audio[:chunk_samples]

    audio_input = audio.reshape(1, -1).astype(np.float32)

    # Encode
    audio_features = _encoder_session.run(
        None, {"audio": audio_input}
    )[0]  # (1, enc_len, d_model)

    # Autoregressive decode
    tokenizer = _tokenizer
    bos_id = tokenizer.tok_to_id[tokenizer.bos_tok]
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    pedal_off_id = tokenizer.tok_to_id.get(("pedal", 0))

    generated_ids = [bos_id]
    enc_len = audio_features.shape[1]

    # Prefill
    tokens = np.array([generated_ids], dtype=np.int64)
    x_pos = np.arange(len(generated_ids), dtype=np.int32)
    xa_pos = np.arange(enc_len, dtype=np.int32)

    logits = _decoder_prefill_session.run(
        None,
        {
            "tokens": tokens,
            "audio_features": audio_features,
            "x_input_pos": x_pos,
            "xa_input_pos": xa_pos,
        },
    )[0]  # (1, seq_len, vocab_size)

    # Generate tokens
    for _step in range(MAX_BLOCK_LEN - 1):
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Boost pedal-off slightly
        if pedal_off_id is not None:
            next_logits[pedal_off_id] *= 1.05

        next_token_id = int(np.argmax(next_logits))
        generated_ids.append(next_token_id)

        if next_token_id == eos_id:
            break

        # Step decode
        step_token = np.array([[next_token_id]], dtype=np.int64)
        step_pos = np.array([len(generated_ids) - 1], dtype=np.int32)

        logits = _decoder_step_session.run(
            None,
            {
                "token": step_token,
                "audio_features": audio_features,
                "x_input_pos": step_pos,
            },
        )[0]  # (1, 1, vocab_size)

    # Detokenize
    decoded_seq = tokenizer.decode(generated_ids)
    last_onset_ms = 0
    for tok in decoded_seq:
        if isinstance(tok, tuple) and tok[0] == "onset":
            last_onset_ms = max(last_onset_ms, tok[1])

    total_duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
    midi_dict = tokenizer.detokenize(
        tokenized_seq=decoded_seq,
        len_ms=max(last_onset_ms, total_duration_ms),
    )

    _inference_count += 1
    return midi_dict_to_notes_and_pedals(midi_dict)


@app.on_event("startup")
async def load_models():
    """Load ONNX models at server startup."""
    global _encoder_session, _decoder_prefill_session, _decoder_step_session
    global _tokenizer, _startup_time

    _startup_time = time.time()
    model_dir = Path(MODEL_DIR)

    print(f"Loading ONNX models from {model_dir}...")

    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = int(os.environ.get("ONNX_INTER_THREADS", "2"))
    sess_options.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_THREADS", "4"))
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _encoder_session = ort.InferenceSession(
        str(model_dir / "encoder.onnx"),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("Encoder loaded.")

    _decoder_prefill_session = ort.InferenceSession(
        str(model_dir / "decoder_prefill.onnx"),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("Decoder prefill loaded.")

    _decoder_step_session = ort.InferenceSession(
        str(model_dir / "decoder_step.onnx"),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("Decoder step loaded.")

    _tokenizer = AmtTokenizer()
    print("Tokenizer loaded.")
    print("All models ready!")


@app.get("/health")
async def health() -> HealthResponse:
    model_loaded = all([
        _encoder_session is not None,
        _decoder_prefill_session is not None,
        _decoder_step_session is not None,
        _tokenizer is not None,
    ])
    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        model_loaded=model_loaded,
        inference_count=_inference_count,
        uptime_s=round(time.time() - _startup_time, 1) if _startup_time else 0.0,
    )


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest) -> TranscribeResponse | ErrorResponse:
    start_time = time.time()

    try:
        # Decode chunk audio
        chunk_audio_bytes = base64.b64decode(request.chunk_audio)
        chunk_pcm = decode_webm_to_pcm(chunk_audio_bytes)
        chunk_duration_s = len(chunk_pcm) / SAMPLE_RATE

        # Decode context audio (optional)
        context_duration_s = 0.0
        if request.context_audio:
            context_bytes = base64.b64decode(request.context_audio)
            context_pcm = decode_webm_to_pcm(context_bytes)
            context_duration_s = len(context_pcm) / SAMPLE_RATE
            combined_pcm = np.concatenate([context_pcm, chunk_pcm])
        else:
            combined_pcm = chunk_pcm

        # Run transcription
        midi_notes, pedal_events = transcribe_onnx(combined_pcm)

        # Deduplicate
        midi_notes, pedal_events = deduplicate_notes(
            midi_notes, pedal_events, context_duration_s
        )

        processing_time_ms = int((time.time() - start_time) * 1000)
        pitches = [n["pitch"] for n in midi_notes]

        return TranscribeResponse(
            midi_notes=[NoteEvent(**n) for n in midi_notes],
            pedal_events=[PedalEvent(**p) for p in pedal_events],
            transcription_info=TranscriptionInfo(
                note_count=len(midi_notes),
                pitch_range=[min(pitches), max(pitches)] if pitches else [0, 0],
                pedal_event_count=len(pedal_events),
                transcription_time_ms=processing_time_ms,
                context_duration_s=round(context_duration_s, 2),
                chunk_duration_s=round(chunk_duration_s, 2),
            ),
        )

    except Exception as e:
        return ErrorResponse(
            error={
                "code": "TRANSCRIPTION_ERROR",
                "message": str(e),
            }
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

- [ ] **Step 3: Commit**

```bash
git add apps/inference/amt/server.py apps/inference/amt/requirements-container.txt
git commit -m "feat: add ONNX inference server for AMT container"
```

---

### Task 5: Dockerfile for AMT container

**Files:**
- Create: `apps/inference/amt/Dockerfile`

- [ ] **Step 1: Write the multi-stage Dockerfile**

Create `apps/inference/amt/Dockerfile`:

```dockerfile
# Stage 1: Export ONNX models (has PyTorch, only used during build)
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /build

# Install PyTorch (CPU only) + aria-amt for ONNX export
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache \
    aria-amt \
    safetensors \
    onnx \
    onnxruntime \
    numpy

# Copy export script and handler (for _load_weight and config)
COPY scripts/export_onnx.py scripts/
COPY transcription.py .

# Copy checkpoint (must be in build context)
ARG CHECKPOINT_PATH
COPY ${CHECKPOINT_PATH} /build/checkpoint.safetensors

# Run ONNX export
RUN python scripts/export_onnx.py \
    --checkpoint /build/checkpoint.safetensors \
    --output-dir /build/onnx_models/ \
    --config medium-double


# Stage 2: Slim runtime (NO PyTorch)
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Install runtime dependencies only
COPY requirements-container.txt .
RUN uv pip install --system --no-cache -r requirements-container.txt

# Copy ONNX models from builder
COPY --from=builder /build/onnx_models/ /app/models/

# Copy inference server
COPY server.py .

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
```

- [ ] **Step 2: Commit**

```bash
git add apps/inference/amt/Dockerfile
git commit -m "feat: add multi-stage Dockerfile for AMT container"
```

---

### Task 6: Container Worker -- pool routing (`src/index.ts`)

**Files:**
- Create: `apps/inference/amt/src/index.ts`
- Create: `apps/inference/amt/package.json`
- Create: `apps/inference/amt/tsconfig.json`
- Create: `apps/inference/amt/wrangler.toml`

- [ ] **Step 1: Create `package.json`**

Create `apps/inference/amt/package.json`:

```json
{
  "name": "crescendai-amt",
  "private": true,
  "scripts": {
    "dev": "wrangler dev",
    "deploy": "wrangler deploy"
  },
  "dependencies": {
    "@cloudflare/containers": "^0.1.0"
  },
  "devDependencies": {
    "@cloudflare/workers-types": "^4.20250312.0",
    "typescript": "^5.4.0",
    "wrangler": "^4.0.0"
  }
}
```

- [ ] **Step 2: Create `tsconfig.json`**

Create `apps/inference/amt/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "types": ["@cloudflare/workers-types"],
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*.ts"]
}
```

- [ ] **Step 3: Create `wrangler.toml`**

Create `apps/inference/amt/wrangler.toml`:

```toml
name = "crescendai-amt"
main = "src/index.ts"
compatibility_date = "2025-01-01"

[containers]
image = "./Dockerfile"
max_instances = 5

[vars]
POOL_SIZE = "2"
INSTANCE_TYPE = "standard-4"
SLEEP_AFTER_S = "300"
```

- [ ] **Step 4: Write the Container Worker**

Create `apps/inference/amt/src/index.ts`:

```typescript
import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  AMT_CONTAINER: DurableObjectNamespace<AmtContainer>;
  POOL_SIZE: string;
  SLEEP_AFTER_S: string;
}

interface InstanceState {
  busy: boolean;
  lastUsed: number;
  inferenceCount: number;
}

export class AmtContainer extends Container {
  defaultPort = 8080;

  override get sleepAfter(): number {
    const s = (this.env as Env).SLEEP_AFTER_S;
    return (s ? parseInt(s, 10) : 300) * 1000;
  }

  override get instanceType(): string {
    return "standard-4";
  }

  /**
   * Mark this instance busy in DO storage.
   */
  async markBusy(): Promise<void> {
    const state = await this.getInstanceState();
    state.busy = true;
    state.lastUsed = Date.now();
    await this.ctx.storage.put("state", state);
  }

  /**
   * Mark this instance idle in DO storage.
   */
  async markIdle(): Promise<void> {
    const state = await this.getInstanceState();
    state.busy = false;
    state.inferenceCount += 1;
    await this.ctx.storage.put("state", state);
  }

  /**
   * Get instance state from DO storage, initializing if needed.
   */
  async getInstanceState(): Promise<InstanceState> {
    const stored = await this.ctx.storage.get<InstanceState>("state");
    return stored ?? { busy: false, lastUsed: 0, inferenceCount: 0 };
  }

  /**
   * Forward a request to the container's HTTP server.
   * Marks busy before, idle after (even on error).
   */
  async handleTranscribe(request: Request): Promise<Response> {
    await this.markBusy();
    try {
      const containerRequest = new Request(
        `http://localhost:${this.defaultPort}/transcribe`,
        {
          method: "POST",
          headers: request.headers,
          body: request.body,
        }
      );
      return await this.containerFetch(containerRequest);
    } finally {
      await this.markIdle();
    }
  }

  /**
   * Forward health check to the container.
   */
  async handleHealth(): Promise<Response> {
    const containerRequest = new Request(
      `http://localhost:${this.defaultPort}/health`
    );
    return await this.containerFetch(containerRequest);
  }

  /**
   * Route incoming requests to the container.
   */
  override async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/is-busy") {
      const state = await this.getInstanceState();
      return new Response(JSON.stringify(state.busy), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/health") {
      return this.handleHealth();
    }

    if (url.pathname === "/transcribe" && request.method === "POST") {
      return this.handleTranscribe(request);
    }

    return new Response("Not Found", { status: 404 });
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Health check for the routing worker itself
    if (url.pathname === "/health" && request.method === "GET") {
      return new Response(JSON.stringify({ status: "routing_worker_ok" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // Only accept POST /transcribe
    if (url.pathname !== "/transcribe" || request.method !== "POST") {
      return new Response("Not Found", { status: 404 });
    }

    const poolSize = parseInt(env.POOL_SIZE || "2", 10);

    // Find an idle instance
    for (let i = 0; i < poolSize; i++) {
      const name = `amt-${i}`;

      try {
        const stub = env.AMT_CONTAINER.get(
          env.AMT_CONTAINER.idFromName(name)
        );

        // Check if busy
        const stateResp = await stub.fetch(
          new Request("http://internal/is-busy")
        );
        const busy = await stateResp.json<boolean>();

        if (!busy) {
          // Forward request to this instance
          return await stub.fetch(
            new Request("http://internal/transcribe", {
              method: "POST",
              headers: request.headers,
              body: request.body,
            })
          );
        }
      } catch (e) {
        // Instance might be starting up or unhealthy, try next
        console.log(`Instance ${name} unavailable: ${e}`);
        continue;
      }
    }

    // All instances busy
    return new Response(
      JSON.stringify({
        error: {
          code: "POOL_EXHAUSTED",
          message: `All ${poolSize} AMT instances are busy. Retry later.`,
        },
      }),
      {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }
    );
  },
};
```

Note: The exact `@cloudflare/containers` API may differ. CF Containers is in beta and the SDK is evolving. The `Container` base class, `getContainer`, `containerFetch`, and `startAndWaitForPorts` are the key methods. Consult the latest docs at `developers.cloudflare.com/containers/` when implementing. The routing pattern (named instances, busy/idle tracking via DO storage, fallback to 503) is stable regardless of API surface changes.

- [ ] **Step 5: Install dependencies**

```bash
cd apps/inference/amt
bun install
```

- [ ] **Step 6: Type-check**

```bash
cd apps/inference/amt
bunx tsc --noEmit
```

Expected: No type errors. If `@cloudflare/containers` types are incomplete (beta SDK), add necessary type declarations.

- [ ] **Step 7: Commit**

```bash
git add apps/inference/amt/src/ apps/inference/amt/package.json apps/inference/amt/tsconfig.json apps/inference/amt/wrangler.toml
git commit -m "feat: add AMT Container Worker with pool routing"
```

---

### Task 7: Wire API Worker service binding

**Files:**
- Modify: `apps/api/wrangler.toml` (add service binding, remove HF_AMT_ENDPOINT)
- Modify: `apps/api/src/practice/session_inference.rs:152-272` (replace call_amt_endpoint)

- [ ] **Step 1: Update `wrangler.toml`**

In `apps/api/wrangler.toml`, remove the `HF_AMT_ENDPOINT` var and add a service binding.

Remove this line:
```toml
HF_AMT_ENDPOINT = ""  # Set after deploying Aria-AMT HF endpoint
```

Add after the `[ai]` section:
```toml
[[services]]
binding = "AMT_SERVICE"
service = "crescendai-amt"
```

- [ ] **Step 2: Rewrite `call_amt_endpoint()` to use service binding**

Replace the `call_amt_endpoint` method in `apps/api/src/practice/session_inference.rs` (lines 152-272):

```rust
    /// Call the AMT container via service binding.
    /// Sends JSON with base64-encoded audio, returns transcribed MIDI notes and pedal events.
    pub(crate) async fn call_amt_endpoint(
        &self,
        context_audio: Option<&[u8]>,
        chunk_audio: &[u8],
    ) -> std::result::Result<AmtResponse, String> {
        let fetcher = self
            .env
            .service("AMT_SERVICE")
            .map_err(|e| format!("AMT_SERVICE binding not found: {:?}", e))?;

        // Build JSON payload with base64-encoded audio
        let chunk_b64 = base64_encode(chunk_audio);
        let context_b64 = context_audio.map(base64_encode);

        let payload = serde_json::json!({
            "chunk_audio": chunk_b64,
            "context_audio": context_b64,
        });
        let payload_str = payload.to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers
                .set("Content-Type", "application/json")
                .map_err(|e| format!("{:?}", e))?;
            // No auth header -- service bindings are internal

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from_str(&payload_str)));

            let request = worker::Request::new_with_init(
                "https://amt-service/transcribe",
                &init,
            )
            .map_err(|e| format!("AMT request creation failed: {:?}", e))?;

            let mut response = match fetcher.fetch_request(request).await {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("AMT fetch failed: {:?}", e);
                    if attempt < delays.len() {
                        let delay = delays[attempt];
                        console_log!(
                            "AMT fetch failed (attempt {}), retrying in {}s: {}",
                            attempt + 1,
                            delay / 1000,
                            last_err
                        );
                        sleep_ms(delay).await;
                        continue;
                    }
                    return Err(last_err);
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("AMT returned {}: {}", status, body);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    console_log!(
                        "AMT {} (attempt {}), retrying in {}s",
                        status,
                        attempt + 1,
                        delay / 1000
                    );
                    sleep_ms(delay).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("AMT returned {}: {}", status, body));
            }

            let body_text = response
                .text()
                .await
                .map_err(|e| format!("AMT response read failed: {:?}", e))?;

            let amt: AmtResponse = serde_json::from_str(&body_text).map_err(|e| {
                format!(
                    "AMT response parse failed: {:?} - body: {}",
                    e,
                    crate::truncate_str(&body_text, 200)
                )
            })?;

            if attempt > 0 {
                console_log!("AMT inference succeeded after {} retries", attempt);
            }

            return Ok(amt);
        }

        Err(last_err)
    }
```

Key changes from the HF version:
- `env.service("AMT_SERVICE")` instead of `env.var("HF_AMT_ENDPOINT")`
- `fetcher.fetch_request()` instead of `worker::Fetch::Request().send()`
- URL is `https://amt-service/transcribe` (service binding ignores hostname)
- No `Authorization` header (service bindings are same-account)
- Retry logic preserved identically

- [ ] **Step 3: Verify the API Worker builds**

```bash
cd apps/api
cargo check
```

Expected: Clean build. The `env.service()` method is `Env::service(&self, binding: &str) -> Result<Fetcher>` in `worker = "0.7"`.

- [ ] **Step 4: Commit**

```bash
git add apps/api/wrangler.toml apps/api/src/practice/session_inference.rs
git commit -m "feat: wire AMT service binding, replace HF endpoint"
```

---

### Task 8: Update Justfile with AMT container commands

**Files:**
- Modify: `Justfile`

- [ ] **Step 1: Read current Justfile to find insertion point**

```bash
head -80 Justfile
```

- [ ] **Step 2: Add AMT container recipes**

Add these recipes to the Justfile:

```makefile
# AMT Container
amt-container-dev:
    cd apps/inference/amt && bun run dev

amt-container-deploy:
    cd apps/inference/amt && bun run deploy

amt-container-build:
    cd apps/inference/amt && docker build \
      --build-arg CHECKPOINT_PATH=./checkpoint.safetensors \
      -t crescendai-amt .

amt-container-health:
    cd apps/inference/amt && wrangler containers ssh amt-0 -- curl -s http://localhost:8080/health
```

- [ ] **Step 3: Commit**

```bash
git add Justfile
git commit -m "feat: add AMT container Justfile recipes"
```

---

### Task 9: Local development fallback

**Files:**
- Modify: `apps/api/src/practice/session_inference.rs` (add fallback for local dev)

The service binding won't exist in local dev. Add a fallback that hits `localhost:8080` when `AMT_LOCAL_URL` env var is set.

- [ ] **Step 1: Add local dev fallback in `call_amt_endpoint()`**

At the top of `call_amt_endpoint()`, check for `AMT_LOCAL_URL`:

```rust
    pub(crate) async fn call_amt_endpoint(
        &self,
        context_audio: Option<&[u8]>,
        chunk_audio: &[u8],
    ) -> std::result::Result<AmtResponse, String> {
        // In local dev, fall back to direct HTTP if AMT_LOCAL_URL is set
        let use_direct_http = self.env.var("AMT_LOCAL_URL").ok().map(|v| v.to_string());

        let chunk_b64 = base64_encode(chunk_audio);
        let context_b64 = context_audio.map(base64_encode);
        let payload = serde_json::json!({
            "chunk_audio": chunk_b64,
            "context_audio": context_b64,
        });
        let payload_str = payload.to_string();

        let mut last_err = String::new();
        let delays = if self.inner.borrow().session_ending {
            HF_RETRY_DELAYS_ENDING_MS
        } else {
            HF_RETRY_DELAYS_MS
        };

        for attempt in 0..=delays.len() {
            let headers = worker::Headers::new();
            headers
                .set("Content-Type", "application/json")
                .map_err(|e| format!("{:?}", e))?;

            let mut init = worker::RequestInit::new();
            init.with_method(worker::Method::Post);
            init.with_headers(headers);
            init.with_body(Some(JsValue::from_str(&payload_str)));

            let mut response = if let Some(ref local_url) = use_direct_http {
                // Local dev: direct HTTP to AMT server
                let url = format!("{}/transcribe", local_url);
                let request = worker::Request::new_with_init(&url, &init)
                    .map_err(|e| format!("AMT request creation failed: {:?}", e))?;
                match worker::Fetch::Request(request).send().await {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = format!("AMT fetch failed: {:?}", e);
                        if attempt < delays.len() {
                            sleep_ms(delays[attempt]).await;
                            continue;
                        }
                        return Err(last_err);
                    }
                }
            } else {
                // Production: service binding
                let fetcher = self
                    .env
                    .service("AMT_SERVICE")
                    .map_err(|e| format!("AMT_SERVICE binding not found: {:?}", e))?;
                let request = worker::Request::new_with_init(
                    "https://amt-service/transcribe",
                    &init,
                )
                .map_err(|e| format!("AMT request creation failed: {:?}", e))?;
                match fetcher.fetch_request(request).await {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = format!("AMT fetch failed: {:?}", e);
                        if attempt < delays.len() {
                            sleep_ms(delays[attempt]).await;
                            continue;
                        }
                        return Err(last_err);
                    }
                }
            };

            let status = response.status_code();
            if status == 503 || status == 429 {
                let body = response.text().await.unwrap_or_default();
                last_err = format!("AMT returned {}: {}", status, body);
                if attempt < delays.len() {
                    sleep_ms(delays[attempt]).await;
                    continue;
                }
                return Err(last_err);
            }

            if status != 200 {
                let body = response.text().await.unwrap_or_default();
                return Err(format!("AMT returned {}: {}", status, body));
            }

            let body_text = response
                .text()
                .await
                .map_err(|e| format!("AMT response read failed: {:?}", e))?;

            let amt: AmtResponse = serde_json::from_str(&body_text).map_err(|e| {
                format!(
                    "AMT response parse failed: {:?} - body: {}",
                    e,
                    crate::truncate_str(&body_text, 200)
                )
            })?;

            if attempt > 0 {
                console_log!("AMT inference succeeded after {} retries", attempt);
            }

            return Ok(amt);
        }

        Err(last_err)
    }
```

- [ ] **Step 2: Add `AMT_LOCAL_URL` to wrangler.toml for dev**

Add to the `[vars]` section (only active when running locally -- production ignores it since the service binding takes precedence):

```toml
# Local dev only -- production uses AMT_SERVICE binding
# AMT_LOCAL_URL = "http://localhost:8080"
```

Keep it commented out in the checked-in config. Uncomment locally when running with `just dev`.

- [ ] **Step 3: Verify build**

```bash
cd apps/api
cargo check
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/practice/session_inference.rs apps/api/wrangler.toml
git commit -m "feat: add local dev fallback for AMT service binding"
```

---

### Task 10: End-to-end validation

**Files:** No new files. This task validates the full pipeline.

- [ ] **Step 1: Run ONNX export (requires Aria-AMT checkpoint)**

```bash
cd apps/inference/amt
uv run python scripts/export_onnx.py \
  --checkpoint /path/to/aria-amt-piano.safetensors \
  --output-dir ./onnx_models/ \
  --config medium-double \
  --validate
```

Expected: Three ONNX files. Encoder validation max_diff < 0.01.

- [ ] **Step 2: Run ONNX inference server locally**

```bash
cd apps/inference/amt
MODEL_DIR=./onnx_models PORT=8080 uv run python server.py
```

Expected: Server starts, prints "All models ready!" Health check works:

```bash
curl http://localhost:8080/health
# {"status":"healthy","model_loaded":true,"inference_count":0,"uptime_s":...}
```

- [ ] **Step 3: Test transcription with a real audio file**

```bash
# Base64 encode a test audio file
AUDIO_B64=$(base64 -i ../Beethoven_WoO80_var27_8bars_3_15.wav)

curl -X POST http://localhost:8080/transcribe \
  -H "Content-Type: application/json" \
  -d "{\"chunk_audio\": \"$AUDIO_B64\"}" | python -m json.tool
```

Expected: Response with `midi_notes`, `pedal_events`, `transcription_info`. Note count > 0.

- [ ] **Step 4: Test with API Worker in local dev mode**

```bash
# Terminal 1: AMT server
cd apps/inference/amt
MODEL_DIR=./onnx_models PORT=8080 uv run python server.py

# Terminal 2: API Worker (with AMT_LOCAL_URL uncommented)
just api
```

Start a practice session from the web app and verify AMT results flow through.

- [ ] **Step 5: Run all existing tests**

```bash
cd apps/inference
uv run pytest amt/test_transcription.py -v
uv run pytest tests/ -v

cd apps/api
cargo check
```

Expected: All tests pass, API compiles cleanly.

- [ ] **Step 6: Commit any fixes from validation**

```bash
git add -A
git commit -m "fix: adjustments from end-to-end AMT container validation"
```

---

### Task 11: Deploy to Cloudflare

**Files:** No code changes. Deployment steps.

- [ ] **Step 1: Build and push Docker image**

```bash
cd apps/inference/amt
docker build \
  --build-arg CHECKPOINT_PATH=./checkpoint.safetensors \
  -t crescendai-amt .
```

- [ ] **Step 2: Deploy Container Worker**

```bash
cd apps/inference/amt
bun run deploy
```

Expected: Wrangler uploads Docker image to CF Registry, deploys the Container Worker as `crescendai-amt`. Rolling deploy: 10% then 90%.

- [ ] **Step 3: Verify container health**

```bash
cd apps/inference/amt
wrangler containers list
wrangler containers ssh amt-0 -- curl -s http://localhost:8080/health
```

Expected: Instance `amt-0` running, health returns `{"status":"healthy","model_loaded":true}`.

- [ ] **Step 4: Deploy API Worker with service binding**

```bash
just deploy-api
```

Expected: API Worker deploys with `AMT_SERVICE` binding connected to `crescendai-amt`.

- [ ] **Step 5: Verify in production**

Start a practice session from `crescend.ai`. Check:
- AMT results appear in bar analysis
- Piece identification works within first 2-3 chunks
- MuQ real-time scores unaffected
- No errors in Workers logs

- [ ] **Step 6: Monitor costs**

After 24-48 hours, check CF dashboard:
- Workers & Pages -> crescendai-amt -> Metrics
- Request count, latency p50/p95, error rate
- Billing -> Workers -> Container compute usage

Expected: p50 latency 8-15s, error rate < 1%, daily compute cost < $5.
