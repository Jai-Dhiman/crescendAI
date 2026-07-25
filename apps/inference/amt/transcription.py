"""HuggingFace Inference Endpoints handler for Transkun piano transcription.

Transkun (MIT, ISMIR 2024) automatic music transcription. Accepts a chunk of
audio (WebM/Opus, base64) and returns MIDI notes and pedal events compatible
with PerfNote/PerfPedalEvent Rust structs. The frozen /transcribe contract is
preserved: `midi_notes`, `pedal_events`, and a `transcription_info` block.

Transcription itself is delegated to the shared `transkun_cli` helper, which
shells out to an isolated `uv run --with transkun` env (Transkun's torch deps
conflict with model/.venv). A `context_audio` field is accepted for backward
compatibility but IGNORED: Transkun is a whole-clip transcriber and the aria
overlapping-window/dedup semantics are gone.

    REQUEST FLOW:
    +------------------+     +------------------+     +------------------+
    | WebM/Opus bytes  | --> | ffmpeg decode    | --> | 16kHz mono PCM   |
    | (base64 encoded) |     | to PCM float32   |     | (chunk only)     |
    +------------------+     +------------------+     +------------------+
                                                             |
                                                             v
                                                      +------------------+
                                                      | Transkun         |
                                                      | (transkun_cli)   |
                                                      +------------------+
                                                             |
                                                             v
                                                      [{pitch, onset,
                                                        offset, velocity}]
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensure transkun_cli importable
import transkun_cli

SAMPLE_RATE = 16000
FFMPEG_DECODE_TIMEOUT_S = 60


def decode_webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus encoded audio bytes to 16kHz mono PCM float32.

    Uses ffmpeg subprocess for robust decoding of WebM containers with
    independent EBML headers (each chunk from MediaRecorder has its own).

    Args:
        audio_bytes: Raw WebM/Opus encoded bytes.

    Returns:
        numpy float32 array of audio samples at 16kHz mono.

    Raises:
        RuntimeError: If ffmpeg decoding fails.
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


def build_response(
    notes: list[dict[str, Any]],
    pedals: list[dict[str, Any]],
    chunk_duration_s: float,
    elapsed_ms: int,
) -> dict[str, Any]:
    """Assemble the frozen /transcribe response shape."""
    pitches = [n["pitch"] for n in notes]
    return {
        "midi_notes": notes,
        "pedal_events": pedals,
        "transcription_info": {
            "note_count": len(notes),
            "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
            "pedal_event_count": len(pedals),
            "transcription_time_ms": int(elapsed_ms),
            "chunk_duration_s": round(float(chunk_duration_s), 2),
        },
    }
