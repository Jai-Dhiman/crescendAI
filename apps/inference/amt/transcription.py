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


def _import_warm_transcriber() -> Callable[[np.ndarray], tuple[list, list]] | None:
    """Return an in-process transkun PCM->(notes,pedals) callable if transkun is
    importable in THIS env (warm, LOAD-ONCE), else None.

    The 54MB Transkun model is loaded exactly once here (mirroring
    transkun.transcribe.main's checkpoint/conf loading), then reused for every
    request via the returned closure. Per request we write the PCM to a temp WAV
    and feed it through transkun's OWN readAudio + soxr-resample path, so the
    warm path's audio preprocessing is byte-identical to the CLI path
    (transkun_cli.transcribe_pcm) that Task 4 verified end-to-end -- only the
    model load is amortized. No silent failure: any import problem returns None
    so resolve_transcriber falls back to the CLI explicitly.
    """
    try:
        import moduleconf
        import pkg_resources
        import torch
        from transkun.Data import writeMidi
        from transkun.transcribe import readAudio
    except ImportError:
        return None

    device = "cpu"  # MPS is slower for Transkun's semi-CRF ops; force CPU.
    weight = pkg_resources.resource_filename("transkun", "pretrained/2.0.pt")
    conf_path = pkg_resources.resource_filename("transkun", "pretrained/2.0.conf")

    conf_manager = moduleconf.parseFromFile(conf_path)
    transkun_cls = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    checkpoint = torch.load(weight, map_location=device)
    model = transkun_cls(conf=conf).to(device)
    state_key = "best_state_dict" if "best_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[state_key], strict=False)
    model.eval()
    torch.set_grad_enabled(False)

    def _warm(pcm_16k: np.ndarray) -> tuple[list, list]:
        import soundfile as _sf
        pcm = np.ascontiguousarray(np.asarray(pcm_16k, dtype=np.float32))
        if pcm.size == 0:
            raise transkun_cli.TranskunError("warm transcribe received empty PCM")
        with tempfile.TemporaryDirectory() as td:
            in_wav = Path(td) / "in.wav"
            out_mid = Path(td) / "out.mid"
            # Same WAV encoding the CLI path uses, so readAudio sees identical bytes.
            _sf.write(str(in_wav), pcm, transkun_cli.SAMPLE_RATE,
                      format="WAV", subtype="FLOAT")
            fs, audio = readAudio(str(in_wav))
            if fs != model.fs:
                import soxr
                audio = soxr.resample(audio, fs, model.fs)
            x = torch.from_numpy(np.ascontiguousarray(audio)).to(device)
            notes_est = model.transcribe(
                x, stepInSecond=None, segmentSizeInSecond=None,
                discardSecondHalf=False,
            )
            writeMidi(notes_est).write(str(out_mid))
            return transkun_cli.midi_to_notes_and_pedals(out_mid)

    return _warm


def resolve_transcriber() -> Callable[[np.ndarray], tuple[list, list]]:
    """Resolve ONE transcriber at init. Prefer warm in-process transkun; else the
    CLI helper (requires `uv`); else raise so the service refuses to start."""
    warm = _import_warm_transcriber()
    if warm is not None:
        return warm
    if shutil.which("uv") is not None:
        return transkun_cli.transcribe_pcm
    raise RuntimeError(
        "No Transkun path available: transkun is not importable and `uv` is not on PATH."
    )
