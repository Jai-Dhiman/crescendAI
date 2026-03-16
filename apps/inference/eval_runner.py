"""Local inference batch runner for pipeline evaluation.

Directly initializes ModelCache + TranscriptionModel (bypassing EndpointHandler's
HF Inference Endpoint path conventions) to run MuQ + AMT locally.

Usage:
    CRESCEND_DEVICE=mps python eval_runner.py
    CRESCEND_DEVICE=cpu python eval_runner.py --audio-dir ../../data/eval/youtube_amt/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Set device before any torch imports (auto = CUDA > MPS > CPU)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

from audio_chunker import chunk_audio_file
from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.loader import get_model_cache
from models.inference import extract_muq_embeddings, predict_with_ensemble
from models.transcription import TranscriptionModel, TranscriptionError

DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).parents[1].parent / "model" / "data" / "checkpoints" / "model_improvement" / "A1"
)
DEFAULT_AUDIO_DIR = str(Path(__file__).parents[1].parent / "model" / "data" / "eval" / "youtube_amt")
DEFAULT_CACHE_DIR = str(Path(__file__).parents[1].parent / "model" / "data" / "eval" / "inference_cache")


def get_git_sha() -> tuple[str, bool]:
    """Return (sha, is_dirty)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", True


def build_model_fingerprint(model_info: dict) -> str:
    """Build a cache directory name from model info."""
    name = model_info.get("name", "unknown").lower().replace(" ", "-")
    arch = model_info.get("architecture", "unknown").lower().replace(" ", "-")
    return f"{name}_{arch}"


def run_inference_on_chunk(
    audio: np.ndarray,
    cache,
    transcription: TranscriptionModel,
) -> dict:
    """Run MuQ + AMT on a single audio chunk. Returns result dict."""
    # MuQ embeddings
    embeddings = extract_muq_embeddings(audio, cache)

    # Ensemble predictions
    predictions = predict_with_ensemble(embeddings, cache)
    pred_dict = {dim: float(predictions[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)}

    # AMT transcription
    midi_notes = None
    pedal_events = None
    transcription_info = None
    amt_error = None

    try:
        midi_notes, pedal_events = transcription.transcribe(audio, 24000)
        pitches = [n["pitch"] for n in midi_notes]
        transcription_info = {
            "note_count": len(midi_notes),
            "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
            "pedal_event_count": len(pedal_events),
        }
    except TranscriptionError as e:
        print(f"    AMT failed (graceful degradation): {e}")
        amt_error = str(e)

    result = {
        "predictions": pred_dict,
        "midi_notes": midi_notes,
        "pedal_events": pedal_events,
        "transcription_info": transcription_info,
    }
    if amt_error:
        result["amt_error"] = amt_error

    return result


def run(
    checkpoint_dir: str,
    audio_dir: str,
    cache_dir: str,
) -> None:
    """Run batch inference on all audio files in audio_dir."""
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = sorted(
        p for p in audio_path.iterdir()
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".opus"}
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    device = os.environ.get("CRESCEND_DEVICE", "auto")
    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device: {device}")

    # Initialize models directly (bypass EndpointHandler's HF path conventions)
    print("Loading models...")
    model_cache = get_model_cache()
    model_cache.initialize(device=device, checkpoint_dir=Path(checkpoint_dir))

    if not model_cache.muq_heads:
        raise RuntimeError(
            f"No prediction heads loaded from {checkpoint_dir}. "
            f"Expected fold_{{0-3}}/ subdirectories with .ckpt files."
        )

    print(f"Loaded {len(model_cache.muq_heads)} prediction heads")

    # Initialize AMT (with MPS fallback to CPU)
    try:
        transcription = TranscriptionModel(device=device)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            print(f"AMT failed on {device}, falling back to CPU: {e}")
            transcription = TranscriptionModel(device="cpu")
        else:
            raise

    # Build fingerprint from model info
    fingerprint = build_model_fingerprint(MODEL_INFO)
    versioned_cache = Path(cache_dir) / fingerprint
    versioned_cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {versioned_cache}")

    git_sha, _git_dirty = get_git_sha()

    for i, audio_file in enumerate(audio_files):
        recording_id = audio_file.stem
        cache_file = versioned_cache / f"{recording_id}.json"

        if cache_file.exists():
            print(f"[{i+1}/{len(audio_files)}] {recording_id} -- cached, skipping")
            continue

        start = time.time()

        # Chunk audio into 15s segments
        try:
            chunks = chunk_audio_file(str(audio_file))
        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] {recording_id} -- SKIP (audio error: {e})")
            continue

        # Run inference on each chunk
        chunk_results = []
        for ci, chunk_audio in enumerate(chunks):
            chunk_start = time.time()
            try:
                result = run_inference_on_chunk(chunk_audio, model_cache, transcription)
                chunk_ms = int((time.time() - chunk_start) * 1000)

                chunk_results.append({
                    "chunk_index": ci,
                    "predictions": result["predictions"],
                    "midi_notes": result.get("midi_notes", []),
                    "pedal_events": result.get("pedal_events", []),
                    "transcription_info": result.get("transcription_info"),
                    "audio_duration_seconds": len(chunk_audio) / 24000,
                    "processing_time_ms": chunk_ms,
                })
            except Exception as e:
                print(f"  chunk {ci} failed: {e}")
                continue

        elapsed = time.time() - start

        # Write cache file
        cache_data = {
            "recording_id": recording_id,
            "model_fingerprint": fingerprint,
            "git_sha": git_sha,
            "chunks": chunk_results,
            "total_duration_seconds": sum(c["audio_duration_seconds"] for c in chunk_results),
            "total_chunks": len(chunk_results),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"[{i+1}/{len(audio_files)}] {recording_id} ({elapsed:.1f}s, {len(chunk_results)} chunks)")

    print(f"\nDone. Cache: {versioned_cache}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local inference batch runner")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    run(args.checkpoint_dir, args.audio_dir, args.cache_dir)
