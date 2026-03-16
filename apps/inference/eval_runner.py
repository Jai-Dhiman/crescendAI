"""Local inference batch runner for pipeline evaluation.

Loads EndpointHandler with auto-detected device, processes audio files
in a directory, and writes versioned JSON cache.

Usage:
    CRESCEND_DEVICE=cpu python eval_runner.py --audio-dir ../../data/eval/youtube_amt/
    CRESCEND_DEVICE=mps python eval_runner.py  # uses defaults
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

# Set device before any torch imports (auto = CUDA > MPS > CPU)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

from audio_chunker import chunk_audio_file
from handler import EndpointHandler

DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).parents[1].parent / "model" / "data" / "checkpoints" / "model_improvement" / "A1"
)
DEFAULT_AUDIO_DIR = str(Path(__file__).parents[1].parent / "data" / "eval" / "youtube_amt")
DEFAULT_CACHE_DIR = str(Path(__file__).parents[1].parent / "data" / "eval" / "inference_cache")


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
    """Build a cache directory name from model info.

    Format: {name}_{architecture} matching spec convention (e.g., "a1-max_muq-l9-12").
    """
    name = model_info.get("name", "unknown").lower().replace(" ", "-")
    arch = model_info.get("architecture", "unknown").lower().replace(" ", "-")
    return f"{name}_{arch}"


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

    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device: {os.environ.get('CRESCEND_DEVICE', 'auto')}")

    # Initialize handler
    print("Loading models...")
    handler = EndpointHandler(path=checkpoint_dir)

    # Determine cache directory from model fingerprint
    # Run a tiny inference to get model_info
    test_result = handler({"inputs": audio_files[0].read_bytes(), "parameters": {"max_duration_seconds": 5}})
    if "error" in test_result:
        raise RuntimeError(f"Test inference failed: {test_result['error']}")

    fingerprint = build_model_fingerprint(test_result.get("model_info", {}))
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
            buf = io.BytesIO()
            sf.write(buf, chunk_audio, 24000, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()

            result = handler({"inputs": audio_b64})
            if "error" in result:
                print(f"  chunk {ci} failed: {result['error']}")
                continue

            chunk_results.append({
                "chunk_index": ci,
                "predictions": result.get("predictions", {}),
                "midi_notes": result.get("midi_notes", []),
                "pedal_events": result.get("pedal_events", []),
                "transcription_info": result.get("transcription_info"),
                "audio_duration_seconds": result.get("audio_duration_seconds", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
            })

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
