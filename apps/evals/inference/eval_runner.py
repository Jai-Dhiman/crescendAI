"""Local inference batch runner for pipeline evaluation.

Directly initializes ModelCache + TranscriptionModel (bypassing EndpointHandler's
HF Inference Endpoint path conventions) to run MuQ + AMT locally.

Usage:
    cd apps/evals/
    CRESCEND_DEVICE=mps uv run python -m inference.eval_runner
    CRESCEND_DEVICE=cpu uv run python -m inference.eval_runner --audio-dir /path/to/audio
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add apps/evals/ to path for paths import, then apps/inference/ for model imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from paths import INFERENCE_DIR, MODEL_DATA

sys.path.insert(0, str(INFERENCE_DIR))

import numpy as np

# Set device before any torch imports (auto = CUDA > MPS > CPU)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

from audio_chunker import chunk_audio_file

# Model imports deferred -- only needed for in-process mode, not --auto-t5 HTTP mode.
_model_imports_loaded = False


def _load_model_imports():
    global _model_imports_loaded
    if _model_imports_loaded:
        return
    global MODEL_INFO, PERCEPIANO_DIMENSIONS
    global extract_muq_embeddings, predict_with_ensemble
    global get_model_cache
    global TranscriptionError, TranscriptionModel
    from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
    from models.inference import extract_muq_embeddings, predict_with_ensemble
    from models.loader import get_model_cache
    from models.transcription import TranscriptionError, TranscriptionModel
    _model_imports_loaded = True

DEFAULT_CHECKPOINT_DIR = str(MODEL_DATA / "checkpoints" / "model_improvement" / "A1")
DEFAULT_AUDIO_DIR = str(MODEL_DATA / "eval" / "youtube_amt")
DEFAULT_CACHE_DIR = str(MODEL_DATA / "eval" / "inference_cache")


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
    pred_dict = {
        dim: float(predictions[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
    }

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
    _load_model_imports()
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = sorted(
        p
        for p in audio_path.iterdir()
        if p.suffix.lower()
        in {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".opus"}
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
            print(f"[{i + 1}/{len(audio_files)}] {recording_id} -- cached, skipping")
            continue

        start = time.time()

        # Chunk audio into 15s segments
        try:
            chunks = chunk_audio_file(str(audio_file))
        except Exception as e:
            print(
                f"[{i + 1}/{len(audio_files)}] {recording_id} -- SKIP (audio error: {e})"
            )
            continue

        # Run inference on each chunk
        chunk_results = []
        for ci, chunk_audio in enumerate(chunks):
            chunk_start = time.time()
            try:
                result = run_inference_on_chunk(chunk_audio, model_cache, transcription)
                chunk_ms = int((time.time() - chunk_start) * 1000)

                chunk_results.append(
                    {
                        "chunk_index": ci,
                        "predictions": result["predictions"],
                        "midi_notes": result.get("midi_notes", []),
                        "pedal_events": result.get("pedal_events", []),
                        "transcription_info": result.get("transcription_info"),
                        "audio_duration_seconds": len(chunk_audio) / 24000,
                        "processing_time_ms": chunk_ms,
                    }
                )
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
            "total_duration_seconds": sum(
                c["audio_duration_seconds"] for c in chunk_results
            ),
            "total_chunks": len(chunk_results),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(
            f"[{i + 1}/{len(audio_files)}] {recording_id} ({elapsed:.1f}s, {len(chunk_results)} chunks)"
        )

    print(f"\nDone. Cache: {versioned_cache}")


# --- HTTP client mode for --auto-t5 ---


def _health_check_servers(muq_url: str, amt_url: str) -> None:
    """Verify both local inference servers are running."""
    import httpx

    for name, url in [("MuQ", muq_url), ("AMT", amt_url)]:
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code != 200:
                raise RuntimeError(f"{name} server at {url} returned {resp.status_code}")
        except httpx.ConnectError:
            raise RuntimeError(
                f"{name} server not running at {url}. "
                f"Start it with: just {'muq' if name == 'MuQ' else 'amt'}"
            )


def _run_http_chunk_inference(
    audio_bytes: bytes,
    muq_url: str,
    amt_url: str,
) -> dict:
    """Run inference on a single audio chunk via HTTP."""
    import httpx

    # MuQ quality scoring
    muq_resp = httpx.post(muq_url, content=audio_bytes, timeout=120.0)
    muq_resp.raise_for_status()
    muq_result = muq_resp.json()

    # AMT transcription
    amt_resp = httpx.post(amt_url, content=audio_bytes, timeout=120.0)
    amt_resp.raise_for_status()
    amt_result = amt_resp.json()

    return {
        "predictions": muq_result.get("predictions", {}),
        "midi_notes": amt_result.get("notes", []),
        "pedal_events": amt_result.get("pedal_events", []),
        "transcription_info": amt_result.get("transcription_info"),
    }


def run_auto_t5(
    cache_dir: str,
    muq_url: str = "http://localhost:8000",
    amt_url: str = "http://localhost:8001",
) -> None:
    """Scan T5 manifests, generate inference cache for uncached recordings via HTTP."""
    from tqdm import tqdm

    T5_PIECES = [
        "bach_prelude_c_wtc1",
        "bach_invention_1",
        "fur_elise",
        "nocturne_op9no2",
    ]
    MANIFEST_BASE = MODEL_DATA / "evals" / "skill_eval"

    _health_check_servers(muq_url, amt_url)
    print("Both servers healthy.")

    fingerprint = "auto-t5_http"
    cache_path = Path(cache_dir) / fingerprint
    cache_path.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in cache_path.glob("*.json")}
    total_cached = 0
    total_skipped = 0

    for piece_id in T5_PIECES:
        manifest_path = MANIFEST_BASE / piece_id / "manifest.yaml"
        if not manifest_path.exists():
            print(f"  {piece_id}: no manifest.yaml, skipping")
            continue

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        all_recs = [r for r in manifest.get("recordings", []) if r.get("downloaded", False)]
        uncached = [r for r in all_recs if r["video_id"] not in existing]

        if not uncached:
            print(f"  {piece_id}: all {len(all_recs)} downloaded recordings cached")
            continue

        print(f"\n  {piece_id}: {len(uncached)} uncached of {len(all_recs)} downloaded")
        audio_dir = MANIFEST_BASE / piece_id / "audio"

        for rec in tqdm(uncached, desc=f"  {piece_id}"):
            video_id = rec["video_id"]
            audio_path = audio_dir / f"{video_id}.wav"

            if not audio_path.exists():
                total_skipped += 1
                continue

            try:
                # Chunk the audio file (reuse existing chunker)
                audio_chunks = chunk_audio_file(str(audio_path))

                import io
                import soundfile as sf

                chunks = []
                for i, chunk_audio in enumerate(audio_chunks):
                    # Convert numpy array to WAV bytes for HTTP
                    buf = io.BytesIO()
                    sf.write(buf, chunk_audio, 24000, format="WAV")
                    wav_bytes = buf.getvalue()

                    result = _run_http_chunk_inference(wav_bytes, muq_url, amt_url)
                    result["chunk_index"] = i
                    result["audio_duration_seconds"] = len(chunk_audio) / 24000.0
                    chunks.append(result)

                # Write cache file
                git_sha, _ = get_git_sha()
                cache_entry = {
                    "recording_id": video_id,
                    "model_fingerprint": fingerprint,
                    "git_sha": git_sha,
                    "chunks": chunks,
                    "total_chunks": len(chunks),
                    "total_duration_seconds": sum(
                        c["audio_duration_seconds"] for c in chunks
                    ),
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }

                out_path = cache_path / f"{video_id}.json"
                out_path.write_text(json.dumps(cache_entry, indent=2) + "\n")
                total_cached += 1

            except Exception as e:
                print(f"\n    {video_id}: inference failed: {e}")
                total_skipped += 1

    print(f"\nDone. Cached: {total_cached}, Skipped: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval inference runner")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--auto-t5",
        action="store_true",
        help="Scan T5 manifests and generate cache via local HTTP servers",
    )
    parser.add_argument("--muq-url", default="http://localhost:8000")
    parser.add_argument("--amt-url", default="http://localhost:8001")
    args = parser.parse_args()

    if args.auto_t5:
        run_auto_t5(args.cache_dir, args.muq_url, args.amt_url)
    else:
        run(args.checkpoint_dir, args.audio_dir, args.cache_dir)
