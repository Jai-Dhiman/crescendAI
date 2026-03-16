"""Run inference on skill eval recordings.

Processes all recordings in a piece manifest through the specified
inference config and caches results.

Usage:
    cd apps/evals/
    uv run python -m model.skill_eval.run_inference --config ensemble_4fold --piece fur_elise
    uv run python -m model.skill_eval.run_inference --config single_fold_0 --piece nocturne_op9no2
    uv run python -m model.skill_eval.run_inference --config no_amt --piece fur_elise --subset 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add apps/evals/ to path for paths import, then apps/inference/ for model imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from paths import INFERENCE_DIR, MODEL_DATA

sys.path.insert(0, str(INFERENCE_DIR))

# Set device before torch imports
os.environ.setdefault("CRESCEND_DEVICE", "auto")

import numpy as np
import yaml
from audio_chunker import chunk_audio_file
from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.inference import extract_muq_embeddings, predict_with_ensemble
from models.loader import ModelCache, _resolve_device, get_model_cache
from models.transcription import TranscriptionError, TranscriptionModel

DATA_DIR = MODEL_DATA / "skill_eval"
CHECKPOINT_DIR = MODEL_DATA / "checkpoints" / "model_improvement" / "A1"

MIN_CHUNKS = 2

CONFIGS = {
    "ensemble_4fold": {
        "folds": [0, 1, 2, 3],
        "use_amt": True,
        "description": "Full A1-Max 4-fold ensemble + AMT",
    },
    "single_fold_0": {
        "folds": [0],
        "use_amt": True,
        "description": "Fold 0 only (77.7% pairwise) + AMT",
    },
    "single_fold_best": {
        "folds": [0],  # Fold 0 has highest val pairwise (77.7%)
        "use_amt": True,
        "description": "Best single fold (fold 0, 77.7% pairwise) + AMT",
    },
    "no_amt": {
        "folds": [0, 1, 2, 3],
        "use_amt": False,
        "description": "Full ensemble, skip AMT (latency test only)",
    },
    "cpu_only": {
        "folds": [0, 1, 2, 3],
        "use_amt": True,
        "device_override": "cpu",
        "description": "Full ensemble + AMT on CPU (latency test only)",
    },
}


def reset_model_cache():
    """Reset the ModelCache singleton so we can reinitialize with different config."""
    ModelCache._instance = None


def load_models(config_name: str) -> tuple:
    """Load models for the specified config. Returns (model_cache, transcription_model_or_None)."""
    config = CONFIGS[config_name]
    device = config.get("device_override", os.environ.get("CRESCEND_DEVICE", "auto"))

    reset_model_cache()
    cache = get_model_cache()
    cache.initialize(device=device, checkpoint_dir=CHECKPOINT_DIR)

    # Trim to requested folds
    requested_folds = config["folds"]
    if len(requested_folds) < len(cache.muq_heads):
        cache.muq_heads = [cache.muq_heads[i] for i in requested_folds]
        print(f"  Trimmed to fold(s): {requested_folds} ({len(cache.muq_heads)} heads)")

    # AMT
    transcription = None
    if config["use_amt"]:
        resolved_device = str(_resolve_device(device))
        try:
            transcription = TranscriptionModel(device=resolved_device)
        except RuntimeError as e:
            if "mps" in str(e).lower():
                print(f"  AMT failed on {resolved_device}, falling back to CPU")
                transcription = TranscriptionModel(device="cpu")
            else:
                raise

    return cache, transcription


def run_chunk_inference(
    audio: np.ndarray,
    cache: ModelCache,
    transcription: TranscriptionModel | None,
) -> dict:
    """Run inference on a single audio chunk."""
    start = time.time()

    # MuQ embeddings + predictions
    embeddings = extract_muq_embeddings(audio, cache)
    predictions = predict_with_ensemble(embeddings, cache)
    pred_dict = {
        dim: float(predictions[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)
    }

    # AMT (optional)
    midi_notes = None
    pedal_events = None
    transcription_info = None

    if transcription is not None:
        try:
            midi_notes, pedal_events = transcription.transcribe(audio, 24000)
            pitches = [n["pitch"] for n in midi_notes]
            transcription_info = {
                "note_count": len(midi_notes),
                "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
                "pedal_event_count": len(pedal_events),
            }
        except TranscriptionError as e:
            print(f"    AMT failed: {e}")

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "predictions": pred_dict,
        "midi_notes": midi_notes,
        "pedal_events": pedal_events,
        "transcription_info": transcription_info,
        "processing_time_ms": elapsed_ms,
    }


def process_recording(
    video_id: str,
    audio_path: Path,
    cache: ModelCache,
    transcription: TranscriptionModel | None,
) -> dict | None:
    """Process a single recording: chunk + infer. Returns result dict or None on failure."""
    try:
        chunks = chunk_audio_file(str(audio_path))
    except Exception as e:
        print(f"    Chunking failed: {e}")
        return None

    chunk_results = []
    for ci, chunk_audio in enumerate(chunks):
        try:
            result = run_chunk_inference(chunk_audio, cache, transcription)
            result["chunk_index"] = ci
            result["audio_duration_seconds"] = len(chunk_audio) / 24000
            chunk_results.append(result)
        except Exception as e:
            print(f"    Chunk {ci} failed: {e}")

    if len(chunk_results) < MIN_CHUNKS:
        print(f"    Only {len(chunk_results)} successful chunks (min {MIN_CHUNKS}), skipping")
        return None

    # Aggregate: mean scores across chunks
    mean_scores = {}
    for dim in PERCEPIANO_DIMENSIONS:
        dim_scores = [c["predictions"][dim] for c in chunk_results]
        mean_scores[dim] = float(np.mean(dim_scores))

    mean_time = int(np.mean([c["processing_time_ms"] for c in chunk_results]))

    return {
        "video_id": video_id,
        "chunk_count": len(chunk_results),
        "mean_scores": mean_scores,
        "per_chunk_scores": chunk_results,
        "mean_processing_time_ms": mean_time,
    }


def run(config_name: str, piece_id: str, subset: int | None = None):
    """Run inference for a config on a piece."""
    config = CONFIGS[config_name]
    manifest_path = DATA_DIR / piece_id / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}. Run collect.py first.")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    recordings = manifest.get("recordings", [])
    # Filter to downloaded only
    recordings = [r for r in recordings if r.get("downloaded", False)]

    if subset:
        recordings = recordings[:subset]

    print(f"=== {config['description']} ===")
    print(f"Piece: {piece_id}, Recordings: {len(recordings)}")

    # Check for cached results
    results_dir = DATA_DIR / config_name / piece_id
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"

    existing_results = {}
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        for r in data.get("recordings", []):
            existing_results[r["video_id"]] = r
        print(f"  {len(existing_results)} cached results found")

    # Load models
    print("Loading models...")
    cache, transcription = load_models(config_name)

    # Process recordings
    all_results = []
    audio_dir = DATA_DIR / piece_id / "audio"

    for i, rec in enumerate(recordings):
        video_id = rec["video_id"]

        # Use cached result if available
        if video_id in existing_results:
            all_results.append(existing_results[video_id])
            print(f"  [{i+1}/{len(recordings)}] {video_id} -- cached")
            continue

        audio_path = audio_dir / f"{video_id}.wav"
        if not audio_path.exists():
            print(f"  [{i+1}/{len(recordings)}] {video_id} -- audio not found, skipping")
            continue

        print(f"  [{i+1}/{len(recordings)}] {video_id} -- inferring...")
        result = process_recording(video_id, audio_path, cache, transcription)

        if result:
            result["skill_bucket"] = rec["skill_bucket"]
            all_results.append(result)

            # Save incrementally
            save_results(results_path, config_name, piece_id, all_results)

    print(f"\n  Complete: {len(all_results)}/{len(recordings)} recordings processed")
    save_results(results_path, config_name, piece_id, all_results)


def save_results(path: Path, config_name: str, piece_id: str, results: list[dict]):
    """Save results JSON."""
    data = {
        "config": config_name,
        "piece": piece_id,
        "model_info": {
            "name": MODEL_INFO.get("name", "A1-Max"),
            "config_description": CONFIGS[config_name]["description"],
        },
        "recording_count": len(results),
        "recordings": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run skill eval inference")
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--piece", required=True, choices=["fur_elise", "nocturne_op9no2"])
    parser.add_argument("--subset", type=int, default=None, help="Only process first N recordings")
    args = parser.parse_args()

    run(args.config, args.piece, args.subset)


if __name__ == "__main__":
    main()
