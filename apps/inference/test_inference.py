"""Local test script for D9c AsymmetricGatedFusion inference."""

import time
from pathlib import Path

import numpy as np
import librosa

from constants import MODEL_CONFIG, PERCEPIANO_DIMENSIONS, MODEL_INFO
from models.loader import get_model_cache
from models.inference import (
    extract_mert_embeddings,
    extract_muq_embeddings,
    predict_with_fusion_ensemble,
)


def test_inference(audio_path: str):
    """Run inference on a local audio file."""
    print(f"Testing D9c inference on: {audio_path}")
    print("=" * 60)

    # Load and preprocess audio
    print("\n1. Loading audio...")
    audio, sr = librosa.load(audio_path, sr=MODEL_CONFIG["target_sr"], mono=True)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    # Initialize models
    print("\n2. Initializing models...")
    start = time.time()
    cache = get_model_cache()
    cache.initialize(device="cuda", checkpoint_dir=Path("checkpoints"))
    print(f"   Models loaded in {time.time() - start:.1f}s")
    print(f"   Fusion heads: {len(cache.fusion_heads)}")

    # Extract MERT embeddings
    print("\n3. Extracting MERT embeddings...")
    start = time.time()
    mert_emb = extract_mert_embeddings(audio, cache)
    print(f"   Shape: {mert_emb.shape}, Time: {time.time() - start:.1f}s")

    # Extract MuQ embeddings
    print("\n4. Extracting MuQ embeddings...")
    start = time.time()
    muq_emb = extract_muq_embeddings(audio, cache)
    print(f"   Shape: {muq_emb.shape}, Time: {time.time() - start:.1f}s")

    # Run fusion ensemble
    print("\n5. Running D9c fusion ensemble...")
    start = time.time()
    predictions = predict_with_fusion_ensemble(mert_emb, muq_emb, cache)
    print(f"   Time: {time.time() - start:.3f}s")

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTIONS (19 dimensions)")
    print("=" * 60)
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        bar = "#" * int(predictions[i] * 20)
        print(f"{dim:30s}: {predictions[i]:.3f} |{bar}")

    print("\n" + "=" * 60)
    print(f"Model: {MODEL_INFO['name']}")
    print(f"R2: {MODEL_INFO['r2']}")
    print("=" * 60)

    return predictions


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default test file
        audio_file = "Beethoven_WoO80_var27_8bars_3_15.wav"

    test_inference(audio_file)
