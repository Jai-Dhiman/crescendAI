# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "pytorch-lightning>=2.0.0",
#     "muq",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "torchaudio>=2.0.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
# ]
# ///
"""Smoke test: MuQ inference on Beethoven_WoO80_var27_8bars_3_15.wav

Usage:
    cd apps/inference && uv run smoke_test.py
    cd apps/inference && uv run smoke_test.py --wav path/to/other.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CRESCEND_DEVICE", "auto")

DEFAULT_WAV = str(Path(__file__).resolve().parent / "Beethoven_WoO80_var27_8bars_3_15.wav")
DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).resolve().parents[2]
    / "model"
    / "data"
    / "checkpoints"
    / "ablation"
    / "optimized_weights"
)


def run_muq(wav_path: str, checkpoint_dir: str) -> None:
    from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
    from models.inference import extract_muq_embeddings, predict_with_ensemble
    from models.loader import get_model_cache
    from preprocessing.audio import preprocess_audio_from_bytes

    print(f"\n[MuQ] Loading model from {checkpoint_dir}...")
    cache = get_model_cache()
    cache.initialize(
        device=os.environ.get("CRESCEND_DEVICE", "auto"),
        checkpoint_dir=Path(checkpoint_dir),
    )
    print(f"[MuQ] Loaded {len(cache.muq_heads)} prediction heads")

    print(f"\n[MuQ] Loading audio: {wav_path}")
    audio_bytes = Path(wav_path).read_bytes()
    audio, duration = preprocess_audio_from_bytes(audio_bytes, max_duration=300)
    print(f"[MuQ] Audio: {duration:.1f}s")

    t0 = time.time()
    embeddings = extract_muq_embeddings(audio, cache)
    predictions = predict_with_ensemble(embeddings, cache)
    elapsed_ms = int((time.time() - t0) * 1000)

    print(f"\n[MuQ] Inference done in {elapsed_ms}ms")
    print(f"[MuQ] Model: {MODEL_INFO['name']} (pairwise={MODEL_INFO['pairwise']:.1%})\n")

    print(f"  {'Dimension':<16} {'Score':>8}")
    print(f"  {'-'*26}")
    for i, dim in enumerate(PERCEPIANO_DIMENSIONS):
        print(f"  {dim:<16} {float(predictions[i]):>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MuQ smoke test")
    parser.add_argument("--wav", default=DEFAULT_WAV, help="Path to audio file")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"ERROR: WAV not found: {args.wav}")
        sys.exit(1)

    if not Path(args.checkpoint_dir).exists():
        print(f"ERROR: Checkpoint dir not found: {args.checkpoint_dir}")
        sys.exit(1)

    run_muq(args.wav, args.checkpoint_dir)


if __name__ == "__main__":
    main()
