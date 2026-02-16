"""Extract MuQ embeddings for PercePiano segments and save as single .pt file.

Run on Thunder Compute (GPU required):
    cd /workspace/crescendai/model
    python scripts/extract_percepiano_muq.py

Expects:
    data/percepiano_pianoteq_rendered/pianoteq/HB_Steinway_Model_D/*.wav
    data/percepiano_cache/labels.json (for key list)

Produces:
    data/percepiano_cache/muq_embeddings.pt
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from audio_experiments.extractors.muq import MuQExtractor


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    data_dir = MODEL_ROOT / "data"
    audio_dir = data_dir / "percepiano_pianoteq_rendered" / "pianoteq" / "HB_Steinway_Model_D"
    cache_dir = data_dir / "percepiano_cache"
    per_file_cache = cache_dir / "_muq_file_cache"
    output_path = cache_dir / "muq_embeddings.pt"

    with open(cache_dir / "labels.json") as f:
        labels = json.load(f)
    keys = sorted(labels.keys())
    log(f"Extracting MuQ embeddings for {len(keys)} segments")
    log(f"Audio dir: {audio_dir}")
    log(f"Output: {output_path}")

    per_file_cache.mkdir(parents=True, exist_ok=True)
    extractor = MuQExtractor(cache_dir=per_file_cache)
    log(f"Model loaded on {extractor.device}")

    missing = []
    failed = []
    for i, key in enumerate(keys, 1):
        wav_path = audio_dir / f"{key}.wav"
        if not wav_path.exists():
            missing.append(key)
            continue

        cache_path = per_file_cache / f"{key}.pt"
        if cache_path.exists():
            if i % 100 == 0:
                log(f"[{i}/{len(keys)}] cached: {key}")
            continue

        t0 = time.time()
        try:
            extractor.extract_from_file(wav_path)
            dt = time.time() - t0
            log(f"[{i}/{len(keys)}] {key} ({dt:.1f}s)")
        except Exception as e:
            failed.append(key)
            log(f"[{i}/{len(keys)}] FAILED {key}: {e}")

        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if missing:
        log(f"WARNING: {len(missing)} wav files not found")
    if failed:
        log(f"WARNING: {len(failed)} extractions failed")

    # Combine per-file caches into single dict
    log("Combining embeddings into single file...")
    embeddings = {}
    for key in keys:
        pt_path = per_file_cache / f"{key}.pt"
        if pt_path.exists():
            embeddings[key] = torch.load(pt_path, map_location="cpu", weights_only=True)

    log(f"Saving {len(embeddings)} embeddings to {output_path}")
    torch.save(embeddings, output_path)

    sample_key = next(iter(embeddings))
    log(f"Verification: {len(embeddings)} keys, sample shape: {embeddings[sample_key].shape}")
    log("Done.")


if __name__ == "__main__":
    main()
