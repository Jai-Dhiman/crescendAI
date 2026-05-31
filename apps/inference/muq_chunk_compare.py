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
"""Chunk-level MuQ comparison across labeled recordings.

Scores every 15s chunk of each recording (production shape) and reports the
per-dimension median across chunks, so we compare distributions rather than a
single whole-file mean. Usage:

    cd apps/inference && uv run muq_chunk_compare.py \
        --checkpoint-dir ../../model/data/checkpoints/a1_max_sweep/A1max_r32_L7-12_ls0.1 \
        --label beginner_ord1 ../../model/data/results/pipeline_test/k545_beginner_ord1_meowrjlmmjE.wav \
        --label advanced_ord5 ../../model/data/results/pipeline_test/k545_advanced_ord5_tvwDf0Y83eo.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CRESCEND_DEVICE", "auto")

DIMS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]


def score_chunks(wav_path: str, cache) -> np.ndarray:
    from audio_chunker import chunk_audio_file
    from models.inference import extract_muq_embeddings, predict_with_ensemble

    chunks = chunk_audio_file(wav_path, max_duration=600)
    rows = []
    for ch in chunks:
        emb = extract_muq_embeddings(ch, cache)
        preds = predict_with_ensemble(emb, cache)
        rows.append([float(x) for x in preds])
    return np.array(rows)  # shape (n_chunks, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--label",
        action="append",
        nargs=2,
        metavar=("NAME", "WAV"),
        required=True,
        help="A label name and the wav path; repeatable.",
    )
    parser.add_argument("--out-json", help="Optional path to dump per-chunk scores per label as JSON.")
    args = parser.parse_args()

    from models.loader import get_model_cache

    print(f"Loading MuQ from {args.checkpoint_dir} ...")
    cache = get_model_cache()
    cache.initialize(
        device=os.environ.get("CRESCEND_DEVICE", "auto"),
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    results = {}
    for name, wav in args.label:
        if not Path(wav).exists():
            print(f"MISSING: {wav}")
            continue
        scores = score_chunks(wav, cache)
        results[name] = scores
        print(f"\n=== {name} ({scores.shape[0]} chunks) ===")
        print(f"  {'dim':<16}{'median':>9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for i, d in enumerate(DIMS):
            col = scores[:, i]
            print(
                f"  {d:<16}{np.median(col):>9.4f}{col.mean():>9.4f}"
                f"{col.std():>9.4f}{col.min():>9.4f}{col.max():>9.4f}"
            )

    if args.out_json:
        import json
        Path(args.out_json).write_text(
            json.dumps({name: scores.tolist() for name, scores in results.items()}, indent=2)
        )
        print(f"\nWrote per-chunk scores -> {args.out_json}")

    if len(results) == 2:
        (n1, s1), (n2, s2) = results.items()
        m1, m2 = np.median(s1, axis=0), np.median(s2, axis=0)
        print(f"\n=== MEDIAN DELTA: {n2} - {n1} (expect >0 if {n2} more skilled) ===")
        wins = 0
        for i, d in enumerate(DIMS):
            delta = m2[i] - m1[i]
            wins += delta > 0
            print(f"  {d:<16}{delta:>+9.4f}  {'OK' if delta > 0 else 'WRONG'}")
        print(f"\n  Dims where {n2} > {n1}: {wins}/6")


if __name__ == "__main__":
    main()
