#!/usr/bin/env python3
"""
Dataset similarity probe to compare audio domain characteristics.

Computes aggregate stats over a sample of files from a manifest:
- RMS mean/std and dynamic range proxy (95th - 5th percentile of frame RMS in dB)
- Spectral centroid mean/std
- Spectral rolloff (0.85) mean/std
- Onset strength mean and onset density (onsets/sec)
- Tempo estimate (median BPM across files)

Usage:
- uv run python -m src.tools.dataset_probe --manifest data/manifests/maestro.jsonl --out data/reports/maestro_stats.json --max-files 100

Explicit exceptions: invalid manifest lines or fatal audio parsing issues raise; per-file failures are counted and skipped.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import librosa

# Canonical loader for mono 22050
try:
    from src.data.audio_io import load_audio_mono_22050
except Exception as e:
    raise ImportError(
        "Failed to import src.data.audio_io.load_audio_mono_22050. Run `uv sync` to install deps."
    ) from e

TARGET_SR = 22050


@dataclass
class Args:
    manifest: Path
    out: Path
    max_files: int
    seed: int


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Dataset similarity probe from a manifest")
    ap.add_argument("--manifest", required=True, type=Path, help="JSONL manifest with path fields")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON file for aggregate stats")
    ap.add_argument("--max-files", default=100, type=int, help="Max files to analyze")
    ap.add_argument("--seed", default=42, type=int, help="Random seed for sampling")
    ns = ap.parse_args()
    return Args(manifest=ns.manifest.resolve(), out=ns.out.resolve(), max_files=ns.max_files, seed=ns.seed)


def read_manifest_lines(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON line in manifest {path}: {e}")
    return items


def rms_db(x: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length, center=True)
    rms = rms[0]
    # Avoid log of zero
    rms = np.maximum(rms, 1e-10)
    rms_db = 20.0 * np.log10(rms)
    return rms_db


def analyze_file(path: Path) -> Optional[Dict[str, float]]:
    try:
        y = load_audio_mono_22050(path, target_sr=TARGET_SR)
        if y is None or y.size == 0:
            return None
        # RMS and dynamic range proxy
        rdb = rms_db(y)
        dyn_range = float(np.percentile(rdb, 95) - np.percentile(rdb, 5))
        rms_mean = float(np.mean(rdb))
        rms_std = float(np.std(rdb))
        # Spectral features
        sc = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)
        sr_roll = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR, roll_percent=0.85)
        sc_mean = float(np.mean(sc))
        sc_std = float(np.std(sc))
        roll_mean = float(np.mean(sr_roll))
        roll_std = float(np.std(sr_roll))
        # Onsets and tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=TARGET_SR)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=TARGET_SR)
        duration_sec = float(len(y) / float(TARGET_SR))
        onset_density = float(len(onsets) / duration_sec) if duration_sec > 0 else 0.0
        tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=TARGET_SR, aggregate=np.median))
        return {
            "rms_db_mean": rms_mean,
            "rms_db_std": rms_std,
            "dynamic_range_db": dyn_range,
            "spectral_centroid_mean": sc_mean,
            "spectral_centroid_std": sc_std,
            "spectral_rolloff_mean": roll_mean,
            "spectral_rolloff_std": roll_std,
            "onset_density_per_s": onset_density,
            "tempo_bpm": tempo,
        }
    except Exception:
        return None


def aggregate(stats: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not stats:
        return agg
    keys = stats[0].keys()
    for k in keys:
        vals = np.array([s[k] for s in stats if k in s], dtype=float)
        if vals.size == 0:
            continue
        agg[k + "_mean"] = float(np.mean(vals))
        agg[k + "_median"] = float(np.median(vals))
        agg[k + "_std"] = float(np.std(vals))
    return agg


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    items = read_manifest_lines(args.manifest)
    if not items:
        raise RuntimeError("Empty manifest")

    paths = [Path(obj.get("path", "")) for obj in items if obj.get("path")]
    paths = [p for p in paths if p.exists()]

    if not paths:
        raise RuntimeError("No existing files referenced by manifest")

    random.seed(args.seed)
    if len(paths) > args.max_files:
        paths = random.sample(paths, args.max_files)

    results: List[Dict[str, float]] = []
    failures = 0
    for p in paths:
        r = analyze_file(p)
        if r is None:
            failures += 1
        else:
            results.append(r)

    summary = aggregate(results)
    out = {
        "sampled_files": len(paths),
        "successful": len(results),
        "failures": failures,
        "target_sr": TARGET_SR,
        "aggregates": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
