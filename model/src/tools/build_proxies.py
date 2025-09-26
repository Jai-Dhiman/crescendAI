#!/usr/bin/env python3
"""
Build window-level proxy targets from an audio manifest.

Reads a JSONL manifest with at least {"audio_path": "..."} per line, computes
per-window proxies on mel windows [128x128] matching the project’s preprocessing
contract, and writes a JSONL of window targets.

Proxies per window (initial minimal set):
- rms_dr_db: RMS dynamic range in dB (95th - 5th percentile of frame RMS dB)
- onset_density: onsets per second (librosa.onset)
- centroid_mean: spectral centroid mean
- rolloff_mean: spectral rolloff (0.85) mean
- tempo_bpm: estimated tempo in BPM (median aggregator)

Usage:
  uv run python -m src.tools.build_proxies \
    --manifest data/manifests/maestro_train_half.jsonl \
    --out data/proxies/maestro_train_half_windows.jsonl \
    --max-files 200

Explicit exceptions per user preference: invalid manifest or fatal audio parsing
errors abort unless --skip-errors is provided.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import librosa

from src.data.audio_io import load_audio_mono_22050, mel_db_time_major

TARGET_SR = 22050
SEG_LEN = 128
HOP = 512


@dataclass
class Args:
    manifest: Path
    out: Path
    max_files: Optional[int]
    skip_errors: bool


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Build window-level proxy targets from a manifest")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--skip-errors", action="store_true")
    ns = ap.parse_args()
    return Args(manifest=ns.manifest.resolve(), out=ns.out.resolve(), max_files=ns.max_files, skip_errors=ns.skip_errors)


def iter_manifest(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON line in manifest {path}: {e}")


def rms_db(x: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    rms = np.maximum(rms, 1e-10)
    return 20.0 * np.log10(rms)


def proxies_for_window(y: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    # RMS dynamics
    rdb = rms_db(y)
    dyn = float(np.percentile(rdb, 95) - np.percentile(rdb, 5))
    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    centroid_mean = float(np.mean(sc))
    rolloff_mean = float(np.mean(ro))
    # Onsets and tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    dur_sec = float(len(y) / float(sr)) if len(y) > 0 else 0.0
    onset_density = float(len(onsets) / dur_sec) if dur_sec > 0 else 0.0
    tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median))
    return {
        "rms_dr_db": dyn,
        "onset_density": onset_density,
        "centroid_mean": centroid_mean,
        "rolloff_mean": rolloff_mean,
        "tempo_bpm": tempo,
    }


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # iterate items
    items = list(iter_manifest(args.manifest))
    if args.max_files is not None:
        items = items[: max(0, args.max_files)]

    written = 0
    failures: List[str] = []

    with args.out.open("w", encoding="utf-8") as fout:
        for it in items:
            p = Path(it.get("audio_path") or it.get("path") or "")
            if not p.exists():
                if args.skip_errors:
                    continue
                raise FileNotFoundError(f"Audio path missing: {p}")
            try:
                # Load audio and mel once; then slide windows using hop alignment in samples
                y = load_audio_mono_22050(p, target_sr=TARGET_SR)
                if y.size == 0:
                    continue
                # Compute time-major mel to get frame count
                mel_t = mel_db_time_major(y, sr=TARGET_SR, n_fft=2048, hop_length=HOP, n_mels=128)
                T = mel_t.shape[0]
                if T <= 0:
                    continue
                hop_frames = SEG_LEN // 2
                # Walk windows in mel frames, map to waveform slices for proxies
                for t0 in range(0, max(1, T - SEG_LEN + 1), hop_frames):
                    t1 = t0 + SEG_LEN
                    # Map mel frame range to sample indices (approx via hop mapping)
                    s0 = t0 * HOP
                    s1 = min(len(y), t1 * HOP)
                    y_win = y[s0:s1]
                    if y_win.size <= HOP:  # too short, skip
                        continue
                    prx = proxies_for_window(y_win, sr=TARGET_SR)
                    rec = {
                        "audio_path": str(p.resolve()),
                        "start_frame": int(t0),
                        "end_frame": int(min(T, t1)),
                        "start_sample": int(s0),
                        "end_sample": int(s1),
                        "duration_sec": float((s1 - s0) / float(TARGET_SR)),
                        "proxies": prx,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    written += 1
            except Exception as e:
                msg = f"{p}: {type(e).__name__}: {e}"
                if args.skip_errors:
                    failures.append(msg)
                    continue
                else:
                    raise

    print(f"Wrote {written} window proxy records -> {args.out}")
    if failures:
        print(f"Completed with {len(failures)} failures (skip_errors=True)")


if __name__ == "__main__":
    main()
