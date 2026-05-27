"""
Fixture generator for chroma-DTW cargo tests.

Usage:
  uv run python apps/api/src/wasm/score-analysis/tests/fixtures/generate.py

Reads from the project's canonical eval audio and score JSON (same paths used
by apps/inference/score-align-spike/spike.py). Writes three files per fixture
into tests/fixtures/{slug}/:
  audio_chroma.bin   - raw float32 LE, row-major 12 x N
  score_bars.json    - JSON array of ScoreBar objects for the score slice
  expected.json      - bounds for bar_min, bar_max, n_frames, cost

Run once; commit the outputs. Not part of CI.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 441        # 50 Hz at 22050 Hz
FRAME_RATE = SR / HOP   # ~50.0 Hz
DECIM_HZ = 5.0

AUDIO_WAV = Path("model/data/evals/skill_eval/chopin_ballade_1/audio/HlHBUxlcWfk.wav")
SCORE_JSON = Path("model/data/scores/chopin.ballades.1.json")

FIXTURES_DIR = Path("apps/api/src/wasm/score-analysis/tests/fixtures")

CASES = [
    {
        "slug": "ballade1_forward_2min",
        "start_s": 0.0,
        "dur_s": 120.0,
        # bars 1..~30 expected (forward play, 0-2 min)
        "bar_min_lo": 1,
        "bar_min_hi": 5,
        "bar_max_lo": 20,
        "bar_max_hi": 45,
        "cost_hi": 0.30,
    },
    {
        "slug": "ballade1_coldstart_111s",
        "start_s": 111.0,
        "dur_s": 15.0,
        # bars ~30-35 expected (cold-start at 111s, per spike results)
        "bar_min_lo": 25,
        "bar_min_hi": 35,
        "bar_max_lo": 28,
        "bar_max_hi": 40,
        "cost_hi": 0.30,
    },
]


def build_audio_chroma(wav_path: Path, start_s: float, dur_s: float) -> np.ndarray:
    """Load a slice of audio and compute L2-normalized chroma at 50 Hz."""
    y, _ = librosa.load(
        str(wav_path), sr=SR, mono=True,
        offset=start_s, duration=dur_s,
    )
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm
    return chroma  # shape (12, N), float32


def load_score_bars(score_json_path: Path) -> list[dict]:
    data = json.loads(score_json_path.read_text())
    return data["bars"]


def main() -> None:
    import os
    # Allow override via env var for worktree execution; fall back to relative calculation
    env_root = os.environ.get("CRESCEND_ROOT")
    if env_root:
        root = Path(env_root)
    else:
        # From project root: apps/api/src/wasm/score-analysis/tests/fixtures/generate.py
        # parents[0]=fixtures, [1]=tests, [2]=score-analysis, [3]=wasm, [4]=src, [5]=api, [6]=apps, [7]=root
        root = Path(__file__).parents[7]
    wav = root / AUDIO_WAV
    score_json = root / SCORE_JSON

    if not wav.exists():
        print(f"ERROR: audio not found at {wav}", file=sys.stderr)
        sys.exit(1)
    if not score_json.exists():
        print(f"ERROR: score not found at {score_json}", file=sys.stderr)
        sys.exit(1)

    all_bars = load_score_bars(score_json)

    for case in CASES:
        slug = case["slug"]
        out_dir = root / FIXTURES_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {slug} ...")

        # Audio chroma
        chroma = build_audio_chroma(wav, case["start_s"], case["dur_s"])
        n_frames = chroma.shape[1]

        # Write audio_chroma.bin: row-major float32 LE, 12 * n_frames floats
        bin_bytes = struct.pack(f"<{12 * n_frames}f", *chroma.flatten().tolist())
        (out_dir / "audio_chroma.bin").write_bytes(bin_bytes)
        print(f"  audio_chroma.bin: 12 x {n_frames} frames ({len(bin_bytes)} bytes)")

        # score_bars.json: full bars array (Rust reads the whole score for score-chroma build)
        (out_dir / "score_bars.json").write_text(json.dumps(all_bars))
        print(f"  score_bars.json: {len(all_bars)} bars")

        # Compute decimated frame count
        decim_step = int(round(FRAME_RATE / DECIM_HZ))
        decim_n = (n_frames + decim_step - 1) // decim_step

        # expected.json
        expected = {
            "bar_min_lo": case["bar_min_lo"],
            "bar_min_hi": case["bar_min_hi"],
            "bar_max_lo": case["bar_max_lo"],
            "bar_max_hi": case["bar_max_hi"],
            "n_frames": n_frames,
            "decim_n": decim_n,
            "cost_hi": case["cost_hi"],
        }
        (out_dir / "expected.json").write_text(json.dumps(expected, indent=2))
        print(f"  expected.json: {expected}")

    print("Done. Commit the fixtures directory.")


if __name__ == "__main__":
    main()
