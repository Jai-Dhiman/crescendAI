# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
# ]
# ///
"""Smoke test: Aria-AMT transcription on Beethoven_WoO80_var27_8bars_3_15.wav

Runs the full transcription pipeline and prints note/pedal counts.

Usage:
    cd apps/inference && uv run smoke_test_amt.py
    cd apps/inference && uv run smoke_test_amt.py --wav path/to/other.wav
    cd apps/inference && uv run smoke_test_amt.py --checkpoint path/to/weights/
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

# Must insert amt/ so transcription.py (and its patched amt.config) is importable
AMT_DIR = str(Path(__file__).resolve().parent / "amt")
sys.path.insert(0, AMT_DIR)

os.environ.setdefault("CRESCEND_DEVICE", "auto")

DEFAULT_WAV = str(Path(__file__).resolve().parent / "Beethoven_WoO80_var27_8bars_3_15.wav")
DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).resolve().parents[2]
    / "model"
    / "data"
    / "weights"
    / "aria-amt"
)


def run_amt(wav_path: str, checkpoint_dir: str) -> None:
    from transcription import EndpointHandler

    print(f"\n[AMT] Loading model from {checkpoint_dir}...")
    handler = EndpointHandler(path=checkpoint_dir)

    print(f"\n[AMT] Loading audio: {wav_path}")
    audio_bytes = Path(wav_path).read_bytes()
    chunk_b64 = base64.b64encode(audio_bytes).decode()

    print("[AMT] Running transcription (chunk only, no context)...")
    t0 = time.time()
    result = handler({"chunk_audio": chunk_b64, "context_audio": None})
    elapsed_ms = int((time.time() - t0) * 1000)

    if "error" in result:
        print(f"[AMT] ERROR: {result['error']}")
        sys.exit(1)

    notes = result.get("midi_notes", [])
    pedals = result.get("pedal_events", [])
    info = result.get("transcription_info", {})

    print(f"\n[AMT] Transcription done in {elapsed_ms}ms")
    print(f"[AMT] Notes:        {len(notes)}")
    print(f"[AMT] Pedal events: {len(pedals)}")
    print(f"[AMT] Pitch range:  {info.get('pitch_range', 'n/a')}")

    if notes:
        print(f"\n  First 5 notes (pitch, onset_s, offset_s, velocity):")
        for n in notes[:5]:
            print(f"    pitch={n.get('pitch'):>3}  onset={n.get('onset'):>7.3f}s  "
                  f"offset={n.get('offset'):>7.3f}s  vel={n.get('velocity'):>3}")

    if pedals:
        print(f"\n  First 5 pedal events (time_s, value):")
        for p in pedals[:5]:
            print(f"    time={p.get('time'):>7.3f}s  value={p.get('value'):>3}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AMT smoke test")
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"ERROR: WAV not found: {args.wav}")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint dir not found: {args.checkpoint}")
        sys.exit(1)

    run_amt(args.wav, args.checkpoint)


if __name__ == "__main__":
    main()
