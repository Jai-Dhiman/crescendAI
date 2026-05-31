# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
# ]
# ///
"""Transcribe a wav via Aria-AMT and dump notes + pedals to JSON.

    cd apps/inference && uv run amt_to_json.py --wav path.wav --out notes.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

AMT_DIR = str(Path(__file__).resolve().parent / "amt")
sys.path.insert(0, AMT_DIR)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

DEFAULT_CKPT = str(Path(__file__).resolve().parents[2] / "model" / "data" / "weights" / "aria-amt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = p.parse_args()

    from transcription import EndpointHandler

    handler = EndpointHandler(path=args.checkpoint)
    audio_bytes = Path(args.wav).read_bytes()
    chunk_b64 = base64.b64encode(audio_bytes).decode()

    t0 = time.time()
    result = handler({"chunk_audio": chunk_b64, "context_audio": None})
    elapsed_ms = int((time.time() - t0) * 1000)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    notes = result.get("midi_notes", [])
    pedals = result.get("pedal_events", [])
    info = result.get("transcription_info", {})
    Path(args.out).write_text(json.dumps({"notes": notes, "pedals": pedals, "info": info}, indent=2))

    print(f"Transcribed {len(notes)} notes, {len(pedals)} pedals in {elapsed_ms}ms -> {args.out}")
    print(f"Pitch range: {info.get('pitch_range', 'n/a')}")
    # quick pitch-class histogram for sanity (C-major prelude should be diatonic-heavy)
    from collections import Counter
    pcs = Counter(n["pitch"] % 12 for n in notes)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    total = sum(pcs.values()) or 1
    print("Pitch-class distribution:")
    for pc in range(12):
        bar = "#" * int(40 * pcs.get(pc, 0) / total)
        print(f"  {names[pc]:<2} {pcs.get(pc, 0):>4}  {bar}")


if __name__ == "__main__":
    main()
