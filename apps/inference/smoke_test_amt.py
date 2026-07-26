# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "soundfile>=0.12.0",
#     "pretty_midi>=0.2.10",
# ]
# ///
"""Smoke test: Transkun transcription on the committed piano fixture.

Runs the full transcription pipeline (Transkun via transkun_cli) and prints
note/pedal counts. Defaults to the guaranteed-present committed fixture so the
gate genuinely runs on every fresh checkout.

Usage:
    cd apps/inference && uv run smoke_test_amt.py
    cd apps/inference && uv run smoke_test_amt.py --wav path/to/other.wav
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

# Must insert amt/ so transcription.py + transkun_cli are importable
AMT_DIR = str(Path(__file__).resolve().parent / "amt")
sys.path.insert(0, AMT_DIR)

DEFAULT_WAV = str(
    Path(__file__).resolve().parent / "amt" / "fixtures" / "piano_sample_5s_16k.wav"
)


def run_amt(wav_path: str, checkpoint_dir: str) -> None:
    from transcription import EndpointHandler

    print(f"\n[AMT] Resolving Transkun transcriber (path ignored: {checkpoint_dir!r})...")
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
    # Transkun manages its own bundled weights; path is retained for call-site
    # compatibility only (EndpointHandler ignores it). No aria checkpoint gate.
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"ERROR: WAV not found: {args.wav}")
        sys.exit(1)

    run_amt(args.wav, args.checkpoint)


if __name__ == "__main__":
    main()
