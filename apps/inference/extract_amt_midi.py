# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "pretty_midi>=0.2.10",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
#     "aria @ git+https://github.com/EleutherAI/aria.git",
# ]
# ///
"""Batch Aria-AMT transcription: a directory of WAVs -> per-segment {seg_id}.mid.

This is the core, tier-agnostic step of the #72 AMT MIDI extraction pipeline.
It reuses the production-parity `EndpointHandler` (apps/inference/amt/transcription.py),
which already carries the MPS/CPU device port and KV-cache monkey-patch, so local
transcription matches what the HF endpoint produces.

The handler returns notes/pedals as JSON lists; this script serializes them to a
real .mid file (via pretty_midi) because the Aria symbolic path
(model/src/model_improvement/aria_encoder.py::AriaMidiPairDataset) loads
`{midi_dir}/{seg_id}.mid` with `MidiDict.from_midi(path)`.

Design:
  - Idempotent / resumable: an existing {seg_id}.mid is skipped unless --overwrite.
  - Explicit failure surfacing (no silent fallback): per-segment failures are
    recorded to {out_dir}/_failures.jsonl and the process exits non-zero if any
    occurred, so a partial run can never masquerade as complete.
  - Per-segment sanity stats appended to {out_dir}/_sanity.jsonl for spot-checking.

Usage:
    cd apps/inference && uv run extract_amt_midi.py \
        --wav-dir  /abs/path/to/wavs \
        --out-dir  /abs/path/to/model/data/midi/amt/t1 \
        [--checkpoint /abs/path/to/aria-amt] [--overwrite] [--limit N]
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

AMT_DIR = str(Path(__file__).resolve().parent / "amt")
sys.path.insert(0, AMT_DIR)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

DEFAULT_CKPT = str(
    Path(__file__).resolve().parents[2] / "model" / "data" / "weights" / "aria-amt"
)


def log(msg: str) -> None:
    print(json.dumps({"ts": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)


def notes_pedals_to_midi(notes: list[dict], pedals: list[dict], out_path: Path) -> None:
    """Write AMT notes + pedal events to a Standard MIDI File via pretty_midi.

    Notes: {pitch, onset, offset, velocity} in seconds. Pedals: {time, value}.
    Pedal is written as CC64 (sustain). One acoustic-grand instrument (program 0).
    Written atomically (tmp + rename) so a killed run never leaves a truncated .mid.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for n in notes:
        pitch = int(n["pitch"])
        if not 0 <= pitch <= 127:
            continue
        velocity = max(1, min(127, int(n["velocity"])))
        start = float(n["onset"])
        end = float(n["offset"])
        if end <= start:
            end = start + 0.01
        inst.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )
    for p in pedals:
        value = max(0, min(127, int(p["value"])))
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, value=value, time=float(p["time"]))
        )
    pm.instruments.append(inst)

    tmp_path = out_path.with_suffix(".mid.tmp")
    pm.write(str(tmp_path))
    tmp_path.replace(out_path)


def pitch_class_entropy(notes: list[dict]) -> float:
    """Shannon entropy (bits) over the 12 pitch classes. Sanity signal: a sane
    tonal transcription is neither flat (~3.58 = uniform noise) nor degenerate
    (~0 = one repeated note). Real piano music typically lands ~2.5-3.4."""
    if not notes:
        return 0.0
    counts = collections.Counter(int(n["pitch"]) % 12 for n in notes)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return round(ent, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav-dir", required=True, help="Directory of {seg_id}.wav inputs.")
    ap.add_argument("--out-dir", required=True, help="Directory for {seg_id}.mid outputs.")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument(
        "--seg-ids",
        default=None,
        help="Optional newline-delimited file of seg_ids to restrict to. "
        "Default: every *.wav in --wav-dir.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Re-transcribe even if .mid exists.")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N transcriptions (debug).")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"--wav-dir does not exist: {wav_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seg_ids:
        seg_ids = [
            s.strip() for s in Path(args.seg_ids).read_text().splitlines() if s.strip()
        ]
    else:
        seg_ids = sorted(p.stem for p in wav_dir.glob("*.wav"))
    if not seg_ids:
        raise FileNotFoundError(f"No seg_ids to process (no *.wav in {wav_dir}?)")

    log(f"Loading Aria-AMT from {args.checkpoint}")
    from transcription import EndpointHandler

    handler = EndpointHandler(path=args.checkpoint)
    log(f"Model ready. {len(seg_ids)} candidate segments.")

    failures_path = out_dir / "_failures.jsonl"
    sanity_path = out_dir / "_sanity.jsonl"
    n_done = n_skip = n_fail = 0
    t_start = time.time()

    for i, seg_id in enumerate(seg_ids):
        out_path = out_dir / f"{seg_id}.mid"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        wav_path = wav_dir / f"{seg_id}.wav"
        if not wav_path.exists():
            n_fail += 1
            with failures_path.open("a") as f:
                f.write(json.dumps({"seg_id": seg_id, "error": "missing_wav"}) + "\n")
            continue

        try:
            chunk_b64 = base64.b64encode(wav_path.read_bytes()).decode()
            result = handler({"chunk_audio": chunk_b64, "context_audio": None})
            if "error" in result:
                raise RuntimeError(json.dumps(result["error"]))
            notes = result.get("midi_notes", [])
            pedals = result.get("pedal_events", [])
            notes_pedals_to_midi(notes, pedals, out_path)

            info = result.get("transcription_info", {})
            with sanity_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "seg_id": seg_id,
                            "note_count": len(notes),
                            "pitch_range": info.get("pitch_range"),
                            "pedal_count": len(pedals),
                            "pc_entropy": pitch_class_entropy(notes),
                            "ms": info.get("transcription_time_ms"),
                        }
                    )
                    + "\n"
                )
            n_done += 1
        except Exception as e:  # noqa: BLE001 -- record + continue, surfaced at exit
            n_fail += 1
            with failures_path.open("a") as f:
                f.write(
                    json.dumps(
                        {"seg_id": seg_id, "error": str(e), "tb": traceback.format_exc()}
                    )
                    + "\n"
                )

        if (i + 1) % 25 == 0 or (i + 1) == len(seg_ids):
            rate = n_done / max(1e-9, time.time() - t_start)
            log(
                f"{i + 1}/{len(seg_ids)}  done={n_done} skip={n_skip} fail={n_fail}  "
                f"{rate:.2f} seg/s"
            )
        if args.limit and n_done >= args.limit:
            log(f"Hit --limit {args.limit}, stopping.")
            break

    log(f"FINISHED done={n_done} skip={n_skip} fail={n_fail} out_dir={out_dir}")
    if n_fail:
        log(f"{n_fail} failures recorded in {failures_path} -- exiting non-zero.")
        sys.exit(1)


if __name__ == "__main__":
    main()
