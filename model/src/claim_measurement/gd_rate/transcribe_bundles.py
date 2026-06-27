# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "soundfile>=0.12.0",
#     "scipy>=1.10.0",
#     "numba>=0.59.0",
#     "llvmlite>=0.42.0",
#     "pretty_midi>=0.2.10",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
#     "aria @ git+https://github.com/EleutherAI/aria.git",
# ]
# ///
"""G-D transcription harness: real generator-paired clips -> minimal AMT bundles.

For each (recording_id) that has BOTH baseline_v1 generator prose AND local audio,
transcribe STRATIFIED 27s windows spanning the clip in-process via aria-amt and pool
the notes into a minimal bundle sufficient for the dynamics @whole_piece verdict.

Why stratified windows, not full coverage: the dynamics @whole_piece statistic is
mean AMT note-velocity (#101 G-B). On MPS, dense-polyphony decode costs ~30-130s per
27s window, so full coverage of ~8.6-min clips is ~28h for 94 clips. Mean velocity
varies by section (Ballade opening ~63, mid ~50, climax ~77), so a SINGLE window is
temporally biased. Evenly-spaced windows across [frac_lo, frac_hi]*duration remove the
temporal bias at a fraction of the cost; per-clip sampling noise averages out across
the corpus when the faithfulness RATE is aggregated over claims. The bundle records
``window_starts_sec`` / ``coverage_note`` so the approximation is auditable.

The bundle carries STUB ``measure_table`` + ``anchors`` (2 points spanning the real
duration). These satisfy LocationResolver's structural precondition for the whole_piece
tier (>=2 anchors) but are NEVER consumed by the whole_piece dynamics measurement, which
reads only ``notes[*].velocity``. No score alignment is performed or implied -- region/bar
claims on these bundles are out of scope for G-D (dynamics whole_piece only).

Truth-label purity: aria-amt is a non-LLM transcription model. No LLM touches the bundle.

Run (from the worktree, pointing at PRIMARY-tree data which is gitignored/absent here):
    CRESCEND_DEVICE=auto uv run --script transcribe_bundles.py \
        --baseline   /ABS/crescendai/apps/evals/results/baseline_v1.jsonl \
        --audio-root /ABS/crescendai/model/data/evals/skill_eval/chopin_ballade_1/audio \
        --weights    /ABS/crescendai/model/data/weights/aria-amt \
        --out        /ABS/crescendai/model/data/evals/gd_bundles \
        --piece-id chopin_ballade_1 --windows 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_HERE = Path(__file__).resolve()
# parents[4] == repo root when this file lives at
# model/src/claim_measurement/gd_rate/transcribe_bundles.py
REPO = _HERE.parents[4]

SAMPLE_RATE = 16000
CHUNK_S = 27.0  # matches chroma_dtw_eval.amt_regen.AMT_CHUNK_S
BUNDLE_SCHEMA_VERSION = "v1-gd-whole-piece"


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    audio, in_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != SAMPLE_RATE:
        g = gcd(int(in_sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(in_sr) // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _window_starts(duration_sec: float, n_windows: int,
                   frac_lo: float, frac_hi: float) -> list[float]:
    """Evenly-spaced 27s-window START times (sec) across [frac_lo, frac_hi]*duration.

    Clamped so each window fits inside the clip. Deduped/sorted. One window degenerates
    to the clip midpoint.
    """
    usable_end = max(0.0, duration_sec - CHUNK_S)
    if usable_end <= 0:
        return [0.0]
    lo = frac_lo * duration_sec
    hi = min(frac_hi * duration_sec, usable_end)
    if n_windows <= 1:
        return [min(max(lo, 0.0), usable_end)]
    if hi <= lo:
        hi = usable_end
    starts = np.linspace(lo, hi, n_windows)
    starts = np.clip(starts, 0.0, usable_end)
    return sorted({round(float(s), 3) for s in starts})


def _paired_recording_ids(baseline_path: Path, audio_root: Path,
                          piece_id: str) -> list[tuple[str, Path]]:
    """recording_ids for piece_id that have non-empty prose AND a local wav. Sorted."""
    audio_by_id = {p.stem: p for p in sorted(audio_root.glob("*.wav"))}
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for line in baseline_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("piece_slug") != piece_id:
            continue
        if not (r.get("synthesis_text") or "").strip():
            continue
        rid = r["recording_id"]
        if rid in seen or rid not in audio_by_id:
            continue
        seen.add(rid)
        out.append((rid, audio_by_id[rid]))
    return out


def _bundle_path(out_root: Path, piece_id: str, rid: str) -> Path:
    return out_root / piece_id / f"{rid}.json"


def _build_bundle(piece_id: str, rid: str, notes: list[dict],
                 pedal_events: list[dict], duration_sec: float,
                 window_starts: list[float]) -> dict:
    # Stub localization scaffold: 2 anchors + 2 measure rows spanning the real clip.
    # Consumed ONLY to clear LocationResolver's >=2-anchor whole_piece precondition;
    # the whole_piece dynamics measurement ignores it (reads notes[*].velocity).
    dur = round(float(duration_sec), 3)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "piece_id": piece_id,
        "video_id": rid,
        "duration_sec": dur,
        "notes": notes,
        "pedal_events": pedal_events,
        "measure_table": [
            {"bar_number": 1, "start_sec": 0.0},
            {"bar_number": 2, "start_sec": dur},
        ],
        "anchors": {"perf_audio_sec": [0.0, dur], "score_audio_sec": [0.0, dur]},
        "substrate_versions": {"amt": "aria-amt/piano-medium-double-1.0"},
        "coverage_note": (
            "stratified-window whole_piece sampling; NOT full coverage. "
            f"{len(window_starts)} x {CHUNK_S}s windows."
        ),
        "window_starts_sec": window_starts,
    }


def _transcribe_windows(handler, audio: np.ndarray, starts: list[float]) -> tuple[list[dict], list[dict]]:
    """Transcribe each 27s window, offset times to clip-relative, pool notes+pedals."""
    chunk_len = int(CHUNK_S * SAMPLE_RATE)
    all_notes: list[dict] = []
    all_pedals: list[dict] = []
    for s in starts:
        start = int(round(s * SAMPLE_RATE))
        pcm = audio[start:start + chunk_len]
        if len(pcm) < chunk_len:
            pcm = np.concatenate([pcm, np.zeros(chunk_len - len(pcm), dtype=np.float32)])
        notes, pedals = handler._transcribe(pcm)
        for n in notes:
            all_notes.append({
                "pitch": int(n["pitch"]),
                "onset": round(float(n["onset"]) + s, 4),
                "offset": round(float(n["offset"]) + s, 4),
                "velocity": int(n["velocity"]),
            })
        for p in pedals:
            all_pedals.append({
                "time": round(float(p.get("time", 0.0)) + s, 4),
                "value": int(p.get("value", 0)),
            })
    all_notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    all_pedals.sort(key=lambda p: p["time"])
    return all_notes, all_pedals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gd_rate.transcribe_bundles")
    ap.add_argument("--baseline", type=Path,
                   default=REPO / "apps/evals/results/baseline_v1.jsonl")
    ap.add_argument("--audio-root", type=Path, required=True,
                   help="dir of <recording_id>.wav for the piece")
    ap.add_argument("--weights", type=Path,
                   default=REPO / "model/data/weights/aria-amt")
    ap.add_argument("--out", type=Path,
                   default=REPO / "model/data/evals/gd_bundles")
    ap.add_argument("--piece-id", default="chopin_ballade_1")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--frac-lo", type=float, default=0.15)
    ap.add_argument("--frac-hi", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pairs = _paired_recording_ids(args.baseline, args.audio_root, args.piece_id)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"no paired (prose+audio) recordings for {args.piece_id}")
    (args.out / args.piece_id).mkdir(parents=True, exist_ok=True)
    print(f"{len(pairs)} paired recordings; windows={args.windows} "
         f"frac=[{args.frac_lo},{args.frac_hi}] -> {args.out}", flush=True)

    sys.path.insert(0, str(REPO / "apps/inference/amt"))
    os.environ.setdefault("CRESCEND_DEVICE", "auto")
    t0 = time.time()
    from transcription import EndpointHandler
    handler = EndpointHandler(path=str(args.weights))
    print(f"model ready ({time.time()-t0:.1f}s)", flush=True)

    done = 0
    skipped = 0
    for i, (rid, wav) in enumerate(pairs):
        out_path = _bundle_path(args.out, args.piece_id, rid)
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        ct = time.time()
        try:
            audio = _load_audio_16k_mono(wav)
            dur = len(audio) / SAMPLE_RATE
            starts = _window_starts(dur, args.windows, args.frac_lo, args.frac_hi)
            notes, pedals = _transcribe_windows(handler, audio, starts)
        except Exception as exc:  # explicit: record nothing, keep going
            print(f"  [{i+1}/{len(pairs)}] {rid} FAILED: {exc}", flush=True)
            continue
        if not notes:
            print(f"  [{i+1}/{len(pairs)}] {rid} no notes -- skipped", flush=True)
            continue
        bundle = _build_bundle(args.piece_id, rid, notes, pedals, dur, starts)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(bundle))
        tmp.replace(out_path)  # atomic checkpoint
        done += 1
        mv = float(np.mean([n["velocity"] for n in notes]))
        print(f"  [{i+1}/{len(pairs)}] {rid} dur={dur:5.0f}s notes={len(notes):4d} "
             f"mean_vel={mv:5.1f} ({time.time()-ct:4.0f}s) -> {out_path.name}", flush=True)

    print(f"\nDONE: {done} written, {skipped} already-present, "
         f"{len(pairs)-done-skipped} failed/empty. {(time.time()-t0)/60:.1f} min total.",
         flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
