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
"""FRONT 8b: render+transcribe stratified PercePiano segments -> AMT bundles that carry
BOTH the AMT note velocities AND the ground-truth MIDI mean velocity (#101 / #67).

Purpose: break the FRONT-8 signal-fidelity circularity. Front 8 cued the teacher from AMT
mean velocity and SCORED with AMT mean velocity (same statistic -> rate is circular). Here
the cue/truth is GROUND-TRUTH MIDI velocity and the SCORE is AMT velocity -- two INDEPENDENT
measurements of the same performance. GT MIDI velocity only exists for PercePiano (which IS
MIDI); the real YouTube gd_bundles have no ground truth, which is why this moves substrate.

Also fixes the G-B non-persistence gap: the G-B gate rendered to a tempdir and cached only
180 anonymous scalar means. This persists per-segment bundles (AMT note arrays + gt_vel), so
the downstream oracle rate + any re-score is cheap and reproducible.

Pipeline per segment:  PercePiano .mid --fluidsynth(fixed gain 0.5, 16k)--> wav
  --aria-amt._transcribe--> notes w/ AMT velocity  (+ gt_mean_velocity from pretty_midi).

Stratified sampling (seed 42): segments are binned soft / balanced / loud by GT mean velocity
vs the labeled-corpus GT median (deadband TAU_GT), then sampled evenly per stratum so all
three verdict classes are represented (front 8 had 0 soft cues). The GT median is stored in
every bundle so the score-time label + a tau sweep need no re-render.

Run (from worktree, writing to PRIMARY-tree data which is gitignored/absent in the worktree):
    CRESCEND_DEVICE=auto uv run --script render_percepiano_bundles.py \
        --out /ABS/crescendai/model/data/evals/percepiano_indep_bundles \
        --n 45
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parents[4]
# Default data root; overridable via --data-root (a git worktree's model/data holds only
# tracked configs -- the big MIDI/weights/soundfont data is gitignored, so point this at the
# primary checkout's model/data when running from a worktree).
DEFAULT_DATA_ROOT = REPO / "model/data"

SR = 16000
GAIN = 0.5
CHUNK_S = 27.0
TAU_GT = 6.5  # GT-velocity deadband for stratification (mirrors the AMT tau=6.5, same units)
BUNDLE_SCHEMA_VERSION = "v1-indep-gt-vs-amt"


def render(midi_path: Path, wav_path: Path, sf2: Path) -> None:
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav_path),
         str(sf2), str(midi_path)],
        check=True, capture_output=True,
    )


def gt_mean_velocity(midi_path: Path) -> float | None:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    v = [n.velocity for inst in pm.instruments for n in inst.notes]
    return float(np.mean(v)) if v else None


def transcribe_full(handler, audio: np.ndarray) -> list[dict]:
    """Contiguous 27s chunks covering the WHOLE short segment; pool clip-relative notes."""
    chunk_len = int(CHUNK_S * SR)
    n_chunks = max(1, int(np.ceil(len(audio) / chunk_len)))
    all_notes: list[dict] = []
    for c in range(n_chunks):
        start = c * chunk_len
        pcm = audio[start:start + chunk_len]
        if len(pcm) < chunk_len:
            pcm = np.concatenate([pcm, np.zeros(chunk_len - len(pcm), dtype=np.float32)])
        offset = start / SR
        notes, _pedals = handler._transcribe(pcm)
        for n in notes:
            all_notes.append({
                "pitch": int(n["pitch"]),
                "onset": round(float(n["onset"]) + offset, 4),
                "offset": round(float(n["offset"]) + offset, 4),
                "velocity": int(n["velocity"]),
            })
    all_notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return all_notes


def stratify(gt_by_stem: dict[str, float], median: float, n: int,
             tau_gt: float = TAU_GT, seed: int = 42) -> list[str]:
    """Evenly sample n stems across soft / balanced / loud GT-velocity strata."""
    soft = [s for s, v in gt_by_stem.items() if v < median - tau_gt]
    loud = [s for s, v in gt_by_stem.items() if v > median + tau_gt]
    bal = [s for s, v in gt_by_stem.items() if abs(v - median) <= tau_gt]
    rng = np.random.default_rng(seed)
    per = max(1, n // 3)
    picked: list[str] = []
    for bucket in (soft, bal, loud):
        arr = sorted(bucket)
        take = min(per, len(arr))
        idx = rng.permutation(len(arr))[:take]
        picked.extend(arr[i] for i in sorted(idx))
    return sorted(picked)


def build_bundle(stem: str, notes: list[dict], duration_sec: float,
                 gt_vel: float, gt_median: float) -> dict:
    dur = round(float(duration_sec), 3)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "piece_id": "percepiano",
        "video_id": stem,                     # segment id (== MIDI stem)
        "duration_sec": dur,
        "notes": notes,
        "pedal_events": [],
        "measure_table": [
            {"bar_number": 1, "start_sec": 0.0},
            {"bar_number": 2, "start_sec": dur},
        ],
        "anchors": {"perf_audio_sec": [0.0, dur], "score_audio_sec": [0.0, dur]},
        "substrate_versions": {"amt": "aria-amt/piano-medium-double-1.0"},
        "gt_mean_velocity": round(float(gt_vel), 3),   # INDEPENDENT truth signal
        "gt_corpus_median": round(float(gt_median), 3),
        "coverage_note": "full-segment coverage (contiguous 27s chunks); short PercePiano excerpt.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.render_percepiano_bundles")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="primary-checkout model/data (MIDI/labels/soundfont/weights live here)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <data-root>/evals/percepiano_indep_bundles")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    data_root = args.data_root
    midi_dir = data_root / "midi/percepiano"
    labels_path = data_root / "labels/composite/composite_labels.json"
    sf2 = data_root / "soundfonts/MuseScore_General.sf3"
    weights = args.weights or (data_root / "weights/aria-amt")
    out_dir = args.out or (data_root / "evals/percepiano_indep_bundles")

    labels = json.loads(labels_path.read_text())
    stems = [p.stem for p in sorted(midi_dir.glob("*.mid"))
             if labels.get(p.stem, {}).get("dynamics") is not None]
    print(f"computing GT mean velocity over {len(stems)} labeled segments...", flush=True)
    gt_by_stem: dict[str, float] = {}
    for s in stems:
        v = gt_mean_velocity(midi_dir / f"{s}.mid")
        if v is not None:
            gt_by_stem[s] = v
    gt_median = float(np.median(list(gt_by_stem.values())))
    picked = stratify(gt_by_stem, gt_median, args.n)
    print(f"GT median velocity={gt_median:.1f}; sampled {len(picked)} across strata "
          f"(soft/balanced/loud vs median +-{TAU_GT})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "apps/inference/amt"))
    os.environ.setdefault("CRESCEND_DEVICE", "auto")
    t0 = time.time()
    from transcription import EndpointHandler
    handler = EndpointHandler(path=str(weights))
    print(f"model ready ({time.time()-t0:.1f}s)\n", flush=True)

    done = skipped = 0
    with tempfile.TemporaryDirectory() as td:
        for i, stem in enumerate(picked):
            out_path = out_dir / f"{stem}.json"
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            ct = time.time()
            try:
                wav = Path(td) / "clip.wav"
                render(midi_dir / f"{stem}.mid", wav, sf2)
                audio, _ = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                dur = len(audio) / SR
                notes = transcribe_full(handler, audio)
            except Exception as exc:  # explicit: record nothing, keep going
                print(f"  [{i+1}/{len(picked)}] {stem[:34]} FAILED: {exc}", flush=True)
                continue
            if not notes:
                print(f"  [{i+1}/{len(picked)}] {stem[:34]} no notes -- skipped", flush=True)
                continue
            bundle = build_bundle(stem, notes, dur, gt_by_stem[stem], gt_median)
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(bundle))
            tmp.replace(out_path)  # atomic checkpoint
            done += 1
            amt_v = float(np.mean([n["velocity"] for n in notes]))
            print(f"  [{i+1}/{len(picked)}] {stem[:30]:30s} gt={gt_by_stem[stem]:5.1f} "
                  f"amt={amt_v:5.1f} n={len(notes):4d} ({time.time()-ct:4.0f}s)", flush=True)

    print(f"\nDONE: {done} written, {skipped} present, "
          f"{len(picked)-done-skipped} failed/empty. {(time.time()-t0)/60:.1f} min.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
