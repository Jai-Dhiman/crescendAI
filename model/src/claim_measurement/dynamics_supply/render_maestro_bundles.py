# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "soundfile>=0.12.0",
#     "pretty_midi>=0.2.10",
# ]
# ///
"""FRONT 9 (G-F): REAL-audio Transkun oracle bundles from MAESTRO test split (#101).

FRONT 8b/8c/8d measured the dynamics rate on FLUIDSYNTH-rendered PercePiano (synthetic audio
because PercePiano is MIDI-only). That leaves the deployment-honesty edge (G-F) open and, for
Transkun specifically, ambiguous: 8d's velocity-range COMPRESSION could be a fluidsynth-OOD
artifact (Transkun trained on real recordings). This harness closes G-F by using MAESTRO, the
one corpus with REAL recorded audio AND time-aligned ground-truth MIDI (Yamaha Disklavier
captures MIDI while the acoustic performance is mic'd). GT MIDI is the independent truth signal;
Transkun-from-real-audio is the scored statistic -- the same non-circular oracle as 8b, on real
audio. MAESTRO TEST split only (Transkun trained on MAESTRO train -> test avoids train-on-test).

Multi-oracle: MAESTRO MIDI carries velocity + CC64 pedal + note offsets, so one render persists
GT for all three verifiable dimensions; each downstream scorer reads the GT field it needs:
  - dynamics    : gt_mean_velocity  (truth)  vs bundle notes' AMT velocity (score)
  - pedaling    : gt_pedal_onfraction (truth) vs bundle pedal_events AMT CC64 (score)
  - articulation: gt_notes offsets    (truth) vs bundle notes' AMT offsets (score / offset gate)

Design: transcribe each ~5-min performance ONCE (full musical context, like production), then
window BOTH the AMT output and the aligned GT MIDI into fixed non-overlapping windows post-hoc.
Two passes: (1) MIDI-only -> per-window GT stats -> corpus medians (cheap, deterministic, needs
no transcription); (2) transcribe + window + write bundles with the medians baked in. Pass 2 is
checkpointed per-performance (a perf's windows are written together; re-transcribe on interrupt).

Run (from worktree; --data-root points at the primary tree's model/data):
    uv run --script render_maestro_bundles.py \
        --maestro-dir /ABS/model/data/raw/maestro/files \
        --csv         /ABS/model/data/raw/maestro/data/maestro-v3.0.0.csv \
        --out         /ABS/model/data/evals/maestro_indep_bundles \
        --window-s 27 --max-per-perf 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO / "model/data"

SR = 16000
MINIMUM_NOTES = 20  # mirror DynamicsMeasurer.MINIMUM_NOTES; a window needs this in BOTH AMT and GT
BUNDLE_SCHEMA_VERSION = "v1-maestro-realaudio-gt-vs-transkun"


def load_pcm(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        # linear resample to 16k (transkun_cli expects 16k mono); avoids a scipy dep
        n_out = int(round(len(audio) * SR / sr))
        audio = np.interp(np.linspace(0, len(audio), n_out, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio)


def gt_notes_and_pedals(midi_path: Path) -> tuple[list[dict], list[dict]]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes, pedals = [], []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({"pitch": int(n.pitch), "onset": float(n.start),
                          "offset": float(n.end), "velocity": int(n.velocity)})
        for cc in inst.control_changes:
            if int(cc.number) == 64:
                pedals.append({"time": float(cc.time), "value": int(cc.value)})
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    pedals.sort(key=lambda e: e["time"])
    return notes, pedals


def pedal_on_fraction(pedals: list[dict], t0: float, t1: float, thresh: int = 64) -> float:
    """Fraction of [t0, t1] during which sustain (CC64) is held down (value >= thresh).
    Pedal is a step function: state holds between events. Integrates the down-state duration."""
    if t1 <= t0:
        return 0.0
    # state entering the window = value of the last event at or before t0 (0 if none)
    state = 0
    for e in pedals:
        if e["time"] <= t0:
            state = e["value"]
        else:
            break
    down = 0.0
    cur_t, cur_state = t0, state
    for e in pedals:
        if e["time"] <= t0 or e["time"] >= t1:
            continue
        if cur_state >= thresh:
            down += e["time"] - cur_t
        cur_t, cur_state = e["time"], e["value"]
    if cur_state >= thresh:
        down += t1 - cur_t
    return down / (t1 - t0)


def window_notes(notes: list[dict], t0: float, t1: float) -> list[dict]:
    return [n for n in notes if t0 <= n["onset"] < t1]


def rel(notes: list[dict], t0: float) -> list[dict]:
    return [{"pitch": n["pitch"], "onset": round(n["onset"] - t0, 4),
             "offset": round(n["offset"] - t0, 4), "velocity": n["velocity"]} for n in notes]


def enumerate_perfs(maestro_dir: Path) -> list[tuple[str, Path, Path]]:
    """(stem, wav, midi) for every downloaded performance with both files present."""
    out = []
    for wav in sorted(maestro_dir.rglob("*.wav")):
        midi = wav.with_suffix(".midi")
        if midi.exists():
            out.append((wav.stem, wav, midi))
    return out


def gt_windows_for_perf(midi: Path, window_s: float, max_per_perf: int) -> list[dict]:
    """Pass-1 GT-only stats per valid window: gt_mean_velocity + gt_pedal_onfraction.
    A window is valid if it holds >= MINIMUM_NOTES GT notes (AMT count checked in pass 2)."""
    notes, pedals = gt_notes_and_pedals(midi)
    if not notes:
        return []
    dur = max(n["offset"] for n in notes)
    n_win = min(max_per_perf, int(dur // window_s))
    out = []
    for k in range(n_win):
        t0, t1 = k * window_s, (k + 1) * window_s
        wn = window_notes(notes, t0, t1)
        if len(wn) < MINIMUM_NOTES:
            continue
        out.append({
            "k": k, "t0": t0, "t1": t1,
            "gt_mean_velocity": float(np.mean([n["velocity"] for n in wn])),
            "gt_pedal_onfraction": pedal_on_fraction(pedals, t0, t1),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.render_maestro_bundles")
    ap.add_argument("--maestro-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None, help="unused for stats; kept for provenance")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=27.0)
    ap.add_argument("--max-per-perf", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    perfs = enumerate_perfs(args.maestro_dir)
    if not perfs:
        raise SystemExit(f"no wav+midi pairs under {args.maestro_dir}")
    print(f"{len(perfs)} MAESTRO performances found", flush=True)

    # -------- Pass 1: MIDI-only corpus medians (no transcription) --------
    gt_by_perf: dict[str, list[dict]] = {}
    all_vel, all_ped = [], []
    for stem, _wav, midi in perfs:
        w = gt_windows_for_perf(midi, args.window_s, args.max_per_perf)
        gt_by_perf[stem] = w
        all_vel.extend(x["gt_mean_velocity"] for x in w)
        all_ped.extend(x["gt_pedal_onfraction"] for x in w)
    if not all_vel:
        raise SystemExit("no valid GT windows (all below MINIMUM_NOTES)")
    gt_vel_median = float(np.median(all_vel))
    gt_ped_median = float(np.median(all_ped))
    print(f"corpus: {len(all_vel)} valid GT windows; median GT velocity={gt_vel_median:.2f}, "
          f"median GT pedal on-fraction={gt_ped_median:.3f}", flush=True)

    # -------- Pass 2: transcribe + window + write bundles --------
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "apps/inference/amt"))
    from transkun_cli import transcribe_pcm

    t0_all = time.time()
    written = skipped = 0
    for i, (stem, wav, midi) in enumerate(perfs):
        gtw = gt_by_perf[stem]
        if not gtw:
            continue
        # skip whole perf if all its bundles already exist
        expected = [args.out / f"{stem}__w{w['k']}.json" for w in gtw]
        if all(p.exists() for p in expected) and not args.force:
            skipped += len(expected)
            continue
        ct = time.time()
        try:
            pcm = load_pcm(wav)
            amt_notes, amt_pedals = transcribe_pcm(pcm)
        except Exception as exc:  # explicit: record nothing for this perf, keep going
            print(f"  [{i+1}/{len(perfs)}] {stem[:40]} TRANSCRIBE FAILED: {exc}", flush=True)
            continue
        gt_notes_all, _gt_ped = gt_notes_and_pedals(midi)
        perf_written = 0
        for w in gtw:
            t0, t1, k = w["t0"], w["t1"], w["k"]
            amt_win = window_notes(amt_notes, t0, t1)
            if len(amt_win) < MINIMUM_NOTES:
                continue  # AMT under-transcribed this window; GT was fine but pair must hold both
            amt_ped_win = [{"time": round(e["time"] - t0, 4), "value": e["value"]}
                           for e in amt_pedals if t0 <= e["time"] < t1]
            gt_win = window_notes(gt_notes_all, t0, t1)
            bundle = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "piece_id": "maestro",
                "video_id": f"{stem}__w{k}",
                "duration_sec": round(args.window_s, 3),
                "notes": rel(amt_win, t0),
                "pedal_events": amt_ped_win,
                "measure_table": [{"bar_number": 1, "start_sec": 0.0},
                                  {"bar_number": 2, "start_sec": round(args.window_s, 3)}],
                "anchors": {"perf_audio_sec": [0.0, args.window_s],
                            "score_audio_sec": [0.0, args.window_s]},
                "substrate_versions": {"amt": "transkun/2.0.1"},
                "gt_mean_velocity": round(w["gt_mean_velocity"], 3),
                "gt_corpus_median": round(gt_vel_median, 3),
                "gt_pedal_onfraction": round(w["gt_pedal_onfraction"], 4),
                "gt_corpus_pedal_median": round(gt_ped_median, 4),
                "gt_notes": rel(gt_win, t0),
                "coverage_note": "real MAESTRO-test audio, full-performance transcription windowed post-hoc.",
            }
            out_path = args.out / f"{stem}__w{k}.json"
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(bundle))
            tmp.replace(out_path)
            perf_written += 1
        written += perf_written
        amt_v = float(np.mean([n["velocity"] for n in amt_notes])) if amt_notes else 0.0
        print(f"  [{i+1}/{len(perfs)}] {stem[:38]:38s} wins={perf_written:2d} "
              f"amt_vel={amt_v:5.1f} ({time.time()-ct:4.0f}s)", flush=True)

    print(f"\nDONE: {written} bundles written, {skipped} present. "
          f"{(time.time()-t0_all)/60:.1f} min.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
