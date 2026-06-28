# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "soundfile>=0.12.0",
#     "numba>=0.59.0",
#     "llvmlite>=0.42.0",
#     "pretty_midi>=0.2.10",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
#     "aria @ git+https://github.com/EleutherAI/aria.git",
# ]
# ///
"""AMT-fidelity map for onset (gates timing) and offset/duration (gates articulation).

Pipeline per clip: ground-truth PercePiano MIDI -> fluidsynth render (fixed gain,
16k) -> aria-amt _transcribe -> greedy AMT<->GT note match -> onset/duration
fidelity. The render path is byte-identical to the velocity gate that scored 0.965,
so onset/duration numbers are directly comparable.

Why this is the go/no-go: timing reduces to per-note onset_deviation_ms with +-30ms
rush/drag thresholds, so the AMT onset NOISE (std of amt-gt onset error) must sit
well under 30ms. Articulation reduces to mean(perf_dur)/mean(score_dur), so the AMT
OFFSET head -- never tested before -- must recover note duration.

Run (from the amt_fidelity dir, data read from the PRIMARY checkout):
    CRESCEND_DEVICE=auto uv run --script onset_duration_render.py [N]

Worktree note: model/data is gitignored and absent here, so all data paths point
at the primary checkout absolutely. Results land in the PRIMARY tree's results dir.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_metrics import (  # noqa: E402
    match_notes,
    onset_fidelity,
    duration_fidelity,
    spearman,
)

# model/data is gitignored -> absent in worktrees; anchor to the primary checkout.
PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
MIDI_DIR = PRIMARY / "model/data/midi/percepiano"
SF2 = PRIMARY / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = PRIMARY / "model/data/weights/aria-amt"
OUT = PRIMARY / "model/data/results/amt_fidelity_onset_duration.json"

SR = 16000
GAIN = 0.5
AMT_WINDOW_S = 30.0       # aria-amt _transcribe hard-truncates to this
GUARD_S = 1.0             # drop GT notes near the truncation boundary
ONSET_WINDOW_S = 0.1      # match tolerance (pitch-exact, nearest onset)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

sys.path.insert(0, str(PRIMARY / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def load_gt_notes(midi_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({"pitch": int(n.pitch), "onset": float(n.start),
                          "offset": float(n.end), "velocity": int(n.velocity)})
    notes.sort(key=lambda d: d["onset"])
    return notes


def render(midi_path: Path, wav_path: Path):
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav_path),
         str(SF2), str(midi_path)],
        check=True, capture_output=True,
    )


def crop(notes, cutoff_s):
    """Keep notes fully inside the AMT-visible window (fair recall + clean duration)."""
    return [n for n in notes if n["onset"] <= cutoff_s and n["offset"] <= cutoff_s]


print("Loading aria-amt...", flush=True)
from transcription import EndpointHandler  # noqa: E402
handler = EndpointHandler(path=str(WEIGHTS))
print("Model ready.\n", flush=True)

rng = np.random.default_rng(2026)
all_midis = sorted(MIDI_DIR.glob("*.mid"))
idx = sorted(rng.permutation(len(all_midis))[:N])
paths = [all_midis[i] for i in idx]

records = {}
pooled_pairs = []           # all matched (gt, amt) across clips -> pooled onset/dur
clip_mean_gt_dur, clip_mean_amt_dur = [], []   # cross-clip articulation spearman
n_truncated = 0

with tempfile.TemporaryDirectory() as td:
    for i, mp in enumerate(paths):
        wav = Path(td) / "c.wav"
        try:
            gt_full = load_gt_notes(mp)
            render(mp, wav)
            audio, _ = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            clip_dur = len(audio) / SR
            cutoff = min(clip_dur, AMT_WINDOW_S) - GUARD_S
            truncated = clip_dur > AMT_WINDOW_S
            n_truncated += int(truncated)

            gt = crop(gt_full, cutoff)
            amt_notes, _ped = handler._transcribe(audio)
            amt = crop([{"pitch": int(n["pitch"]), "onset": float(n["onset"]),
                         "offset": float(n["offset"]), "velocity": int(n["velocity"])}
                        for n in amt_notes], cutoff)

            m = match_notes(gt, amt, onset_window_s=ONSET_WINDOW_S)
            of = onset_fidelity(m["pairs"])
            df = duration_fidelity(m["pairs"])
            pooled_pairs.extend(m["pairs"])
            if df.get("n", 0) > 0:
                clip_mean_gt_dur.append(df["mean_gt_dur"])
                clip_mean_amt_dur.append(df["mean_amt_dur"])

            rec = {"n_gt": m["n_gt"], "n_amt": m["n_amt"], "n_matched": m["n_matched"],
                   "recall": round(m["recall"], 3), "truncated": truncated,
                   "onset": {k: round(v, 2) for k, v in of.items() if k != "n"},
                   "duration": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in df.items() if k != "n"}}
            records[mp.stem] = rec
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:30]:30s} "
                  f"rec={m['recall']:.2f} n={m['n_matched']:3d} "
                  f"onset bias/noise={of.get('bias_ms', float('nan')):+.1f}/"
                  f"{of.get('noise_ms', float('nan')):.1f}ms "
                  f"dur ratio={df.get('median_ratio', float('nan')):.2f}"
                  f"{' [TRUNC]' if truncated else ''}", flush=True)
        except Exception as exc:  # noqa: BLE001 - surface and keep going
            records[mp.stem] = {"error": repr(exc)}
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:30]:30s} FAIL: {exc!r}", flush=True)

        OUT.write_text(json.dumps(records, indent=2))  # checkpoint every clip

# ---- pooled verdict ---------------------------------------------------------
pooled_onset = onset_fidelity(pooled_pairs)
pooled_dur = duration_fidelity(pooled_pairs)
art_spearman = (spearman(clip_mean_amt_dur, clip_mean_gt_dur)
                if len(clip_mean_gt_dur) >= 2 else float("nan"))
recalls = [r["recall"] for r in records.values() if "recall" in r]

summary = {
    "n_clips": len([r for r in records.values() if "recall" in r]),
    "n_truncated": n_truncated,
    "median_recall": round(float(np.median(recalls)), 3) if recalls else None,
    "pooled_matched_notes": pooled_onset.get("n", 0),
    "onset": {k: round(v, 2) for k, v in pooled_onset.items() if k != "n"},
    "duration_pooled": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in pooled_dur.items() if k != "n"},
    "articulation_clip_mean_dur_spearman": round(art_spearman, 4),
}
records["_summary"] = summary
OUT.write_text(json.dumps(records, indent=2))

print("\n=== AMT-FIDELITY: ONSET (timing) + DURATION (articulation) ===")
print(f"  clips={summary['n_clips']}  truncated={n_truncated}  "
      f"median recall={summary['median_recall']}  pooled notes={pooled_onset.get('n', 0)}")
print(f"  ONSET   bias={pooled_onset.get('bias_ms'):+.1f}ms  "
      f"noise(std)={pooled_onset.get('noise_ms'):.1f}ms  "
      f"median|err|={pooled_onset.get('median_abs_ms'):.1f}ms  "
      f"p90|err|={pooled_onset.get('p90_abs_ms'):.1f}ms")
print(f"          vs +-30ms rush/drag band -> "
      f"{'VIABLE' if pooled_onset.get('noise_ms', 1e9) < 30.0 else 'BLOCKED'} "
      f"(noise {'<' if pooled_onset.get('noise_ms', 1e9) < 30.0 else '>='} 30ms)")
print(f"  DURATION median ratio={pooled_dur.get('median_ratio'):.2f}  "
      f"per-note spearman={pooled_dur.get('spearman_dur'):.3f}  "
      f"clip-mean-dur spearman={art_spearman:.3f} (cf velocity 0.965)")
print(f"\nwrote {summary['n_clips']} clips -> {OUT}")
