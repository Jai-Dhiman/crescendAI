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
"""G-B GATE (dynamics): does AMT-estimated mean note-velocity (from fixed-gain
piano-rendered audio) still track perceived dynamics, the way ground-truth MIDI mean
velocity does (partial-rho ~0.56)?  (#101 GATE 3 UPDATE)

Pipeline per clip:  PercePiano .mid --fluidsynth(fixed gain)--> 16k wav
  --aria-amt._transcribe--> notes w/ velocity --> mean AMT velocity.
Correlate (halo-controlled partial Spearman) vs perceived dynamics. PASS iff >= ~0.5.
Also reports AMT-vel vs GT-vel preservation (THE risk: does AMT normalize level away?).

Run:  CRESCEND_DEVICE=auto uv run --script amt_dynamics_gb_gate.py [N]
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
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[4]
MIDI_DIR = REPO / "model/data/midi/percepiano"
LABELS = json.loads((REPO / "model/data/labels/composite/composite_labels.json").read_text())
SF2 = REPO / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = REPO / "model/data/weights/aria-amt"
ALL_DIMS = ["timing", "dynamics", "pedaling", "articulation", "phrasing", "interpretation"]
SR = 16000
GAIN = 0.5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

sys.path.insert(0, str(REPO / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.vstack([rz, np.ones_like(rz)]).T
    bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    by, *_ = np.linalg.lstsq(design, ry, rcond=None)
    ex, ey = rx - design @ bx, ry - design @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def render(midi_path, wav_path):
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav_path),
         str(SF2), str(midi_path)],
        check=True, capture_output=True,
    )


def gt_mean_velocity(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    v = [n.velocity for inst in pm.instruments for n in inst.notes]
    return float(np.mean(v)) if v else None


print("Loading aria-amt EndpointHandler...", flush=True)
from transcription import EndpointHandler
handler = EndpointHandler(path=str(WEIGHTS))
print("Model ready.\n", flush=True)

labeled = [p for p in sorted(MIDI_DIR.glob("*.mid")) if LABELS.get(p.stem, {}).get("dynamics") is not None]
rng = np.random.default_rng(42)
idx = rng.permutation(len(labeled))[:N]
paths = [labeled[i] for i in sorted(idx)]
rows = []
with tempfile.TemporaryDirectory() as td:
    for i, mp in enumerate(paths):
        wav = Path(td) / "clip.wav"
        try:
            render(mp, wav)
            audio, _ = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            notes, _ped = handler._transcribe(audio)
        except Exception as exc:
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:30]} FAILED: {exc}", flush=True)
            continue
        if not notes:
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:30]} no notes", flush=True)
            continue
        amt_vel = float(np.mean([n["velocity"] for n in notes]))
        gt_vel = gt_mean_velocity(mp)
        lab = LABELS[mp.stem]
        rows.append({
            "amt_vel": amt_vel, "gt_vel": gt_vel, "n_amt": len(notes),
            "dyn": float(lab["dynamics"]),
            "ctrl": float(np.mean([lab[d] for d in ALL_DIMS if d != "dynamics"])),
        })
        print(f"  [{i+1}/{len(paths)}] {mp.stem[:28]:28s} amt_vel={amt_vel:5.1f} "
              f"gt_vel={gt_vel:5.1f} n={len(notes)}", flush=True)

print(f"\n=== n={len(rows)} ===")
amt = [r["amt_vel"] for r in rows]
gt = [r["gt_vel"] for r in rows]
dyn = [r["dyn"] for r in rows]
ctrl = [r["ctrl"] for r in rows]
print(f"amt_vel range [{min(amt):.1f},{max(amt):.1f}]  gt_vel range [{min(gt):.1f},{max(gt):.1f}]")
print(f"AMT-vel  vs perceived dynamics : raw={spearmanr(amt,dyn)[0]:+.3f} partial={partial_spearman(amt,dyn,ctrl):+.3f}")
print(f"GT-vel   vs perceived dynamics : raw={spearmanr(gt,dyn)[0]:+.3f} partial={partial_spearman(gt,dyn,ctrl):+.3f}  (sanity ~0.56)")
print(f"AMT-vel  vs GT-vel (PRESERVED?): {spearmanr(amt,gt)[0]:+.3f}")
brng = np.random.default_rng(7)
amt_a, dyn_a, ctrl_a = np.array(amt), np.array(dyn), np.array(ctrl)
boots = []
for _ in range(2000):
    s = brng.integers(0, len(amt_a), len(amt_a))
    try:
        boots.append(partial_spearman(amt_a[s], dyn_a[s], ctrl_a[s]))
    except Exception:
        pass
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"AMT-vel partial-rho 95% CI: [{lo:+.3f}, {hi:+.3f}]  half-width={(hi-lo)/2:.3f}")
out = REPO / "model/data/results/gb_amt_velocity_gate.json"
out.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2))
print(f"\nrows -> {out}")
