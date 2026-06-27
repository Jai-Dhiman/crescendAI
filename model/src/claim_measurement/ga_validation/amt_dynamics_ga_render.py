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
"""Dynamics G-A stage 1: construction-known velocity-scaling corruption -> render -> AMT.

For N base PercePiano clips, scale ALL MIDI velocities by {0.55 soft, 1.0 neutral,
1.45 loud} (a true, construction-known performance-level change), render at FIXED gain,
transcribe with aria-amt, and record the per-(clip,scale) AMT velocity lists. Stage 2
(amt_dynamics_ga_metrics) feeds these to the REAL DynamicsMeasurer + frozen route_verdict.

Run:  CRESCEND_DEVICE=auto uv run --script amt_dynamics_ga_render.py [N]
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

REPO = Path(__file__).resolve().parents[4]
MIDI_DIR = REPO / "model/data/midi/percepiano"
LABELS = json.loads((REPO / "model/data/labels/composite/composite_labels.json").read_text())
SF2 = REPO / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = REPO / "model/data/weights/aria-amt"
SR = 16000
GAIN = 0.5
SCALES = {"soft": 0.55, "neutral": 1.0, "loud": 1.45}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

sys.path.insert(0, str(REPO / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def render_scaled(midi_path, scale, wav_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for inst in pm.instruments:
        for n in inst.notes:
            n.velocity = int(np.clip(round(n.velocity * scale), 1, 127))
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        tmid = tf.name
    pm.write(tmid)
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav_path), str(SF2), tmid],
        check=True, capture_output=True,
    )
    os.unlink(tmid)


print("Loading aria-amt...", flush=True)
from transcription import EndpointHandler
handler = EndpointHandler(path=str(WEIGHTS))
print("Model ready.\n", flush=True)

labeled = [p for p in sorted(MIDI_DIR.glob("*.mid")) if LABELS.get(p.stem, {}).get("dynamics") is not None]
rng = np.random.default_rng(123)
idx = sorted(rng.permutation(len(labeled))[:N])
paths = [labeled[i] for i in idx]

out = {}
with tempfile.TemporaryDirectory() as td:
    for i, mp in enumerate(paths):
        rec = {}
        for sname, sval in SCALES.items():
            wav = Path(td) / "c.wav"
            try:
                render_scaled(mp, sval, wav)
                audio, _ = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                notes, _ped = handler._transcribe(audio)
                vels = [int(n["velocity"]) for n in notes]
            except Exception as exc:
                print(f"  [{i+1}/{len(paths)}] {mp.stem[:24]} {sname} FAIL: {exc}", flush=True)
                vels = []
            rec[sname] = vels
        out[mp.stem] = rec
        means = {s: (round(float(np.mean(v)), 1) if v else None) for s, v in rec.items()}
        print(f"  [{i+1}/{len(paths)}] {mp.stem[:24]:24s} mean_vel soft/neut/loud="
              f"{means['soft']}/{means['neutral']}/{means['loud']}", flush=True)

outpath = REPO / "model/data/results/ga_dynamics_velocities.json"
outpath.write_text(json.dumps(out, indent=2))
print(f"\nwrote {len(out)} clips -> {outpath}")
