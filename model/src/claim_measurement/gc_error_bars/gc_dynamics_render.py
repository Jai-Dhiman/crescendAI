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
"""G-C stage 1: empirical substrate error of the dynamics statistic (mean AMT velocity).

For each of N PercePiano clips: render ONCE at the measurer's calibration gain, then
re-transcribe the SAME performance under K perceptually neutral recording nuisances
(sub-JND gain jitter + high-SNR additive white noise). aria-amt decodes greedily, so
re-transcribing an identical WAV is a no-op (verified per clip: DETERMINISM CHECK) --
the churn we record is the substrate's response to nuisance-equivalent captures, i.e.
exactly what re-recording the same performance would exercise. Stage 2
(gc_churn_metrics) reduces the per-variant velocities to the measured 1-sigma the
dead-band must exceed.

Nuisance model (reported in the output so the dead-band is auditable, not a hidden knob):
  gain jitter  ~ U(-GAIN_JITTER_DB, +GAIN_JITTER_DB) dB  (below the ~1 dB loudness JND)
  additive     ~ white Gaussian at SNR_DB relative to clip RMS (inaudible)

Run:  CRESCEND_DEVICE=auto uv run --script gc_dynamics_render.py [N]
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Reuse the amt_fidelity greedy matcher for per-note churn (pitch-exact, nearest onset).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amt_fidelity"))
from fidelity_metrics import match_notes  # noqa: E402

# model/data is gitignored -> absent in worktrees; anchor to the PRIMARY checkout
# (the part of the path before any `.worktrees/<branch>/` segment) where data lives.
_here = Path(__file__).resolve()
_parts = _here.parts
REPO = (Path(*_parts[:_parts.index(".worktrees")]) if ".worktrees" in _parts
        else _here.parents[4])
MIDI_DIR = REPO / "model/data/midi/percepiano"
LABELS = json.loads((REPO / "model/data/labels/composite/composite_labels.json").read_text())
SF2 = REPO / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = REPO / "model/data/weights/aria-amt"
OUT = REPO / "model/data/results/gc_dynamics_churn.json"

SR = 16000
GAIN = 0.5                 # byte-identical to the dynamics G-A/G-B render gain
K_VARIANTS = 6             # nuisance re-captures per clip (plus the base)
SNR_DB = 40.0              # additive white-noise SNR (inaudible)
GAIN_JITTER_DB = 0.5       # half-width of uniform gain jitter (sub-JND)
ONSET_WINDOW_S = 0.1       # per-note match tolerance
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

sys.path.insert(0, str(REPO / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def render(midi_path, wav_path):
    subprocess.run(
        ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav_path),
         str(SF2), str(midi_path)],
        check=True, capture_output=True,
    )


def perturb(audio, rng):
    """One nuisance-equivalent re-capture: sub-JND gain jitter + high-SNR white noise."""
    gain_db = rng.uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB)
    out = audio * (10.0 ** (gain_db / 20.0))
    rms = float(np.sqrt(np.mean(audio ** 2))) or 1e-9
    noise_rms = rms / (10.0 ** (SNR_DB / 20.0))
    out = out + rng.normal(0.0, noise_rms, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


print("Loading aria-amt...", flush=True)
from transcription import EndpointHandler  # noqa: E402
handler = EndpointHandler(path=str(WEIGHTS))
print("Model ready.\n", flush=True)


def transcribe(audio):
    notes, _ped = handler._transcribe(audio)
    return [{"pitch": int(n["pitch"]), "onset": float(n["onset"]),
             "offset": float(n["offset"]), "velocity": int(n["velocity"])} for n in notes]


labeled = [p for p in sorted(MIDI_DIR.glob("*.mid"))
           if LABELS.get(p.stem, {}).get("dynamics") is not None]
rng = np.random.default_rng(2026)
idx = sorted(rng.permutation(len(labeled))[:N])
paths = [labeled[i] for i in idx]

records = {}
with tempfile.TemporaryDirectory() as td:
    for i, mp in enumerate(paths):
        wav = Path(td) / "c.wav"
        try:
            render(mp, wav)
            audio, _ = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            base = transcribe(audio)
            base_vel = [n["velocity"] for n in base]
            # DETERMINISM CHECK: identical audio -> identical velocities (greedy decode).
            base2_vel = [n["velocity"] for n in transcribe(audio)]
            deterministic = base_vel == base2_vel

            variant_means = [float(np.mean(base_vel))] if base_vel else []
            per_note_deltas = []
            for k in range(K_VARIANTS):
                v = transcribe(perturb(audio, rng))
                if v:
                    variant_means.append(float(np.mean([n["velocity"] for n in v])))
                m = match_notes(base, v, onset_window_s=ONSET_WINDOW_S)
                per_note_deltas.extend(
                    float(amt["velocity"] - gt["velocity"]) for gt, amt in m["pairs"]
                )

            records[mp.stem] = {
                "n_base_notes": len(base),
                "deterministic": deterministic,
                "variant_mean_vels": [round(x, 3) for x in variant_means],
                "per_note_deltas": per_note_deltas,
            }
            sd = float(np.std(variant_means, ddof=1)) if len(variant_means) > 1 else float("nan")
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:26]:26s} n={len(base):3d} "
                  f"det={deterministic} stat_sd={sd:.3f} "
                  f"note_deltas={len(per_note_deltas)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - surface and keep going
            records[mp.stem] = {"error": repr(exc)}
            print(f"  [{i+1}/{len(paths)}] {mp.stem[:26]:26s} FAIL: {exc!r}", flush=True)
        OUT.write_text(json.dumps(records, indent=2))  # checkpoint every clip

meta = {
    "_meta": {
        "gain": GAIN, "k_variants": K_VARIANTS, "snr_db": SNR_DB,
        "gain_jitter_db": GAIN_JITTER_DB, "onset_window_s": ONSET_WINDOW_S,
        "n_clips": len([r for r in records.values() if "n_base_notes" in r]),
        "all_deterministic": all(r.get("deterministic", False)
                                 for r in records.values() if "n_base_notes" in r),
    }
}
records.update(meta)
OUT.write_text(json.dumps(records, indent=2))
print(f"\nwrote {meta['_meta']['n_clips']} clips -> {OUT}")
