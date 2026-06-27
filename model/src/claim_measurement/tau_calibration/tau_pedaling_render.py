# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.0.0", "numpy>=1.24.0", "safetensors>=0.4.0", "soundfile>=0.12.0",
#     "numba>=0.59.0", "llvmlite>=0.42.0", "pretty_midi>=0.2.10",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
#     "aria @ git+https://github.com/EleutherAI/aria.git",
# ]
# ///
"""Front-4 pedaling tau calibration set (#101): render N natural (uncorrupted) PercePiano
clips at fixed gain -> aria-amt -> sustain time-on-fraction, paired with the composite
pedaling perceptual label by stem. Broad random sample across the FULL pedaling spectrum
(incl. dry clips -- the low-tail under-pedal anomalies). No corruption.

Run:  CRESCEND_DEVICE=auto uv run --script tau_pedaling_render.py [N]
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf, pretty_midi

REPO = Path(__file__).resolve().parents[4]
MIDI_DIR = REPO / "model/data/midi/percepiano"
LABELS = json.loads((REPO / "model/data/labels/composite/composite_labels.json").read_text())
SF2 = REPO / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = REPO / "model/data/weights/aria-amt"
OUT = REPO / "model/data/results/tau_cal_pedaling.json"
SR, GAIN = 16000, 0.5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 180
sys.path.insert(0, str(REPO / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def cc64_spans(pm):
    ev = sorted(((c.time, c.value) for inst in pm.instruments for c in inst.control_changes if c.number == 64), key=lambda e: e[0])
    spans, on = [], None
    for t, v in ev:
        if v >= 64 and on is None: on = t
        elif v < 64 and on is not None: spans.append((on, t)); on = None
    if on is not None: spans.append((on, pm.get_end_time()))
    return spans


def spans_from_events(events):
    ev = sorted(((float(e["time"]), int(e["value"])) for e in events), key=lambda e: e[0])
    spans, on = [], None
    for t, v in ev:
        if v >= 64 and on is None: on = t
        elif v < 64 and on is not None: spans.append((on, t)); on = None
    if on is not None and ev: spans.append((on, ev[-1][0]))
    return spans


def on_fraction(spans, total):
    return float(min(sum(max(0.0, b - a) for a, b in spans) / total, 1.0)) if total > 0 else 0.0


def render(mp, wav):
    subprocess.run(["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav), str(SF2), str(mp)],
                   check=True, capture_output=True)


print("Loading aria-amt...", flush=True)
from transcription import EndpointHandler
handler = EndpointHandler(path=str(WEIGHTS))
print("Model ready.\n", flush=True)

labeled = [p for p in sorted(MIDI_DIR.glob("*.mid")) if LABELS.get(p.stem, {}).get("pedaling") is not None]
rng = np.random.default_rng(2024)
paths = [labeled[i] for i in sorted(rng.permutation(len(labeled))[:N])]

out = {}
with tempfile.TemporaryDirectory() as td:
    for i, mp in enumerate(paths):
        wav = Path(td) / "c.wav"
        try:
            base = pretty_midi.PrettyMIDI(str(mp))
            total = base.get_end_time()
            true_frac = on_fraction(cc64_spans(base), total)
            render(mp, wav)
            audio, _ = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1: audio = audio.mean(axis=1)
            _n, pedals = handler._transcribe(audio)
            amt_frac = on_fraction(spans_from_events(pedals), total)
        except Exception as exc:
            print(f"  [{i+1}/{N}] {mp.stem[:24]} FAIL: {exc}", flush=True); continue
        out[mp.stem] = {"amt_on_fraction": round(amt_frac, 4), "true_midi_frac": round(true_frac, 4),
                        "composite_pedaling": LABELS[mp.stem]["pedaling"]}
        OUT.write_text(json.dumps(out, indent=2))  # incremental checkpoint (survives kill)
        print(f"  [{i+1}/{N}] {mp.stem[:24]:24s} amt={amt_frac:.3f} true={true_frac:.3f} label={LABELS[mp.stem]['pedaling']:.3f}", flush=True)

OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {len(out)} clips -> {OUT}")
