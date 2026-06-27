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
"""Pedal G-A stage 1: construction-known CC64-span corruption -> render -> AMT (#101 front-3).

For N base PercePiano clips that contain sustain pedal, rewrite the CC64 stream to make
the pedal SPARSE (shorten each on-span x0.4), NEUTRAL (unchanged) or DENSE (extend each
on-span to fill 60% of the gap to the next on). All three are true, construction-known
on-fraction changes. Render at fixed gain, transcribe with aria-amt, and record the
MIDI-true and AMT time-on-fraction per (clip, level) plus the raw AMT pedal events, so
``amt_pedaling_ga_metrics`` can (a) confirm AMT preserves on-fraction ORDERING (the
gating risk -- pedal transcription is noisier than velocity and saturates when wet) and
(b) feed the REAL PedalingMeasurer + frozen route_verdict.

Run:  CRESCEND_DEVICE=auto uv run --script amt_pedaling_ga_render.py [N] [out.json]
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
SPARSE_SCALE = 0.4      # "scaled" mode: shorten each on-span
DENSE_GAP_FILL = 0.6    # "scaled" mode: extend each on-span to fill this fraction of the next gap
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else (REPO / "model/data/results/ga_pedal_onfraction.json")
# Corruption strength. "maximal" (the G-A gate, #101 front-3): sparse=remove ALL pedal
# (true dry, frac->0), dense=full pedal down (frac->1) -- the strongest construction-known
# swing, needed because AMT pedal SATURATES (~0.55 ceiling), so a gentle dense swing is
# invisible. "scaled": gentler x0.4 / gap-fill swing (kept for reference). The maximal
# run is what proves under-pedal G-A and exposes the over-pedal saturation ceiling.
MODE = sys.argv[3] if len(sys.argv) > 3 else "maximal"

sys.path.insert(0, str(REPO / "apps/inference/amt"))
os.environ.setdefault("CRESCEND_DEVICE", "auto")


def cc64_spans(pm: pretty_midi.PrettyMIDI) -> list[tuple[float, float]]:
    events = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64:
                events.append((cc.time, cc.value))
    events.sort(key=lambda e: e[0])
    spans, on = [], None
    for t, v in events:
        if v >= 64 and on is None:
            on = t
        elif v < 64 and on is not None:
            spans.append((on, t)); on = None
    end = pm.get_end_time()
    if on is not None:
        spans.append((on, end))
    return spans


def cc64_spans_from_events(events: list[dict]) -> list[tuple[float, float]]:
    evs = sorted(((float(e["time"]), int(e["value"])) for e in events), key=lambda e: e[0])
    spans, on = [], None
    for t, v in evs:
        if v >= 64 and on is None:
            on = t
        elif v < 64 and on is not None:
            spans.append((on, t)); on = None
    if on is not None and evs:
        spans.append((on, evs[-1][0]))
    return spans


def on_fraction(spans, total_dur):
    if total_dur <= 0:
        return 0.0
    on = sum(max(0.0, b - a) for a, b in spans)
    return float(min(on / total_dur, 1.0))


def rewrite_cc64(pm, spans):
    for inst in pm.instruments:
        inst.control_changes = [c for c in inst.control_changes if c.number != 64]
    target = pm.instruments[0]
    for a, b in spans:
        target.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=a))
        target.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=b))
    target.control_changes.sort(key=lambda c: c.time)


def corrupt(spans, level, total_dur):
    if level == "neutral":
        return list(spans)
    if MODE == "maximal":
        if level == "sparse":
            return []                       # remove ALL pedal -> true dry (frac->0)
        if level == "dense":
            return [(0.0, total_dur)]        # full pedal down (frac->1)
    elif MODE == "scaled":
        if level == "sparse":
            return [(a, a + (b - a) * SPARSE_SCALE) for a, b in spans]
        if level == "dense":
            out = []
            for i, (a, b) in enumerate(spans):
                next_on = spans[i + 1][0] if i + 1 < len(spans) else total_dur
                new_b = b + (next_on - b) * DENSE_GAP_FILL
                out.append((a, max(b, min(new_b, next_on))))
            return out
    raise ValueError(f"mode={MODE!r} level={level!r}")


def render(pm, wav_path):
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

labeled = [p for p in sorted(MIDI_DIR.glob("*.mid"))
           if LABELS.get(p.stem, {}).get("dynamics") is not None]
rng = np.random.default_rng(123)
order = rng.permutation(len(labeled))

out, picked = {}, 0
with tempfile.TemporaryDirectory() as td:
    for ix in order:
        if picked >= N:
            break
        mp = labeled[int(ix)]
        base = pretty_midi.PrettyMIDI(str(mp))
        spans = cc64_spans(base)
        total_dur = base.get_end_time()
        if not spans or on_fraction(spans, total_dur) < 0.02:
            continue
        rec = {}
        for level in ("sparse", "neutral", "dense"):
            new_spans = corrupt(spans, level, total_dur)
            true_frac = on_fraction(new_spans, total_dur)
            pm = pretty_midi.PrettyMIDI(str(mp))
            rewrite_cc64(pm, new_spans)
            wav = Path(td) / "c.wav"
            pedals = []
            try:
                render(pm, wav)
                audio, _ = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                _notes, pedals = handler._transcribe(audio)
                amt_frac = on_fraction(cc64_spans_from_events(pedals), total_dur)
            except Exception as exc:
                print(f"  {mp.stem[:24]} {level} FAIL: {exc}", flush=True)
                amt_frac = None
            rec[level] = {"true_frac": round(true_frac, 4),
                          "amt_frac": round(amt_frac, 4) if amt_frac is not None else None,
                          "amt_pedal_events": pedals if amt_frac is not None else []}
        out[mp.stem] = {"total_dur": round(total_dur, 4), "levels": rec}
        picked += 1
        OUT.write_text(json.dumps(out, indent=2))  # incremental checkpoint (survives kill)
        t = {k: rec[k]["true_frac"] for k in rec}
        a = {k: rec[k]["amt_frac"] for k in rec}
        print(f"  [{picked}/{N}] {mp.stem[:24]:24s} TRUE s/n/d={t['sparse']}/{t['neutral']}/{t['dense']}"
              f"  AMT s/n/d={a['sparse']}/{a['neutral']}/{a['dense']}", flush=True)

OUT.write_text(json.dumps(out, indent=2))
print(f"\nwrote {len(out)} clips -> {OUT}")
