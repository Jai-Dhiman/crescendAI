# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
#     "safetensors>=0.4.0",
#     "soundfile>=0.12.0",
#     "numba>=0.59.0",
#     "llvmlite>=0.42.0",
#     "pretty_midi>=0.2.10",
#     "remotezip>=0.12.0",
#     "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
#     "aria @ git+https://github.com/EleutherAI/aria.git",
# ]
# ///
"""Phase 1 -- cross-render feature stability (MIREX 2026 difficulty, #104).

The kill-gate experiment for the symbolic-from-AMT approach. For each MAESTRO
segment we obtain the difficulty features THREE ways and measure whether they
agree across renderings:

  oracle     = features from the ground-truth MAESTRO MIDI (cropped to the window)
  AMT-real   = features from aria-amt transcription of the REAL human audio
  AMT-synth  = features from aria-amt transcription of a fluidsynth render of the
               same (cropped) MIDI -- our stand-in for MIREX's "synthesized rendering"

Reported per feature, across the N segments:
  cross-render  = Spearman corr(AMT-real, AMT-synth)   <- the decision number
  fidelity_real = Spearman corr(oracle,   AMT-real)
  fidelity_synth= Spearman corr(oracle,   AMT-synth)

Kill-gate: a feature with cross-render rho > ~0.9 is render-stable and kept; below
that it is render-fragile. If MOST features are unstable, symbolic-from-AMT is
compromised for this task (that IS the independently-scored anti-cheat failure mode).

MAESTRO audio is not stored locally (200GB); we stream only the first ~30s of each
WAV out of the remote zip via remotezip range requests (deflate members inflate
lazily, so a 30s prefix costs a few MB, not the full member). aria-amt truncates at
30s regardless. MIDI + soundfont + weights are read from the PRIMARY checkout.

Run (data read from the primary checkout, results written there):
    CRESCEND_DEVICE=auto uv run --script phase1_cross_render.py [N] [--smoke]
"""
import csv as csvmod
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from remotezip import RemoteZip
from scipy import signal as scipy_signal
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_features import extract_difficulty_features  # noqa: E402

# model/data is gitignored -> absent in worktrees; anchor to the primary checkout.
PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
MAESTRO_DIR = PRIMARY / "model/data/raw/maestro-v3.0.0"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
SF2 = PRIMARY / "model/data/soundfonts/MuseScore_General.sf3"
WEIGHTS = PRIMARY / "model/data/weights/aria-amt"
OUT = PRIMARY / "model/data/results/mirex_phase1_cross_render.json"

MAESTRO_ZIP_URL = ("https://storage.googleapis.com/magentadata/datasets/maestro/"
                   "v3.0.0/maestro-v3.0.0.zip")
ZIP_PREFIX = "maestro-v3.0.0/"

SR = 16000                # aria-amt + fluidsynth sample rate
GAIN = 0.5                # fluidsynth gain (byte-identical to the amt_fidelity render)
AMT_WINDOW_S = 30.0       # aria-amt hard-truncates here
GUARD_S = 1.0             # drop notes near the truncation boundary
CUTOFF_S = AMT_WINDOW_S - GUARD_S   # onset cutoff for every feature computation
PREFIX_BYTES = 9 * 1024 * 1024      # ~9MB decompressed ~= 50s of 44.1k stereo 16bit
MIN_DURATION_S = 45.0     # need a full 30s window with headroom

FEATURES = ["pitch_entropy", "pitch_lz_complexity", "polyphony", "pitch_range"]


def select_segments(n: int) -> list[dict]:
    """Deterministic composer-spread sample of MAESTRO rows with a full window.

    Round-robin across composers (sorted) so N segments are not dominated by one
    composer -- a light guard toward the composer-disjoint spirit of the task.
    """
    with open(MAESTRO_CSV, newline="") as fh:
        rows = [r for r in csvmod.DictReader(fh)
                if float(r["duration"]) >= MIN_DURATION_S]
    by_composer: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: (r["canonical_composer"], r["midi_filename"])):
        by_composer.setdefault(r["canonical_composer"], []).append(r)
    composers = sorted(by_composer)
    picked: list[dict] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for c in composers:
            if depth < len(by_composer[c]):
                picked.append(by_composer[c][depth])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        depth += 1
    return picked


def fetch_real_audio_16k(audio_filename: str) -> np.ndarray:
    """Stream the first ~30s of a MAESTRO WAV from the remote zip, mono, 16kHz."""
    member = ZIP_PREFIX + audio_filename
    with RemoteZip(MAESTRO_ZIP_URL) as z:
        with z.open(member) as handle:
            raw = handle.read(PREFIX_BYTES)
    wav = _patch_wav_prefix(raw)
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio[: int(sr * AMT_WINDOW_S)]
    if sr != SR:
        audio = scipy_signal.resample_poly(audio, SR, sr).astype(np.float32)
    return audio


def _patch_wav_prefix(raw: bytes) -> bytes:
    """Rewrite RIFF/data chunk sizes to the bytes actually present so libsndfile
    reads the truncated prefix instead of erroring on the declared full length."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("remote member is not a RIFF/WAVE file")
    pos = 12
    out = bytearray(raw)
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        csize = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = pos + 8
        if cid == b"data":
            keep = min(csize, len(raw) - body)
            out[pos + 4:pos + 8] = keep.to_bytes(4, "little")
            out = out[:body + keep]
            out[4:8] = (len(out) - 8).to_bytes(4, "little")
            return bytes(out)
        pos = body + csize + (csize & 1)
    raise ValueError("no data chunk found in the fetched prefix")


def load_and_crop_midi(midi_path: Path) -> tuple[list[dict], pretty_midi.PrettyMIDI]:
    """GT notes with onset <= CUTOFF (oracle substrate) + a cropped PrettyMIDI to render."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    cropped = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for src in pm.instruments:
        if src.is_drum:
            continue
        for note in src.notes:
            if note.start <= CUTOFF_S:
                notes.append({"pitch": int(note.pitch), "onset": float(note.start),
                              "offset": float(note.end), "velocity": int(note.velocity)})
                inst.notes.append(pretty_midi.Note(
                    velocity=note.velocity, pitch=note.pitch,
                    start=note.start, end=min(note.end, AMT_WINDOW_S + 1.0)))
    cropped.instruments.append(inst)
    notes.sort(key=lambda d: d["onset"])
    return notes, cropped


def render_16k(cropped: pretty_midi.PrettyMIDI) -> np.ndarray:
    """fluidsynth render of the cropped MIDI at 16kHz (the 'synthesized rendering')."""
    with tempfile.TemporaryDirectory() as td:
        mid = Path(td) / "c.mid"
        wav = Path(td) / "c.wav"
        cropped.write(str(mid))
        subprocess.run(
            ["fluidsynth", "-ni", "-g", str(GAIN), "-r", str(SR), "-F", str(wav),
             str(SF2), str(mid)],
            check=True, capture_output=True,
        )
        audio, _ = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio[: int(SR * (AMT_WINDOW_S + 1.0))]


def transcribe(handler, audio: np.ndarray) -> list[dict]:
    """aria-amt -> note dicts with onset <= CUTOFF (same window as oracle)."""
    amt_notes, _ped = handler._transcribe(audio)
    return [{"pitch": int(n["pitch"]), "onset": float(n["onset"]),
             "offset": float(n["offset"]), "velocity": int(n["velocity"])}
            for n in amt_notes if float(n["onset"]) <= CUTOFF_S]


def corr(a: list[float], b: list[float]) -> dict:
    """Spearman (rank, primary) + Pearson (linear, reference) with n."""
    if len(a) < 3:
        return {"spearman": None, "pearson": None, "n": len(a)}
    sp = stats.spearmanr(a, b).statistic
    pe = stats.pearsonr(a, b).statistic
    return {"spearman": None if np.isnan(sp) else round(float(sp), 4),
            "pearson": None if np.isnan(pe) else round(float(pe), 4),
            "n": len(a)}


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv
    n = int(argv[0]) if argv else (2 if smoke else 24)

    print("Loading aria-amt...", flush=True)
    sys.path.insert(0, str(PRIMARY / "apps/inference/amt"))
    os.environ.setdefault("CRESCEND_DEVICE", "auto")
    from transcription import EndpointHandler  # noqa: E402
    handler = EndpointHandler(path=str(WEIGHTS))
    print("Model ready.\n", flush=True)

    segments = select_segments(n)
    print(f"selected {len(segments)} segments (>= {MIN_DURATION_S}s), N target {n}\n", flush=True)

    per_segment = {}
    vals = {f: {"oracle": [], "amt_real": [], "amt_synth": []} for f in FEATURES}

    for i, row in enumerate(segments):
        title = f"{row['canonical_composer']} - {row['canonical_title']}"[:48]
        try:
            midi_path = MAESTRO_DIR / row["midi_filename"]
            gt_notes, cropped = load_and_crop_midi(midi_path)
            if len(gt_notes) < 5:
                raise ValueError(f"too few GT notes in window ({len(gt_notes)})")

            real_audio = fetch_real_audio_16k(row["audio_filename"])
            synth_audio = render_16k(cropped)

            amt_real_notes = transcribe(handler, real_audio)
            amt_synth_notes = transcribe(handler, synth_audio)
            if len(amt_real_notes) < 5 or len(amt_synth_notes) < 5:
                raise ValueError("too few AMT notes (real=%d synth=%d)"
                                 % (len(amt_real_notes), len(amt_synth_notes)))

            f_oracle = extract_difficulty_features(gt_notes)
            f_real = extract_difficulty_features(amt_real_notes)
            f_synth = extract_difficulty_features(amt_synth_notes)

            for f in FEATURES:
                vals[f]["oracle"].append(f_oracle[f])
                vals[f]["amt_real"].append(f_real[f])
                vals[f]["amt_synth"].append(f_synth[f])

            per_segment[row["audio_filename"]] = {
                "title": title, "n_gt": len(gt_notes),
                "n_amt_real": len(amt_real_notes), "n_amt_synth": len(amt_synth_notes),
                "oracle": {k: round(v, 4) for k, v in f_oracle.items()},
                "amt_real": {k: round(v, 4) for k, v in f_real.items()},
                "amt_synth": {k: round(v, 4) for k, v in f_synth.items()}}
            print(f"  [{i+1}/{len(segments)}] {title:48s} "
                  f"gt={len(gt_notes):3d} real={len(amt_real_notes):3d} "
                  f"synth={len(amt_synth_notes):3d}", flush=True)
        except Exception as exc:  # noqa: BLE001 - surface and keep going
            per_segment[row["audio_filename"]] = {"title": title, "error": repr(exc)}
            print(f"  [{i+1}/{len(segments)}] {title:48s} FAIL: {exc!r}", flush=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"per_segment": per_segment}, indent=2))

    # ---- per-feature verdict -------------------------------------------------
    n_ok = len(vals[FEATURES[0]]["oracle"])
    feature_table = {}
    for f in FEATURES:
        cross = corr(vals[f]["amt_real"], vals[f]["amt_synth"])
        feature_table[f] = {
            "cross_render": cross,
            "fidelity_real": corr(vals[f]["oracle"], vals[f]["amt_real"]),
            "fidelity_synth": corr(vals[f]["oracle"], vals[f]["amt_synth"]),
            "render_stable": (cross["spearman"] is not None and cross["spearman"] > 0.9),
        }
    n_stable = sum(1 for f in FEATURES if feature_table[f]["render_stable"])
    verdict = ("PASS: most features render-stable" if n_stable > len(FEATURES) / 2
               else "KILL-SIGNAL: majority of features render-fragile")

    summary = {"n_segments_ok": n_ok, "n_stable_features": n_stable,
               "feature_table": feature_table, "verdict": verdict}
    OUT.write_text(json.dumps({"summary": summary, "per_segment": per_segment}, indent=2))

    print("\n=== PHASE 1: CROSS-RENDER FEATURE STABILITY ===")
    print(f"  segments OK: {n_ok}   render-stable features: {n_stable}/{len(FEATURES)}")
    print(f"  {'feature':22s} {'cross-render':>13s} {'fid(real)':>11s} {'fid(synth)':>11s}  stable")
    for f in FEATURES:
        t = feature_table[f]
        cr = t["cross_render"]["spearman"]
        fr = t["fidelity_real"]["spearman"]
        fs = t["fidelity_synth"]["spearman"]
        print(f"  {f:22s} {str(cr):>13s} {str(fr):>11s} {str(fs):>11s}  "
              f"{'YES' if t['render_stable'] else 'no'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
