# /// script
# requires-python = ">=3.11"
# dependencies = ["remotezip>=0.12.0","soundfile>=0.12.0","numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10"]
# ///
"""Stage 0 fetch: stream N MAESTRO test clips (first WINDOW_S) + truncate their ground-truth MIDI.

MAESTRO audio is a 108GB remote zip (range-readable); we stream only the leading bytes of each WAV
member (phase1's remotezip trick) instead of downloading it. Ground-truth MIDI is LOCAL (real
Disklavier velocities + sustain pedal) -- truncated to the same window so metrics compare like windows.

    uv run --script stage0_fetch.py [--n 30] [--window 30]
Writes audio/{seg}.wav (16k mono) + gt/{seg}.mid + manifest.json under the bench dir.
"""
import io, json, sys
from pathlib import Path

import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly
from remotezip import RemoteZip

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
MAESTRO = PRIMARY / "model/data/raw/maestro-v3.0.0"
META = MAESTRO / "maestro-v3.0.0.json"
ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
ZIP_PREFIX = "maestro-v3.0.0/"
BENCH = PRIMARY / "model/data/results/transcription_bench"
SR = 16000


def _patch_wav_prefix(raw: bytes) -> bytes:
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not RIFF/WAVE")
    pos, out = 12, bytearray(raw)
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
    raise ValueError("no data chunk in prefix")


def _truncate_midi(src: Path, dst: Path, window: float):
    """Keep notes with onset < window (clip offsets); keep CC/pedal with time < window."""
    pm = pretty_midi.PrettyMIDI(str(src))
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum)
        for n in inst.notes:
            if n.start < window:
                ni.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                                  start=n.start, end=min(n.end, window)))
        for cc in inst.control_changes:
            if cc.time < window:
                ni.control_changes.append(cc)
        out.instruments.append(ni)
    out.write(str(dst))


def _select(n: int) -> list[dict]:
    d = json.loads(META.read_text())
    test = [k for k, s in d["split"].items() if s == "test"]
    # composer-diverse stride so a subset is not one performer
    by_comp: dict[str, list[str]] = {}
    for k in test:
        by_comp.setdefault(d["canonical_composer"][k], []).append(k)
    order, comps = [], sorted(by_comp)
    while len(order) < len(test):
        for c in comps:
            if by_comp[c]:
                order.append(by_comp[c].pop(0))
    picked = order[:n]
    return [{"seg": f"mae{i:03d}", "audio": d["audio_filename"][k], "midi": d["midi_filename"][k],
             "composer": d["canonical_composer"][k], "title": d["canonical_title"][k]}
            for i, k in enumerate(picked)]


def main():
    argv = sys.argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 30
    window = float(argv[argv.index("--window") + 1]) if "--window" in argv else 30.0
    (BENCH / "audio").mkdir(parents=True, exist_ok=True)
    (BENCH / "gt").mkdir(parents=True, exist_ok=True)
    manifest = _select(n)
    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(manifest)} MAESTRO test clips; streaming first {window:.0f}s each", flush=True)

    ok = fail = 0
    with RemoteZip(ZIP_URL) as z:
        for m in manifest:
            wav_path = BENCH / "audio" / f"{m['seg']}.wav"
            gt_path = BENCH / "gt" / f"{m['seg']}.mid"
            if wav_path.exists() and gt_path.exists():
                ok += 1
                continue
            try:
                # ~30s of 44.1k stereo int16 ~= 5.3MB; grab 8MB to be safe, patch, trim
                with z.open(ZIP_PREFIX + m["audio"]) as h:
                    raw = h.read(8_000_000)
                audio, sr = sf.read(io.BytesIO(_patch_wav_prefix(raw)), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio[: int(sr * window)]
                if sr != SR:
                    audio = resample_poly(audio, SR, sr).astype("float32")
                sf.write(wav_path, audio, SR)
                _truncate_midi(MAESTRO / m["midi"], gt_path, window)
                ok += 1
                print(f"  ok {m['seg']}  {m['composer'][:28]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  FAIL {m['seg']}: {exc!r}", flush=True)
    print(f"\nfetched {ok}/{len(manifest)} ({fail} failed). Next: transcribe + metrics.", flush=True)


if __name__ == "__main__":
    main()
