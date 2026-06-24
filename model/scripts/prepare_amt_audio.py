# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare per-segment WAVs for AMT transcription (#72), one tier at a time.

Produces `{wav_dir}/{seg_id}.wav` so that `extract_amt_midi.py` can turn each into
`{seg_id}.mid`. seg_ids are exactly the keys the Aria symbolic path expects:
PercePiano label keys (T1) and competition `segment_id`s (T2).

Tiers:
  t1       PercePiano: render each ground-truth MIDI -> WAV via fluidsynth.
           NOTE (timbre caveat): the original PercePiano audio was Pianoteq-rendered
           and is gone locally (and absent from R2). A general-MIDI soundfont render
           is a reproducible *proxy* whose timbre differs from real/Pianoteq piano;
           AMT (trained on real acoustic piano) may transcribe it slightly worse, so
           the D4 "AMT good enough for Aria" check (#77) should treat T1 numbers as a
           lower bound and, if Pianoteq/real audio is ever recovered, re-render.
  cliburn  Cliburn 2022: rclone 82 full recordings from R2 -> ffmpeg slice per segment.
  chopin   Chopin 2021: yt-dlp each recording's source_url -> ffmpeg slice per segment.

External tools (must be on PATH): fluidsynth (t1), ffmpeg (all), yt-dlp (chopin),
rclone with an `r2:` remote (cliburn).

Usage (run pointing at the PRIMARY checkout's data, since worktrees lack gitignored data):
    uv run model/scripts/prepare_amt_audio.py --tier t1 \
        --data-root /abs/path/to/model/data --soundfont /abs/path/to/piano.sf2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

FLUIDR3_URL = "https://github.com/urish/cinto/raw/master/media/FluidR3%20GM.sf2"


def log(msg: str) -> None:
    print(json.dumps({"ts": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)


def run(cmd: list[str]) -> None:
    """Run a subprocess, raising with captured stderr on failure (explicit, no fallback)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )


def ensure_soundfont(arg: str | None, cache_dir: Path) -> Path:
    """Resolve the soundfont path, downloading FluidR3 GM once if none supplied."""
    if arg:
        p = Path(arg)
        if not p.exists():
            raise FileNotFoundError(f"--soundfont not found: {p}")
        return p
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf = cache_dir / "FluidR3_GM.sf2"
    if not sf.exists():
        log(f"No --soundfont given; downloading FluidR3 GM (MIT) -> {sf}")
        tmp = sf.with_suffix(".sf2.tmp")
        urllib.request.urlretrieve(FLUIDR3_URL, tmp)
        tmp.replace(sf)
        log(f"Soundfont ready ({sf.stat().st_size // 1_000_000} MB).")
    return sf


def prepare_t1(data_root: Path, wav_dir: Path, soundfont: Path) -> tuple[int, int]:
    midi_dir = data_root / "midi" / "percepiano"
    labels = json.loads((data_root / "labels" / "percepiano" / "labels.json").read_text())
    seg_ids = sorted(labels.keys())
    log(f"T1: {len(seg_ids)} segments, rendering with {soundfont.name}")
    done = skip = 0
    for i, seg_id in enumerate(seg_ids):
        out = wav_dir / f"{seg_id}.wav"
        if out.exists():
            skip += 1
            continue
        midi = midi_dir / f"{seg_id}.mid"
        if not midi.exists():
            raise FileNotFoundError(f"T1 ground-truth MIDI missing: {midi}")
        tmp = out.with_suffix(".wav.tmp")
        run(["fluidsynth", "-ni", "-g", "0.8", "-r", "44100", "-F", str(tmp),
             str(soundfont), str(midi)])
        tmp.replace(out)
        done += 1
        if (i + 1) % 100 == 0:
            log(f"T1 render {i + 1}/{len(seg_ids)} (done={done} skip={skip})")
    return done, skip


def _slice_recordings(
    metadata_path: Path, recordings: dict[str, Path], wav_dir: Path
) -> tuple[int, int]:
    """Slice each segment out of its full recording via ffmpeg. Resumable."""
    segs = [json.loads(l) for l in metadata_path.read_text().splitlines() if l.strip()]
    done = skip = 0
    for i, s in enumerate(segs):
        seg_id = s["segment_id"]
        out = wav_dir / f"{seg_id}.wav"
        if out.exists():
            skip += 1
            continue
        rec = recordings.get(s["recording_id"])
        if rec is None or not rec.exists():
            raise FileNotFoundError(
                f"recording for {seg_id} not acquired: {s['recording_id']}"
            )
        start = float(s["segment_start"])
        dur = float(s["segment_end"]) - start
        tmp = out.with_suffix(".wav.tmp")
        # `-f wav` is required: ffmpeg infers the output muxer from the file
        # extension, and the atomic-write `.wav.tmp` name is not recognized.
        run(["ffmpeg", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(rec),
             "-ac", "1", "-ar", "16000", "-f", "wav", str(tmp)])
        tmp.replace(out)
        done += 1
        if (i + 1) % 200 == 0:
            log(f"slice {i + 1}/{len(segs)} (done={done} skip={skip})")
    return done, skip


def prepare_cliburn(data_root: Path, wav_dir: Path, rec_cache: Path) -> tuple[int, int]:
    meta = data_root / "manifests" / "competition" / "cliburn_2022" / "metadata.jsonl"
    rec_cache.mkdir(parents=True, exist_ok=True)
    rec_ids = sorted({json.loads(l)["recording_id"] for l in meta.read_text().splitlines() if l.strip()})
    log(f"Cliburn: ensuring {len(rec_ids)} recordings from R2 -> {rec_cache}")
    recordings: dict[str, Path] = {}
    for rid in rec_ids:
        dst = rec_cache / f"{rid}.wav"
        if not dst.exists():
            run(["rclone", "copyto",
                 f"r2:crescendai-bucket/raw/competition/cliburn_2022/audio/{rid}.wav",
                 str(dst)])
            log(f"fetched {rid}")
        recordings[rid] = dst
    return _slice_recordings(meta, recordings, wav_dir)


def prepare_chopin(data_root: Path, wav_dir: Path, rec_cache: Path) -> tuple[int, int]:
    meta = data_root / "manifests" / "competition" / "metadata.jsonl"
    rec_cache.mkdir(parents=True, exist_ok=True)
    segs = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    url_by_rec: dict[str, str] = {}
    for s in segs:
        url_by_rec.setdefault(s["recording_id"], s["source_url"])
    log(f"Chopin: ensuring {len(url_by_rec)} recordings via yt-dlp -> {rec_cache}")
    recordings: dict[str, Path] = {}
    for rid, url in sorted(url_by_rec.items()):
        dst = rec_cache / f"{rid}.wav"
        if not dst.exists():
            run(["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
                 "-o", str(rec_cache / f"{rid}.%(ext)s"), url])
            if not dst.exists():
                raise RuntimeError(f"yt-dlp did not produce {dst} for {url}")
            log(f"downloaded {rid}")
        recordings[rid] = dst
    return _slice_recordings(meta, recordings, wav_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=["t1", "cliburn", "chopin"])
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="model/data root (point at the PRIMARY checkout when run from a worktree).",
    )
    ap.add_argument("--wav-dir", default=None, help="Override staging dir for {seg_id}.wav.")
    ap.add_argument("--soundfont", default=None, help="t1 only: path to a .sf2 (default: fetch FluidR3 GM).")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")
    wav_dir = Path(args.wav_dir) if args.wav_dir else data_root / "raw" / "amt_audio" / args.tier
    wav_dir.mkdir(parents=True, exist_ok=True)
    rec_cache = data_root / "raw" / "amt_recordings" / args.tier

    if args.tier == "t1":
        sf = ensure_soundfont(args.soundfont, data_root / "weights" / "soundfonts")
        done, skip = prepare_t1(data_root, wav_dir, sf)
    elif args.tier == "cliburn":
        done, skip = prepare_cliburn(data_root, wav_dir, rec_cache)
    else:
        done, skip = prepare_chopin(data_root, wav_dir, rec_cache)

    log(f"FINISHED tier={args.tier} prepared={done} skipped={skip} wav_dir={wav_dir}")


if __name__ == "__main__":
    main()
