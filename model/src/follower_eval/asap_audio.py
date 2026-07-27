# model/src/follower_eval/asap_audio.py
"""MAESTRO-audio source for Track A (issue #133).

Track A scores the follower against ASAP's human-verified beat alignment. Run on
ASAP's performance MIDI it is an ISOLATION result -- it measures the matcher on
clean notes. The production question is what happens when the notes come from
audio through the real transcriber, so this module supplies the same
performances as **MAESTRO audio -> Transkun**, keeping ASAP's alignment as the
answer key. The follower and the metric are unchanged; only the note source moves.

WHY A DERIVED SHIFT (do not use metadata `start`):
ASAP's performances are trimmed out of longer MAESTRO recordings, so a
transcription is in MAESTRO's clock while the truth is in ASAP's. The metadata
`start` column looks like the offset but is NOT reliable -- for
`Bach/Fugue/bwv_858/Zhang01M.mid` it reads 90.33 while the true offset is 89.83.
Half a second is a large systematic error at beat resolution, and it would have
silently corrupted the number rather than failing. Both MIDIs are on disk, so we
DERIVE the offset by matching notes and refuse to proceed below
``MIN_MATCH_FRAC`` agreement.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import partitura as pa

DEFAULT_ASAP_ROOT = Path("data/raw/asap")
DEFAULT_MAESTRO_MIDI_ROOT = Path("data/raw/maestro-v3.0.0")
DEFAULT_MAESTRO_AUDIO_ROOT = Path("data/raw/maestro/files")
DEFAULT_CACHE = Path("data/evals/asap_audio_bundles")

TRANSCRIBER_ID = "transkun"   # production /transcribe engine (#128)
SAMPLE_RATE = 16000           # production preprocessing: 16 kHz mono
ONSET_TOL = 0.01              # two notes are "the same note" within 10 ms
MIN_MATCH_FRAC = 0.98         # below this the shift is not trustworthy -> raise
PAD_SEC = 2.0                 # lead-in/out kept around the clip when slicing


class AsapAudioError(RuntimeError):
    """Loud failure: no MAESTRO link, missing audio, an underivable time offset,
    or a transcription that came back empty. Never a silent fallback to MIDI --
    that would report a clean-MIDI number as if it came through audio."""


@dataclass(frozen=True)
class Shift:
    seconds: float          # asap_time = maestro_time - seconds
    match_frac: float       # fraction of ASAP notes explained at this shift
    metadata_start: float   # what the CSV claimed, for the record


def load_metadata(asap_root: Path) -> dict[str, dict]:
    """ASAP metadata.csv keyed by the piece key the eval uses (the
    ``midi_performance`` path, which is also the asap_annotations.json key)."""
    with open(asap_root / "metadata.csv") as fh:
        return {r["midi_performance"]: r for r in csv.DictReader(fh)}


def maestro_paths(row: dict, midi_root: Path, audio_root: Path) -> tuple[Path, Path]:
    """(maestro_midi, maestro_audio) for an ASAP row. MAESTRO's MIDI and audio
    live in separate local trees, so both roots are explicit.

    Raises:
        AsapAudioError: the row has no MAESTRO link, or a file is absent.
    """
    link = row.get("maestro_audio_performance") or ""
    if not link:
        raise AsapAudioError(f"{row['midi_performance']}: no maestro_audio_performance link")
    rel = Path(link).relative_to("{maestro}")
    audio = audio_root / rel
    midi = midi_root / Path(row["maestro_midi_performance"]).relative_to("{maestro}")
    if not audio.exists():
        raise AsapAudioError(f"missing MAESTRO audio {audio} (not downloaded)")
    if not midi.exists():
        raise AsapAudioError(f"missing MAESTRO midi {midi}")
    return midi, audio


def derive_shift(asap_midi: Path, maestro_midi: Path, metadata_start: float) -> Shift:
    """The offset with ``asap_time = maestro_time - shift``, derived from the
    notes themselves rather than trusted from metadata.

    Candidate offsets come from same-pitch note pairs; the modal candidate wins
    and is then scored against every ASAP note. Raises if the winner explains
    less than ``MIN_MATCH_FRAC`` of them -- an underivable offset must stop the
    run, not quietly shift the truth.
    """
    a = pa.load_performance_midi(str(asap_midi)).note_array()
    m = pa.load_performance_midi(str(maestro_midi)).note_array()
    if len(a) == 0 or len(m) == 0:
        raise AsapAudioError(f"empty note array: {asap_midi} ({len(a)}) / {maestro_midi} ({len(m)})")

    by_pitch: dict[int, list[float]] = {}
    for t, p in zip(m["onset_sec"], m["pitch"]):
        by_pitch.setdefault(int(p), []).append(float(t))

    # vote on the offset using the first notes of the clip (cheap, unambiguous)
    votes: Counter[float] = Counter()
    for t, p in list(zip(a["onset_sec"], a["pitch"]))[:40]:
        for mt in by_pitch.get(int(p), []):
            votes[round(mt - float(t), 3)] += 1
    if not votes:
        raise AsapAudioError(f"no shared pitches between {asap_midi} and {maestro_midi}")

    best, best_frac = 0.0, -1.0
    for cand, _ in votes.most_common(8):
        hit = sum(1 for t, p in zip(a["onset_sec"], a["pitch"])
                  if any(abs(mt - cand - float(t)) < ONSET_TOL for mt in by_pitch.get(int(p), [])))
        frac = hit / len(a)
        if frac > best_frac:
            best, best_frac = cand, frac
    if best_frac < MIN_MATCH_FRAC:
        raise AsapAudioError(
            f"{asap_midi.name}: could not derive a time offset against {maestro_midi.name} "
            f"(best shift {best:.3f}s explains only {best_frac:.1%} of {len(a)} notes; "
            f"metadata start was {metadata_start:.3f}). Refusing to guess.")
    return Shift(seconds=float(best), match_frac=round(best_frac, 4),
                 metadata_start=metadata_start)


def clip_bounds(asap_midi: Path, shift: float) -> tuple[float, float]:
    """The [start, end] slice of the MAESTRO recording covering this ASAP
    performance, padded, in MAESTRO seconds."""
    a = pa.load_performance_midi(str(asap_midi)).note_array()
    lo = float(min(a["onset_sec"])) + shift - PAD_SEC
    hi = float(max(a["onset_sec"] + a["duration_sec"])) + shift + PAD_SEC
    return max(0.0, lo), hi


def slice_audio(src: Path, dst: Path, lo: float, hi: float) -> Path:
    """Cut [lo, hi] out of a MAESTRO recording as 16 kHz mono WAV -- the same
    preprocessing the production path feeds the transcriber."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{lo:.3f}", "-to", f"{hi:.3f}",
           "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise AsapAudioError(f"ffmpeg failed slicing {src.name} [{lo:.2f},{hi:.2f}]: {r.stderr[-400:]}")
    return dst


def bundle_path(cache_dir: Path, asap_piece: str) -> Path:
    return cache_dir / (asap_piece.replace("/", "__").removesuffix(".mid") + ".json")


def build_one(asap_piece: str, asap_root: Path, midi_root: Path, audio_root: Path,
              cache_dir: Path, transcribe_wav, meta: dict[str, dict],
              force: bool = False) -> dict:
    """Transcribe one ASAP performance from MAESTRO audio into a cached bundle
    whose note onsets are already in ASAP's clock. Idempotent unless ``force``."""
    out = bundle_path(cache_dir, asap_piece)
    if out.exists() and not force:
        return json.loads(out.read_text())

    row = meta.get(asap_piece)
    if row is None:
        raise AsapAudioError(f"{asap_piece}: not in ASAP metadata.csv")
    mmidi, maudio = maestro_paths(row, midi_root, audio_root)
    asap_midi = asap_root / asap_piece
    shift = derive_shift(asap_midi, mmidi, float(row.get("start") or 0.0))
    lo, hi = clip_bounds(asap_midi, shift.seconds)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        wav = slice_audio(maudio, Path(td) / "clip.wav", lo, hi)
        notes, pedals = transcribe_wav(wav)
    if not notes:
        raise AsapAudioError(f"{asap_piece}: transkun returned zero notes")

    # transcription is in sliced-audio time; put it back in ASAP's clock so the
    # ASAP beat truth applies unchanged.
    back = lo - shift.seconds
    for n in notes:
        n["onset"] = float(n["onset"]) + back
        n["offset"] = float(n["offset"]) + back

    bundle = {
        "asap_piece": asap_piece,
        "maestro_audio": str(maudio),
        "clip_maestro_sec": [round(lo, 3), round(hi, 3)],
        "shift_sec": shift.seconds,
        "shift_match_frac": shift.match_frac,
        "metadata_start": shift.metadata_start,
        "notes": notes,
        "pedal_events": pedals,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "substrate_versions": {"transcriber": TRANSCRIBER_ID, "device": "cpu",
                               "sample_rate": SAMPLE_RATE},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle))
    return bundle


def available_pieces(asap_root: Path, midi_root: Path, audio_root: Path) -> list[str]:
    """ASAP piece keys whose MAESTRO audio is actually present locally -- the set
    Track A's audio mode can run on today, without a download."""
    out = []
    for key, row in load_metadata(asap_root).items():
        try:
            maestro_paths(row, midi_root, audio_root)
        except AsapAudioError:
            continue
        out.append(key)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MAESTRO-audio->Transkun bundles for Track A (#133)")
    ap.add_argument("--asap-root", type=Path, default=DEFAULT_ASAP_ROOT)
    ap.add_argument("--maestro-midi-root", type=Path, default=DEFAULT_MAESTRO_MIDI_ROOT)
    ap.add_argument("--maestro-audio-root", type=Path, default=DEFAULT_MAESTRO_AUDIO_ROOT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--pieces", nargs="+", default=None, help="ASAP piece keys; default = all locally available")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="list locally-available pieces and exit")
    args = ap.parse_args()

    pieces = args.pieces or available_pieces(args.asap_root, args.maestro_midi_root, args.maestro_audio_root)
    if args.limit:
        pieces = pieces[:args.limit]
    if args.list:
        print(f"{len(pieces)} ASAP performances have MAESTRO audio locally:")
        for p in pieces:
            print(f"  {p}")
        return

    from follower_eval.build_corpus import _import_transcribe_wav
    transcribe_wav = _import_transcribe_wav()
    meta = load_metadata(args.asap_root)

    ok = 0
    for i, p in enumerate(pieces, 1):
        try:
            b = build_one(p, args.asap_root, args.maestro_midi_root, args.maestro_audio_root,
                          args.cache, transcribe_wav, meta, force=args.force)
            ok += 1
            print(f"[{i}/{len(pieces)}] {p}: {len(b['notes'])} notes, shift {b['shift_sec']:.3f}s "
                  f"(match {b['shift_match_frac']:.1%}, metadata said {b['metadata_start']:.3f})", flush=True)
        except AsapAudioError as exc:
            print(f"[{i}/{len(pieces)}] {p}: FAILED {exc}", flush=True)
    print(f"\n{ok}/{len(pieces)} bundles in {args.cache}")


if __name__ == "__main__":
    sys.exit(main())
