"""Ingest the GiantMIDI-Piano corpus (recognize-only) into the piece-ID catalog.

GiantMIDI-Piano (ByteDance): ~10,855 solo-piano works (or the 7,236 curated
"surname_checked" subset) transcribed from YouTube audio via onset-and-frames.
There are NO engravings, so these are recognize-only (no MEI render) -- they
grow the fingerprint catalog only.

CAVEAT: these are AMT-transcribed PERFORMANCE MIDIs (expressive timing, no
quantized grid), unlike the score-engraved MIDI the self-consistency gate was
tuned for. A meaningful fraction is expected to fail the gate's 16th-note
quantization / bar-wellformedness checks. Run with `--limit N` first to measure
the rejection rate before committing to the full corpus.

Filenames: "{Surname}, {First}, {Title...}, {YouTubeID}.mid" -- the title may
itself contain commas/opus numbers; the LAST comma-field is the 11-char YouTube
ID, which we fold into the piece_id to guarantee uniqueness (musical duplicates
-- the same work uploaded multiple times -- are still collapsed by bulk_ingest's
chroma+DTW dedup before the id is ever used).

Run:
  cd model && uv run python -m score_library.giantmidi --subset surname --limit 300
  cd model && uv run python -m score_library.giantmidi --subset surname
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from score_library.bulk_ingest import Candidate, bulk_ingest

_STAGE = Path.home() / "crescendai_corpus_staging" / "giantmidi" / "GiantMIDI-PIano"
_SUBSETS = {
    "surname": _STAGE / "surname_midis",          # 7,236 curated
    "full": _STAGE / "full_midis",                # 10,855
}
_YTID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _parse(name: str) -> tuple[str, str, str, str] | None:
    """'{Surname}, {First}, {Title...}, {YTID}.mid' -> (surname, first, title, ytid)."""
    stem = name[:-4] if name.endswith(".mid") else name
    parts = [p.strip() for p in stem.split(",")]
    if len(parts) < 3 or not _YTID.match(parts[-1]):
        return None
    surname, first = parts[0], parts[1]
    title = ", ".join(parts[2:-1]) if len(parts) > 3 else parts[2]
    return surname, first, title, parts[-1]


def _candidates(midi_dir: Path, limit: int | None, offset: int = 0):
    n = 0
    for i, midi in enumerate(sorted(midi_dir.rglob("*.mid"))):
        if i < offset:
            continue
        parsed = _parse(midi.name)
        if parsed is None:
            continue
        surname, first, title, ytid = parsed
        pid = f"giantmidi.{_sanitize(surname)}.{_sanitize(title)[:40]}_{ytid}"
        composer = f"{first} {surname}".strip()
        yield Candidate(midi, pid, composer, title)
        n += 1
        if limit and n >= limit:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(_SUBSETS), default="surname")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--note-cap", type=int, default=600,
                    help="dedup window (notes); GiantMIDI is multi-thousand-note performance MIDI")
    args = ap.parse_args()

    midi_dir = _SUBSETS[args.subset]
    if not midi_dir.exists():
        raise SystemExit(f"ABORT: {midi_dir} missing -- unzip the GiantMIDI archive there first")

    res = bulk_ingest(_candidates(midi_dir, args.limit, args.offset), note_cap=args.note_cap)
    total = len(res.ingested) + len(res.dups) + len(res.gate_failures) + len(res.parse_failures)
    print(f"  subset={args.subset} considered={total} ingested={len(res.ingested)} "
          f"dups={len(res.dups)} gate_fail={len(res.gate_failures)} parse_fail={len(res.parse_failures)}")
    if total:
        print(f"  gate-reject rate: {100*(len(res.gate_failures))/total:.1f}%  "
              f"(AMT performance-MIDI vs score-tuned gate)")


if __name__ == "__main__":
    main()
