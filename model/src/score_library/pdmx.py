"""Ingest a solo-piano-classical slice of PDMX (recognize-only).

PDMX (pnlong/PDMX): ~254k MuseScore-derived public-domain scores, MusicXML-native
with derived MIDI. Most of it is pop/junk -- previously rejected for bulk import --
but a clean solo-piano-classical slice IS extractable via the shipped PDMX.csv
metadata. This module joins the on-disk MIDIs (named by IPFS CID hash) to PDMX.csv
and keeps only rows that are:
  * genres contains "classical"
  * n_tracks == 1 (solo)
  * the single track is a piano program (GM program 0; PDMX `tracks` encodes it)

Recognize-only (PDMX MIDI here is for the fingerprint catalog; we do not ship its
engravings). license_conflict is recorded but NOT filtered on -- legality is a
deferred pass per the owner's volume-first directive; the count of conflicted
rows is reported so the later pass knows the exposure.

piece_id = pdmx.<composer>.<title>_<hash8>  (hash suffix guarantees uniqueness;
musical duplicates are still collapsed by bulk_ingest's chroma+DTW dedup).

Run:
  cd model && uv run python -m score_library.pdmx --limit 300
  cd model && uv run python -m score_library.pdmx
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from score_library.bulk_ingest import Candidate, bulk_ingest

_PDMX = Path.home() / "crescendai_corpus_staging" / "pdmx"
_CSV = _PDMX / "PDMX.csv"
_MIDI_ROOT = _PDMX / "extracted" / "mid"


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "untitled"


def _is_solo_piano_classical(row: dict) -> bool:
    if "classical" not in (row.get("genres") or "").lower():
        return False
    try:
        if int(row.get("n_tracks") or 0) != 1:
            return False
    except ValueError:
        return False
    # `tracks` encodes the single track's GM program; piano family = programs 0-7.
    tracks = (row.get("tracks") or "").strip()
    prog = tracks.split("-")[-1] if "-" in tracks else tracks
    try:
        return 0 <= int(prog) <= 7
    except ValueError:
        return False


def _candidates(limit: int | None):
    # Index on-disk MIDIs by hash for an O(1) join against the 254k-row CSV.
    on_disk = {p.stem: p for p in _MIDI_ROOT.rglob("*.mid")}
    n = conflicts = 0
    with open(_CSV, newline="") as f:
        for row in csv.DictReader(f):
            mid = row.get("mid") or ""
            h = Path(mid).stem
            path = on_disk.get(h)
            if path is None or not _is_solo_piano_classical(row):
                continue
            if (row.get("license_conflict") or "").lower() == "true":
                conflicts += 1
            composer = (row.get("composer_name") or row.get("artist_name") or "Unknown").strip()
            title = (row.get("title") or row.get("song_name") or h).strip()
            pid = f"pdmx.{_sanitize(composer)[:24]}.{_sanitize(title)[:32]}_{h[:8]}"
            yield Candidate(path, pid, composer, title)
            n += 1
            if limit and n >= limit:
                break
    print(f"  (license_conflict on {conflicts} selected rows -- deferred legality pass)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not _CSV.exists():
        raise SystemExit(f"ABORT: {_CSV} missing")
    res = bulk_ingest(_candidates(args.limit))
    print(f"  ingested={len(res.ingested)} dups={len(res.dups)} "
          f"gate_fail={len(res.gate_failures)} parse_fail={len(res.parse_failures)}")


if __name__ == "__main__":
    main()
