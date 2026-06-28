"""Render recognize-only MIDI-source catalog pieces to interactive .mxl notation.

GiantMIDI (performance MIDI) and PDMX (score MIDI) ship as MIDI only -- no clean
kern/MEI engraving -- so they are display-less in the Verovio tiers. MuseScore 4
headless imports a MIDI and emits compressed MusicXML (.mxl), which getPieceData
serves as the INTERACTIVE tier (Verovio loadZipDataBuffer -> per-bar highlight +
clip playback), strictly better than a static SVG.

Engine choice + the `-M` import-quantization config were locked by a bake-off
through the real scorehost (see issue #97): MuseScore beats partitura (single
collapsed staff) and midi2ly (solid-ink beam blocks). The OPS config kills the
clutter that raw MuseScore import leaves on performance MIDI: no mid-staff clef
changes, no exotic tuplets, simplified durations, forced grand-staff split,
HumanPerformance on (rounder rhythms + per-performance rubato adaptation).

Pairs with `rebar_from_mei` (generalized to .mxl) -- every rendered interactive
piece MUST be re-barred to the Verovio measure grid or live highlighting
mis-points. The note set is unchanged by render+rebar, so the piece-ID
fingerprint is untouched.

GiantMIDI is recognize-only-NO-license-sorting this pass (owner: volume now,
license later); PDMX is CC0/public-domain.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES = _MODEL_ROOT / "data" / "scores"
_DISPLAY = _MODEL_ROOT / "scores" / "v1"
_MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"

_GIANTMIDI_DIR = Path.home() / "crescendai_corpus_staging" / "giantmidi" / "GiantMIDI-PIano" / "full_midis" / "midis"
_PDMX_ROOT = Path.home() / "crescendai_corpus_staging" / "pdmx"
_PDMX_MIDI = _PDMX_ROOT / "extracted" / "mid"
_PDMX_CSV = _PDMX_ROOT / "PDMX.csv"

# MuseScore MIDI-import operations (the -M file). Schema is MuseScore's
# importmidi_operations.cpp: QuantValue index 0..8 = quarter..1024th.
_OPS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MidiOptions>
  <QuantValue>2</QuantValue>
  <VoiceCount>1</VoiceCount>
  <Duplets>false</Duplets>
  <Triplets>true</Triplets>
  <Quadruplets>false</Quadruplets>
  <Quintuplets>false</Quintuplets>
  <Septuplets>false</Septuplets>
  <Nonuplets>false</Nonuplets>
  <HumanPerformance>true</HumanPerformance>
  <SplitStaff>true</SplitStaff>
  <ClefChanges>false</ClefChanges>
  <SimplifyDurations>true</SimplifyDurations>
  <ShowStaccato>false</ShowStaccato>
  <DottedNotes>true</DottedNotes>
  <RecognizePickupBar>true</RecognizePickupBar>
</MidiOptions>
"""

_YTID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "untitled"


def _has_display(pid: str, display_dir: Path) -> bool:
    return any((display_dir / f"{pid}{ext}").exists() for ext in (".mei", ".mxl", ".svg"))


def iter_giantmidi(scores_dir: Path, display_dir: Path, midi_dir: Path):
    """Yield (pid, midi) for GiantMIDI pieces that are in the catalog (have a score
    JSON) but have no display asset yet. Mirrors giantmidi._parse pid construction."""
    for midi in sorted(midi_dir.glob("*.mid")):
        parts = [p.strip() for p in midi.stem.split(",")]
        if len(parts) < 3 or not _YTID.match(parts[-1]):
            continue
        surname, title, ytid = parts[0], ", ".join(parts[2:-1]), parts[-1]
        pid = f"giantmidi.{_sanitize(surname)}.{_sanitize(title)[:40]}_{ytid}"
        if (scores_dir / f"{pid}.json").exists() and not _has_display(pid, display_dir):
            yield pid, midi


def iter_pdmx(scores_dir: Path, display_dir: Path):
    """Yield (pid, midi) for in-catalog PDMX pieces lacking a display asset.
    Mirrors pdmx._candidates pid construction; joins on-disk MIDIs to PDMX.csv."""
    on_disk = {p.stem: p for p in _PDMX_MIDI.rglob("*.mid")}
    with open(_PDMX_CSV, newline="") as f:
        for row in csv.DictReader(f):
            h = Path(row.get("mid") or "").stem
            midi = on_disk.get(h)
            if midi is None:
                continue
            composer = (row.get("composer_name") or row.get("artist_name") or "Unknown").strip()
            title = (row.get("title") or row.get("song_name") or h).strip()
            pid = f"pdmx.{_sanitize(composer)[:24]}.{_sanitize(title)[:32]}_{h[:8]}"
            if (scores_dir / f"{pid}.json").exists() and not _has_display(pid, display_dir):
                yield pid, midi


def render_mxl(pid: str, midi: Path, out_dir: Path, ops: Path, timeout: int = 120) -> tuple[bool, str]:
    """MuseScore MIDI -> <out_dir>/<pid>.mxl. A leftover <pid>.mxl.inprogress means a
    prior run crashed/timed out on this piece -> promote to a .skip sentinel so the
    resume loop does not wedge on it."""
    dst = out_dir / f"{pid}.mxl"
    sentinel = out_dir / f"{pid}.mxl.inprogress"
    skip = out_dir / f"{pid}.mxl.skip"
    if skip.exists():
        return False, "skip-sentinel (prior crash)"
    if sentinel.exists():
        sentinel.unlink(missing_ok=True)
        skip.write_text("crashed on a prior run\n")
        return False, "promoted to skip-sentinel"
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("rendering\n")
    try:
        proc = subprocess.run(
            [_MSCORE, "-M", str(ops), "-o", str(dst), str(midi)],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        sentinel.unlink(missing_ok=True)
        skip.write_text("timeout\n")
        return False, "TIMEOUT"
    sentinel.unlink(missing_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True, "ok"
    return False, f"no output (exit {proc.returncode}): {proc.stderr.decode()[-120:]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render recognize-only MIDI pieces to interactive .mxl via MuseScore.")
    ap.add_argument("--source", choices=["giantmidi", "pdmx"], required=True)
    ap.add_argument("--scores-dir", type=Path, default=_SCORES)
    ap.add_argument("--out-dir", type=Path, default=_DISPLAY)
    ap.add_argument("--giantmidi-dir", type=Path, default=_GIANTMIDI_DIR)
    ap.add_argument("--limit", type=int, default=0, help="render at most N (0=all)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel MuseScore workers")
    args = ap.parse_args()

    if args.source == "giantmidi":
        cands = list(iter_giantmidi(args.scores_dir, args.out_dir, args.giantmidi_dir))
    else:
        cands = list(iter_pdmx(args.scores_dir, args.out_dir))
    if args.limit:
        cands = cands[: args.limit]
    print(f"{args.source}: {len(cands)} pieces to render -> .mxl, {args.jobs} jobs", flush=True)

    ok = 0
    fails: list[tuple[str, str]] = []
    done = 0
    with tempfile.TemporaryDirectory() as tmp:
        ops = Path(tmp) / "ops.xml"
        ops.write_text(_OPS_XML)

        def task(item):
            pid, midi = item
            return pid, render_mxl(pid, midi, args.out_dir, ops)

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(task, it) for it in cands]
            for fut in as_completed(futs):
                pid, (success, msg) = fut.result()
                done += 1
                if success:
                    ok += 1
                else:
                    fails.append((pid, msg))
                if done % 50 == 0:
                    print(f"  {done}/{len(cands)} ({ok} ok, {len(fails)} failed)", flush=True)
    print(f"\nDONE: {ok}/{len(cands)} rendered; {len(fails)} failed")
    for pid, msg in fails[:20]:
        print(f"  FAIL {pid}: {msg}")


if __name__ == "__main__":
    main()
