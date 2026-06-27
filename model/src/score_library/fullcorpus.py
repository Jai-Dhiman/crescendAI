"""Onboard the solo-keyboard subset of the full KernScores/Humdrum corpus.

Source: the humdrum-tools/humdrum-data meta-repo (`git clone ... && make`), which
sparse-checks-out ~48 humdrum repos into a semantic tree (composer/genre/...)
under ~/crescendai_corpus_staging/humdrum-data, alongside raw clones under
`.source/` (which we IGNORE to avoid double-ingest).

This module is the durable record of the keyboard slice:
  1. Walk a whitelist of keyboard composer/genre directories, skipping `.source/`.
  2. Per-file keyboard guard: a .krn is solo keyboard iff it has a *Ipiano-family
     instrument token OR a grand-staff clef pair (treble *clefG2 + bass *clefF4)
     AND exactly two **kern spines. This drops chamber stragglers (cello/violin
     parts, songs) that share a composer dir.
  3. Segfault-resumable kern->MIDI staging (Verovio SIGSEGVs on a few **kern; an
     `.inprogress` sentinel promotes the crasher to a permanent `.segfault` skip).
  4. Emit a manifest {piece_id: {krn, midi}} consumed by both bulk_ingest (dedup
     + self-consistency catalog ingest) and the manifest-driven MEI render.

piece_id is derived from the file's path under the tree (composer.genre.stem,
sanitized), so it is stable and collision-free across collections without a
per-collection id function. Residual multi-edition duplicates (the NIFC
first-editions ship many publisher editions per work) are collapsed downstream
by bulk_ingest's incremental intra-batch dedup, not here.

Run:
  cd model && uv run python -m score_library.fullcorpus stage     # stage + manifest
  cd model && uv run python -m score_library.fullcorpus ingest    # bulk_ingest manifest
  cd model && uv run python -m score_library.fullcorpus render    # MEI for catalog pieces
"""
from __future__ import annotations

import base64
import json
import re
import sys
import warnings
from pathlib import Path

import verovio

from score_library.bulk_ingest import Candidate, bulk_ingest

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_MEI_OUT = _MODEL_ROOT / "scores" / "v1"
_TREE = Path.home() / "crescendai_corpus_staging" / "humdrum-data"
_MIDI_BASE = Path.home() / "crescendai_corpus_staging" / "fullcorpus_midi"
_MANIFEST = _MODEL_ROOT / "data" / "manifests" / "fullcorpus_keyboard_manifest.json"

# Keyboard composer/genre subtrees (relative to _TREE). The full corpus is
# ~48 repos across all genres; this is the solo-keyboard whitelist. Composer is
# the first path component; everything non-keyboard (essen folksongs, jrp
# vocal, string quartets, symphonies, chorales) is excluded by omission.
_KEYBOARD_DIRS = [
    "beethoven/piano",
    "mozart/piano",
    "haydn/piano",
    "scarlatti/sonata",
    "chopin",          # all chopin genres are solo piano (etude/waltz/...; first-editions)
    "joplin",
    "hummel",
    "scriabin",
    "bach/wtc",
    "bach/inventions",
    "bach/art-of-the-fugue",
    "clementi",
    "kuhlau",
    "field",
    "liszt",
    "schumann",
    "schubert/piano",
    "grieg",
]

_COMPOSER = {
    "beethoven": "Beethoven", "mozart": "Mozart", "haydn": "Haydn",
    "scarlatti": "Scarlatti", "chopin": "Chopin", "joplin": "Scott Joplin",
    "hummel": "Hummel", "scriabin": "Scriabin", "bach": "J.S. Bach",
    "clementi": "Clementi", "kuhlau": "Kuhlau", "field": "John Field",
    "liszt": "Liszt", "schumann": "Schumann", "schubert": "Schubert",
    "grieg": "Grieg",
}

# Some chamber-work part files (e.g. chopin/first-editions/008-1-Sm-003-violoncello.krn)
# carry NO real *I instrument token -- the only *I line is a mis-encoded tempo
# marking (*I"ADAGIO.) -- and a Romantic string part switches between bass, tenor
# and treble clef, so it spuriously satisfies the grand-staff clef heuristic. The
# instrument is only reliably encoded in the filename suffix, so guard on that too.
_FOREIGN_FILENAME = re.compile(
    r"-(violoncello|violino|violin|viola|cello|contrabass|flute|oboe|clarinet|"
    r"bassoon|horn|trumpet|trombone|tuba|harp|voice|vocal|soprano|alto|tenor|bass)",
    re.IGNORECASE,
)

_KEYBOARD_INSTR = re.compile(r"^\*I(piano|forte|hpsi|clav|cemb|organ)", re.MULTILINE)
# Foreign (non-keyboard) instrument tandem interpretations: any presence means
# the file is chamber/vocal/orchestral, not solo keyboard, even if a piano part
# carries grand-staff clefs.
_FOREIGN_INSTR = re.compile(
    r"^\*I(viol|cello|viola|violn|violino|violoncello|cb|"
    r"flt|flute|oboe|clars?|fagot|bassn|"
    r"cor|horn|tromb|trumpet|tromp|tuba|"
    r"guitar|lute|harp|mand|"
    r"vox|sopr|alto|tenor|barit|bass|cant|"
    r"timpani|perc)",
    re.MULTILINE,
)


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _piece_id(krn: Path) -> str:
    rel = krn.relative_to(_TREE).with_suffix("")
    return ".".join(_sanitize(p) for p in rel.parts)


def _is_keyboard(krn: Path) -> bool:
    """Solo-keyboard guard: exclude any file with a foreign (string/wind/vocal)
    instrument tandem; otherwise accept a piano-family instrument token, or a
    grand-staff clef pair (treble *clefG2 + bass *clefF4). Spine count is NOT
    used -- Romantic piano is routinely encoded with 3+ voice spines."""
    if _FOREIGN_FILENAME.search(krn.name):
        return False
    try:
        text = krn.read_text(errors="ignore")
    except Exception:
        return False
    if _FOREIGN_INSTR.search(text):
        return False
    if _KEYBOARD_INSTR.search(text):
        return True
    return "*clefG2" in text and "*clefF4" in text


def _iter_keyboard_krns():
    seen: set[Path] = set()
    for sub in _KEYBOARD_DIRS:
        base = _TREE / sub
        if not base.exists():
            continue
        for krn in sorted(base.rglob("*.krn")):
            if ".source" in krn.parts or krn in seen:
                continue
            seen.add(krn)
            yield krn


def stage() -> dict:
    """Keyboard-filter + segfault-resumable kern->MIDI; write the manifest.

    Idempotent + resumable: existing MIDIs and existing manifest entries are
    kept; an `.inprogress` sentinel left from a prior SIGSEGV is promoted to a
    permanent `.segfault` skip. Drive with an until-loop to clear segfaults.
    """
    _MIDI_BASE.mkdir(parents=True, exist_ok=True)
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else {}

    tk = verovio.toolkit()
    tk.setInputFrom("humdrum")

    examined = kept = new = nonkb = segfault = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for krn in _iter_keyboard_krns():
            examined += 1
            if not _is_keyboard(krn):
                nonkb += 1
                continue
            pid = _piece_id(krn)
            mid = _MIDI_BASE / f"{pid}.mid"
            seg = _MIDI_BASE / f"{pid}.segfault"
            inp = _MIDI_BASE / f"{pid}.inprogress"
            if mid.exists():
                kept += 1
                manifest[pid] = {"krn": str(krn), "midi": str(mid)}
                continue
            if seg.exists():
                segfault += 1
                continue
            if inp.exists():
                inp.rename(seg)
                segfault += 1
                continue
            inp.write_bytes(b"")
            try:
                if not tk.loadFile(str(krn)):
                    raise RuntimeError("verovio loadFile returned False")
                mid.write_bytes(base64.b64decode(tk.renderToMIDI()))
            except Exception as e:  # noqa: BLE001
                inp.unlink(missing_ok=True)
                print(f"  STAGE-FAIL {pid}: {e}", file=sys.stderr)
                continue
            inp.unlink(missing_ok=True)
            manifest[pid] = {"krn": str(krn), "midi": str(mid)}
            kept += 1
            new += 1
            if new % 100 == 0:
                _MANIFEST.write_text(json.dumps(manifest, indent=2))
                print(f"  ...staged {new} new (manifest={len(manifest)})", flush=True)

    _MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\n=== stage: examined={examined} keyboard={kept} non-keyboard={nonkb} "
          f"new={new} segfault={segfault} manifest={len(manifest)} ===")
    return manifest


def ingest() -> None:
    manifest = json.loads(_MANIFEST.read_text())

    def candidates():
        for pid, rec in manifest.items():
            composer = _COMPOSER.get(pid.split(".")[0], pid.split(".")[0].title())
            title = pid.split(".", 1)[1].replace(".", " ").replace("_", " ").title()
            yield Candidate(Path(rec["midi"]), pid, composer, title)

    res = bulk_ingest(candidates())
    print(f"  ingested={len(res.ingested)} dups={len(res.dups)} "
          f"gate_fail={len(res.gate_failures)} parse_fail={len(res.parse_failures)}")


def render() -> None:
    """Render MEI for manifest pieces that made it into the catalog."""
    manifest = json.loads(_MANIFEST.read_text())
    _MEI_OUT.mkdir(parents=True, exist_ok=True)
    rendered = resumed = no_catalog = segfault = 0
    failures: list[tuple[str, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pid, rec in manifest.items():
            if not (_SCORES_DIR / f"{pid}.json").exists():
                no_catalog += 1
                continue
            dest = _MEI_OUT / f"{pid}.mei"
            if dest.exists():
                resumed += 1
                continue
            seg = _MEI_OUT / f"{pid}.mei.segfault"
            inp = _MEI_OUT / f"{pid}.mei.inprogress"
            if seg.exists():
                segfault += 1
                continue
            if inp.exists():
                inp.rename(seg)
                segfault += 1
                continue
            inp.write_bytes(b"")
            tk = verovio.toolkit()
            tk.setInputFrom("humdrum")
            ok = tk.loadFile(rec["krn"])
            mei = tk.getMEI() if ok and tk.getPageCount() >= 1 else ""
            inp.unlink(missing_ok=True)
            if mei and len(mei) >= 500:
                dest.write_text(mei)
                rendered += 1
            else:
                failures.append((pid, "render gate failed"))
    print(f"\n=== render: new={rendered} resumed={resumed} not-in-catalog={no_catalog} "
          f"segfault={segfault} fail={len(failures)} ===")
    print(f"scores/v1 now: {len(list(_MEI_OUT.glob('*.mei')))} .mei")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage"
    {"stage": stage, "ingest": ingest, "render": render}[cmd]()


if __name__ == "__main__":
    main()
