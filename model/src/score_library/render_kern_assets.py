"""Build PD-clean renderable scores from KernScores **kern via Verovio-native MEI.

Closes the recognize/display gap with COMMERCIAL-CLEAN assets: the catalog's
KernScores pieces are recognizable (fingerprint index) but were only renderable
via CC-BY-NC ASAP .mxl. Humdrum **kern (craigsapp, public domain) is loaded
DIRECTLY by Verovio's mature Humdrum importer (the one KernScores' own site
uses) and exported as native MEI -- ~100% yield and faithful notation, vs ~46%
through a partitura->MusicXML intermediate that Verovio's MusicXML importer
chokes on. MEI is Verovio's native format, so the scorehost renders it via the
worker's existing loadData() path.

Per piece: verovio.loadFile(kern) must return True AND getPageCount() >= 1
(the render gate), then getMEI() is written to scores/v1/<piece_id>.mei.
Writes are immediate + skip-existing, so a rerun resumes (defensive against a
rare Verovio segfault). Only .krn whose derived piece_id is in the catalog are
rendered, so .mei piece_ids match the fingerprint index.

Run:  cd model && uv run python -m score_library.render_kern_assets
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import verovio

from score_library.kernscores_bulk import (
    _chopin_mazurka_piece_id,
    _joplin_piece_id,
    _scarlatti_piece_id,
)
from score_library.kernscores_expand import (
    _artfugue_piece_id,
    _beethoven_sonata_piece_id,
    _chopin_prelude_piece_id,
    _haydn_sonata_piece_id,
    _hummel_prelude_piece_id,
    _mozart_sonata_piece_id,
    _scriabin_piece_id,
)

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_MEI_OUT = _MODEL_ROOT / "scores" / "v1"
_KERN_CLONE = Path(os.environ.get("KERNSCORES_CLONE_DIR", str(Path.home() / "crescendai_corpus_staging" / "kernscores")))

# (repo, piece_id_fn, kern_glob) -- kern_glob is relative to the repo root and
# defaults to "kern/*.krn"; scriabin nests its **kern by opus instead.
_COLLECTIONS = [
    ("joplin", _joplin_piece_id, "kern/*.krn"),
    ("scarlatti-keyboard-sonatas", _scarlatti_piece_id, "kern/*.krn"),
    ("chopin-mazurkas", _chopin_mazurka_piece_id, "kern/*.krn"),
    ("beethoven-piano-sonatas", _beethoven_sonata_piece_id, "kern/*.krn"),
    ("mozart-piano-sonatas", _mozart_sonata_piece_id, "kern/*.krn"),
    ("haydn-keyboard-sonatas", _haydn_sonata_piece_id, "kern/*.krn"),
    ("chopin-preludes", _chopin_prelude_piece_id, "kern/*.krn"),
    ("hummel-preludes", _hummel_prelude_piece_id, "kern/*.krn"),
    ("art-of-the-fugue", _artfugue_piece_id, "kern/*.krn"),
    ("scriabin", _scriabin_piece_id, "op*/*.krn"),
]


def _kern_to_mei(krn: Path, dest: Path) -> tuple[bool, str]:
    """Verovio Humdrum -> MEI render gate + write. Returns (ok, error)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk = verovio.toolkit()
        tk.setInputFrom("humdrum")
        if not tk.loadFile(str(krn)):
            return False, "verovio loadFile returned False"
        pages = tk.getPageCount()
        if pages < 1:
            return False, f"getPageCount={pages}"
        mei = tk.getMEI()
    if not mei or len(mei) < 500:
        return False, f"empty/short MEI ({len(mei)}B)"
    dest.write_text(mei)
    return True, ""


def main() -> None:
    if not _KERN_CLONE.exists():
        raise SystemExit(f"ABORT: kernscores clone missing at {_KERN_CLONE} -- run acquire_kernscores.sh")
    _MEI_OUT.mkdir(parents=True, exist_ok=True)

    rendered = resumed = no_catalog = 0
    failures: list[tuple[str, str]] = []
    for repo, pid_fn, kern_glob in _COLLECTIONS:
        krns = sorted((_KERN_CLONE / repo).glob(kern_glob))
        if not krns:
            raise SystemExit(f"ABORT: no .krn matched {repo}/{kern_glob}")
        repo_ok = repo_pieces = 0
        for krn in krns:
            try:
                pid = pid_fn(krn.stem)
            except ValueError as e:
                failures.append((krn.name, f"piece_id: {e}")); continue
            if not (_SCORES_DIR / f"{pid}.json").exists():
                no_catalog += 1; continue
            repo_pieces += 1
            dest = _MEI_OUT / f"{pid}.mei"
            if dest.exists():
                resumed += 1; repo_ok += 1; continue
            # Segfault-resume: Verovio's getMEI() SIGSEGVs on a few pathological
            # **kern. A try/except cannot catch a SIGSEGV, so an `.inprogress`
            # sentinel left over from the prior run identifies the crasher and
            # promotes it to a permanent `.segfault` skip-marker (surfaced as a
            # failure). Drive to completion with an until-loop (the recipe does).
            segfault_marker = _MEI_OUT / f"{pid}.mei.segfault"
            inprogress = _MEI_OUT / f"{pid}.mei.inprogress"
            if segfault_marker.exists():
                failures.append((pid, "SIGSEGV (skipped)")); continue
            if inprogress.exists():
                inprogress.rename(segfault_marker)
                failures.append((pid, "SIGSEGV (skipped)")); continue
            inprogress.write_bytes(b"")
            ok, err = _kern_to_mei(krn, dest)
            inprogress.unlink(missing_ok=True)
            if ok:
                repo_ok += 1; rendered += 1
            else:
                failures.append((pid, err))
        print(f"  {repo:30s} {repo_ok}/{repo_pieces} catalog pieces have MEI", flush=True)

    print(f"\n=== KERN -> Verovio-native MEI ===")
    print(f"  rendered this run: {rendered}   (resumed/pre-existing: {resumed})")
    print(f"  skipped (not a catalog piece): {no_catalog}")
    print(f"  failed render gate (excluded): {len(failures)}")
    if failures:
        print("\n-- failures --")
        for pid, err in failures[:30]:
            print(f"  {pid}: {err}")
    print(f"\nscores/v1 now: {len(list(_MEI_OUT.glob('*.mei')))} .mei + {len(list(_MEI_OUT.glob('*.mxl')))} .mxl")


if __name__ == "__main__":
    main()
