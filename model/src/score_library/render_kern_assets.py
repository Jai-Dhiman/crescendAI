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

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_MEI_OUT = _MODEL_ROOT / "scores" / "v1"
_KERN_CLONE = Path(os.environ.get("KERNSCORES_CLONE_DIR", str(Path.home() / "crescendai_corpus_staging" / "kernscores")))

_COLLECTIONS = [
    ("joplin", _joplin_piece_id),
    ("scarlatti-keyboard-sonatas", _scarlatti_piece_id),
    ("chopin-mazurkas", _chopin_mazurka_piece_id),
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
    for repo, pid_fn in _COLLECTIONS:
        kern_dir = _KERN_CLONE / repo / "kern"
        krns = sorted(kern_dir.glob("*.krn"))
        if not krns:
            raise SystemExit(f"ABORT: no .krn in {kern_dir}")
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
            ok, err = _kern_to_mei(krn, dest)
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
