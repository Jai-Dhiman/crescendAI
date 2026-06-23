# ASAP is CC-BY-NC -- these rendered .mxl are for the LOCAL zero-user prototype
# ONLY and must NOT be shipped/deployed to production R2 or a commercial build.
"""Build local renderable .mxl assets from ASAP xml_score.musicxml files.

For each catalog piece that has a corresponding ASAP xml_score.musicxml:
  1. Validate the source XML loads in partitura (fail-loud on bad XML).
  2. Wrap it in a standard MXL ZIP via wrap_as_mxl_zip (DOCTYPE-stripped).
  3. Run the Verovio render gate: loadZipData must return True and
     getPageCount() must be >= 1. Pieces that fail the gate are reported
     and excluded -- no broken assets are written.
  4. Write the result to <output_dir>/<piece_id>.mxl.

Usage (CLI):
    uv run python -m score_library.render_assets \
        --asap-dir data/raw/asap \
        --output-dir ../../scores/v1

All paths default to module-anchored locations so `just` CWD shifts
do not break them.

Explicit exceptions over silent fallbacks (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import partitura
import verovio

from score_library.upload import wrap_as_mxl_zip, _strip_doctype

_logger = logging.getLogger(__name__)

# Anchor to this module file, never CWD (CLAUDE.md: just recipes shift CWD).
_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]  # model/
_DEFAULT_ASAP_DIR = _MODEL_ROOT / "data" / "raw" / "asap"
_DEFAULT_OUTPUT_DIR = _MODEL_ROOT / "scores" / "v1"


def _derive_piece_id(xml_path: Path, asap_dir: Path) -> str:
    """Derive the catalog piece ID from an xml_score.musicxml path.

    Mirrors score_library.discover.derive_piece_id: relative path from
    asap_dir, each path component lowercased, joined with dots.
    The xml_score.musicxml file lives at <piece_dir>/xml_score.musicxml
    so we use its parent directory.
    """
    piece_dir = xml_path.parent
    rel = piece_dir.relative_to(asap_dir)
    return ".".join(p.lower() for p in rel.parts)


def _ensure_xml_declaration(xml_bytes: bytes) -> bytes:
    """Prepend an XML declaration if the bytes do not already have one.

    Some ASAP MusicXML files start directly with <!DOCTYPE ...> or with
    <score-partwise ...> (no prolog). After _strip_doctype removes the DOCTYPE,
    Verovio's format-detection heuristic may fail to recognise the file as
    MusicXML when there is no leading <?xml ... ?> declaration. Adding the
    declaration is safe and idempotent.
    """
    stripped = xml_bytes.lstrip()
    if stripped.startswith(b"<?xml"):
        return xml_bytes
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + stripped


def _validate_partitura(xml_path: Path) -> None:
    """Load xml_path in partitura; raise ValueError if it fails."""
    try:
        partitura.load_score(str(xml_path))
    except Exception as exc:
        raise ValueError(f"partitura load failed for {xml_path}: {exc}") from exc


def _verovio_renders(mxl_bytes: bytes) -> tuple[bool, int]:
    """Return (renders_ok, page_count).

    renders_ok is True iff loadFile returns True and page_count >= 1.
    A fresh toolkit instance is used per file to avoid state leakage.
    Uses a temporary file because the verovio Python binding's in-memory
    ZIP methods (loadZipDataBase64/loadZipDataBuffer) are not available
    or unstable in verovio 6.x -- loadFile with a .mxl path is the
    reliable path.
    """
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mxl", delete=False) as tmp:
        tmp.write(mxl_bytes)
        tmp_path = tmp.name
    try:
        tk = verovio.toolkit()
        ok = tk.loadFile(tmp_path)
        if not ok:
            return False, 0
        pages = tk.getPageCount()
        return pages >= 1, pages
    finally:
        os.unlink(tmp_path)


def _existing_inner_xml(mxl_path: Path) -> bytes | None:
    """Return the inner MusicXML bytes of an existing .mxl, or None."""
    if not mxl_path.exists():
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(mxl_path.read_bytes())) as zf:
            for name in zf.namelist():
                if not name.startswith("META-INF") and name.endswith(".xml"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        _logger.warning("corrupted existing .mxl, will rebuild: %s", mxl_path)
        return None
    return None


@dataclass
class BuildResult:
    """Summary of a build run."""
    produced: list[Path]
    skipped_unchanged: list[str]
    no_musicxml: list[str]
    partitura_failures: list[tuple[str, str]]   # (piece_id, error)
    verovio_failures: list[tuple[str, str]]     # (piece_id, reason)


def build(
    asap_dir: Path = _DEFAULT_ASAP_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    catalog_ids: Sequence[str] | None = None,
) -> BuildResult:
    """Generate one .mxl per catalog piece that has a matching ASAP musicxml.

    Args:
        asap_dir: Root of the cloned ASAP dataset.
        output_dir: Directory to write <piece_id>.mxl files.
        catalog_ids: If provided, restrict processing to these piece IDs.
                     Defaults to all piece IDs derived from ASAP xml files.

    Returns:
        BuildResult summarising produced, skipped, and failed assets.

    Raises:
        FileNotFoundError: if asap_dir does not exist.
    """
    asap_dir = Path(asap_dir)
    output_dir = Path(output_dir)

    if not asap_dir.exists():
        raise FileNotFoundError(
            f"ASAP directory not found: {asap_dir}\n"
            "Clone with: git clone --depth 1 "
            "https://github.com/CPJKU/asap-dataset.git <asap_dir>"
        )

    # Build piece_id -> xml_path map from the ASAP directory
    asap_map: dict[str, Path] = {}
    for xml_path in sorted(asap_dir.rglob("xml_score.musicxml")):
        pid = _derive_piece_id(xml_path, asap_dir)
        asap_map[pid] = xml_path

    if not asap_map:
        raise FileNotFoundError(
            f"No xml_score.musicxml files found in {asap_dir}. "
            "The ASAP clone may be incomplete."
        )

    # Determine which piece IDs to process
    target_ids: Sequence[str] = catalog_ids if catalog_ids is not None else sorted(asap_map)

    output_dir.mkdir(parents=True, exist_ok=True)

    result = BuildResult(
        produced=[],
        skipped_unchanged=[],
        no_musicxml=[],
        partitura_failures=[],
        verovio_failures=[],
    )

    for piece_id in target_ids:
        xml_path = asap_map.get(piece_id)
        if xml_path is None:
            result.no_musicxml.append(piece_id)
            continue

        # Step 1: partitura validation (fail-loud for broken source XML)
        try:
            _validate_partitura(xml_path)
        except ValueError as exc:
            result.partitura_failures.append((piece_id, str(exc)))
            _logger.error("partitura FAIL %s: %s", piece_id, exc)
            continue

        raw_xml = _ensure_xml_declaration(_strip_doctype(xml_path.read_bytes()))
        mxl_path = output_dir / f"{piece_id}.mxl"

        # Idempotent: skip if inner XML already matches what we would write.
        # wrap_as_mxl_zip will strip DOCTYPE again (idempotent); compare against
        # the preprocessed bytes we would actually store.
        if _existing_inner_xml(mxl_path) == raw_xml:
            result.skipped_unchanged.append(piece_id)
            result.produced.append(mxl_path)
            continue

        mxl_bytes = wrap_as_mxl_zip(raw_xml, piece_id)

        # Step 2: Verovio render gate
        renders_ok, page_count = _verovio_renders(mxl_bytes)
        if not renders_ok:
            reason = f"loadZipData=False" if page_count == 0 else f"page_count={page_count}"
            result.verovio_failures.append((piece_id, reason))
            _logger.error("verovio FAIL %s: %s", piece_id, reason)
            continue

        mxl_path.write_bytes(mxl_bytes)
        result.produced.append(mxl_path)
        _logger.info("wrote %s (%d pages)", mxl_path.name, page_count)

    return result


def _print_report(result: BuildResult, output_dir: Path) -> None:
    """Print a prominent build report to stdout."""
    produced = len(result.produced)
    skipped = len(result.skipped_unchanged)
    no_xml = len(result.no_musicxml)
    pf = len(result.partitura_failures)
    vf = len(result.verovio_failures)

    print(f"\n{'='*60}")
    print(f"build-catalog-mxl RESULTS")
    print(f"{'='*60}")
    print(f"  Produced (written or unchanged):  {produced}")
    print(f"    - newly written:                {produced - skipped}")
    print(f"    - skipped (already up-to-date): {skipped}")
    print(f"  No ASAP musicxml (coverage gap):  {no_xml}")
    print(f"  partitura load failures:          {pf}")
    print(f"  Verovio render failures:          {vf}")
    print(f"  Output directory:                 {output_dir}")

    if result.no_musicxml:
        print(f"\n  [NO MUSICXML] {no_xml} piece(s):")
        for pid in result.no_musicxml:
            print(f"    {pid}")

    if result.partitura_failures:
        print(f"\n  [PARTITURA FAIL] {pf} piece(s):")
        for pid, err in result.partitura_failures:
            print(f"    {pid}: {err}")

    if result.verovio_failures:
        print(f"\n  [VEROVIO FAIL] {vf} piece(s):")
        for pid, reason in result.verovio_failures:
            print(f"    {pid}: {reason}")

    print(f"{'='*60}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local renderable .mxl assets from ASAP xml_score.musicxml files."
    )
    parser.add_argument(
        "--asap-dir",
        default=str(_DEFAULT_ASAP_DIR),
        help=f"Root of the cloned ASAP dataset (default: {_DEFAULT_ASAP_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Output directory for .mxl files (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--catalog",
        help=(
            "Path to titles.json (catalog piece IDs). "
            "If provided, only catalog-listed pieces are processed and coverage gaps are reported. "
            f"Default: {_MODEL_ROOT / 'data' / 'scores' / 'titles.json'}"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asap_dir = Path(args.asap_dir)
    output_dir = Path(args.output_dir)

    # Load catalog IDs if a catalog file is provided or found at the default location
    catalog_ids: list[str] | None = None
    catalog_path = Path(args.catalog) if args.catalog else (_MODEL_ROOT / "data" / "scores" / "titles.json")
    if catalog_path.exists():
        import json
        with open(catalog_path) as f:
            catalog_ids = sorted(json.load(f).keys())
        print(f"Loaded {len(catalog_ids)} catalog piece IDs from {catalog_path}")
    else:
        print(f"No catalog file found at {catalog_path}; processing all ASAP pieces")

    # Count before
    before = len(list(output_dir.glob("*.mxl"))) if output_dir.exists() else 0
    print(f"Before: {before} .mxl files in {output_dir}")

    result = build(asap_dir=asap_dir, output_dir=output_dir, catalog_ids=catalog_ids)

    # Count after
    after = len(list(output_dir.glob("*.mxl")))
    print(f"After:  {after} .mxl files in {output_dir}")

    _print_report(result, output_dir)

    # Exit non-zero only if zero assets were produced
    total_failures = len(result.partitura_failures) + len(result.verovio_failures)
    if total_failures:
        print(f"WARNING: {total_failures} failure(s) — see report above.")

    produced_count = len(result.produced)
    if produced_count == 0:
        print("ERROR: zero assets produced — exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
