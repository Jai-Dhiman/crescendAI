# model/src/exercise_corpus/build_render_assets.py
"""Build committed renderable assets (.mxl) from committed exercise-primitive
MusicXML (.xml).

For each model/data/scores/exercise_primitives/*.xml: validate it loads in
partitura, strip its DOCTYPE, wrap it in a standard MXL ZIP (reusing the proven
score_library.upload.wrap_as_mxl_zip container format), and write
model/data/exercise_primitives/mxl/{primitive_id}.mxl.

Deterministic and idempotent: an asset whose inner XML already equals the
freshly-stripped source XML is left untouched. A .xml that fails partitura load
RAISES naming the file (no skip-and-continue) — explicit exceptions over silent
fallbacks (CLAUDE.md).
"""

from __future__ import annotations

import glob
import io
import zipfile
from pathlib import Path

import partitura

from score_library.upload import _strip_doctype, wrap_as_mxl_zip

# Anchor to this module, never CWD (CLAUDE.md: just recipes shift CWD).
_MODEL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XML_DIR = _MODEL_ROOT / "data" / "scores" / "exercise_primitives"
_DEFAULT_OUT_DIR = _MODEL_ROOT / "data" / "exercise_primitives" / "mxl"


def _existing_inner_xml(mxl_path: Path) -> bytes | None:
    """Return the inner MusicXML bytes of an existing .mxl, or None if absent
    or unreadable — used for the idempotent skip-if-unchanged check."""
    if not mxl_path.exists():
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(mxl_path.read_bytes())) as zf:
            for name in zf.namelist():
                if not name.startswith("META-INF") and name.endswith(".xml"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        return None
    return None


def build(
    xml_dir: Path = _DEFAULT_XML_DIR,
    out_dir: Path = _DEFAULT_OUT_DIR,
) -> list[Path]:
    """Generate one .mxl per committed primitive .xml. Returns produced paths
    (sorted by primitive id), including unchanged ones that were skipped.

    Raises:
        FileNotFoundError: if xml_dir contains no .xml files.
        ValueError: if any .xml fails to load in partitura (message names the file).
    """
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    xml_files = sorted(glob.glob(str(xml_dir / "*.xml")))
    if not xml_files:
        raise FileNotFoundError(f"No primitive .xml files found in {xml_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for xml_path_str in xml_files:
        xml_path = Path(xml_path_str)
        primitive_id = xml_path.stem

        # Fail loud if the source XML is not valid for the corpus.
        try:
            partitura.load_score(str(xml_path))
        except Exception as e:  # noqa: BLE001 — re-raise with the offending file named
            raise ValueError(f"primitive .xml failed partitura load: {xml_path} ({e})") from e

        stripped = _strip_doctype(xml_path.read_bytes())
        mxl_path = out_dir / f"{primitive_id}.mxl"

        # Idempotent: skip if the inner XML already matches the stripped source.
        if _existing_inner_xml(mxl_path) == stripped:
            produced.append(mxl_path)
            continue

        mxl_bytes = wrap_as_mxl_zip(stripped, primitive_id)
        mxl_path.write_bytes(mxl_bytes)
        produced.append(mxl_path)

    return produced
