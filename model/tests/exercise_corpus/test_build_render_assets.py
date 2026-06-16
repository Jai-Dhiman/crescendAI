# model/tests/exercise_corpus/test_build_render_assets.py
"""build() must materialize every committed primitive .xml as a valid MXL ZIP
whose inner MusicXML preserves the source note count."""

import glob
import io
import zipfile
from pathlib import Path

import partitura
from partitura.score import Note

from exercise_corpus.build_render_assets import build

_XML_DIR = Path(__file__).resolve().parents[2] / "data" / "scores" / "exercise_primitives"
_ZIP_MAGIC = b"PK\x03\x04"


def _note_count_xml(xml_path: Path) -> int:
    score = partitura.load_score(str(xml_path))
    return sum(1 for _ in list(score.parts)[0].iter_all(Note))


def _inner_xml_bytes(mxl_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(mxl_bytes)) as zf:
        for name in zf.namelist():
            if not name.startswith("META-INF") and name.endswith(".xml"):
                return zf.read(name)
    raise AssertionError("no MusicXML entry inside MXL ZIP")


def test_build_emits_valid_mxl_with_matching_note_count(tmp_path: Path):
    out_dir = tmp_path / "mxl"
    produced = build(xml_dir=_XML_DIR, out_dir=out_dir)

    xml_files = sorted(glob.glob(str(_XML_DIR / "*.xml")))
    assert len(produced) == len(xml_files) == 22

    for xml_path_str in xml_files:
        xml_path = Path(xml_path_str)
        mxl_path = out_dir / f"{xml_path.stem}.mxl"
        assert mxl_path.exists(), f"missing asset for {xml_path.stem}"

        mxl_bytes = mxl_path.read_bytes()
        assert mxl_bytes[:4] == _ZIP_MAGIC, f"{mxl_path.name} is not a ZIP"

        # Inner MusicXML must parse and preserve note count.
        inner = _inner_xml_bytes(mxl_bytes)
        inner_path = tmp_path / f"{xml_path.stem}_inner.musicxml"
        inner_path.write_bytes(inner)
        assert _note_count_xml(inner_path) == _note_count_xml(xml_path)

        # DOCTYPE must be stripped (Verovio WASM parser invariant).
        assert b"<!DOCTYPE" not in inner
