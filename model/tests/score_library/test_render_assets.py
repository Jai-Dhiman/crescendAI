"""Tests for score_library.render_assets -- Verovio render gate on real .mxl assets.

Requires model/scores/v1/ to be populated (run `just build-catalog-mxl` first).
Skips gracefully if no .mxl assets are found.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import verovio

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "scores" / "v1"


def _find_one_generated_mxl() -> Path | None:
    """Return one .mxl from scores/v1/ that is NOT chopin.ballades.1.mxl
    (the pre-existing asset), or fall back to any .mxl if that's all there is."""
    if not _SCORES_DIR.exists():
        return None
    all_mxl = sorted(_SCORES_DIR.glob("*.mxl"))
    if not all_mxl:
        return None
    # Prefer a newly generated piece over the pre-existing one
    generated = [p for p in all_mxl if p.stem != "chopin.ballades.1"]
    return generated[0] if generated else all_mxl[0]


_SAMPLE_MXL = _find_one_generated_mxl()


@pytest.mark.skipif(
    _SAMPLE_MXL is None,
    reason="No .mxl assets found in model/scores/v1/ -- run `just build-catalog-mxl` first",
)
def test_generated_mxl_is_valid_zip() -> None:
    """The generated .mxl is a proper ZIP archive."""
    assert _SAMPLE_MXL is not None
    data = _SAMPLE_MXL.read_bytes()
    assert data[:4] == b"PK\x03\x04", f"{_SAMPLE_MXL.name} is not a ZIP (missing PK magic)"
    with zipfile.ZipFile(_SAMPLE_MXL, "r") as zf:
        assert zf.testzip() is None, f"{_SAMPLE_MXL.name} has a bad entry"
    names = [n for n in zipfile.ZipFile(_SAMPLE_MXL).namelist()]
    assert any(n.endswith(".xml") and not n.startswith("META-INF") for n in names), (
        f"{_SAMPLE_MXL.name}: no inner .xml entry found"
    )


@pytest.mark.skipif(
    _SAMPLE_MXL is None,
    reason="No .mxl assets found in model/scores/v1/ -- run `just build-catalog-mxl` first",
)
def test_generated_mxl_renders_in_verovio() -> None:
    """The generated .mxl loads in Verovio and produces at least one page.

    This is the store's acceptance criterion: if Verovio can engrave the score,
    the web score panel can render it.
    """
    import os
    import tempfile

    assert _SAMPLE_MXL is not None
    mxl_bytes = _SAMPLE_MXL.read_bytes()

    with tempfile.NamedTemporaryFile(suffix=".mxl", delete=False) as tmp:
        tmp.write(mxl_bytes)
        tmp_path = tmp.name
    try:
        tk = verovio.toolkit()
        ok = tk.loadFile(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert ok, (
        f"verovio.toolkit().loadFile() returned False for {_SAMPLE_MXL.name}"
    )
    page_count = tk.getPageCount()
    assert page_count >= 1, (
        f"verovio engraved 0 pages for {_SAMPLE_MXL.name}"
    )


@pytest.mark.skipif(
    _SAMPLE_MXL is None,
    reason="No .mxl assets found in model/scores/v1/ -- run `just build-catalog-mxl` first",
)
def test_generated_mxl_has_no_doctype() -> None:
    """The inner XML of the generated .mxl has no DOCTYPE declaration.

    DOCTYPE causes Verovio's WASM parser to fail in sandboxed environments.
    """
    assert _SAMPLE_MXL is not None
    with zipfile.ZipFile(_SAMPLE_MXL, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("META-INF") and name.endswith(".xml"):
                inner = zf.read(name)
                assert b"<!DOCTYPE" not in inner, (
                    f"{_SAMPLE_MXL.name}: inner XML still contains DOCTYPE declaration"
                )
                break
