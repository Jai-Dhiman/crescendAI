"""Tests for wrap_as_mxl_zip — verifies ZIP structure and Verovio compatibility."""

from __future__ import annotations

import zipfile
import io

from score_library.upload import wrap_as_mxl_zip, _strip_doctype, _zip_xml_entry

SAMPLE_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC
    "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
    "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1"/>
  </part>
</score-partwise>
"""

PIECE_ID = "chopin.ballades.1"


# -- _strip_doctype -----------------------------------------------------------


def test_strip_doctype_removes_declaration() -> None:
    stripped = _strip_doctype(SAMPLE_XML)
    assert b"<!DOCTYPE" not in stripped
    assert b"<score-partwise" in stripped


def test_strip_doctype_idempotent() -> None:
    once = _strip_doctype(SAMPLE_XML)
    twice = _strip_doctype(once)
    assert once == twice


def test_strip_doctype_no_op_when_absent() -> None:
    xml = b"<score-partwise/>"
    assert _strip_doctype(xml) == xml


# -- wrap_as_mxl_zip ----------------------------------------------------------


def _make_zip(xml: bytes = SAMPLE_XML, piece_id: str = PIECE_ID) -> bytes:
    return wrap_as_mxl_zip(xml, piece_id)


def test_output_is_valid_zip() -> None:
    """zipfile.ZipFile can open the output without error."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.testzip() is None


def test_contains_required_entries() -> None:
    """ZIP must contain META-INF/container.xml and {piece_id}.xml."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    assert "META-INF/container.xml" in names
    assert f"{PIECE_ID}.xml" in names


def test_container_xml_references_root_file() -> None:
    """META-INF/container.xml must declare the piece_id root file."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        container = zf.read("META-INF/container.xml").decode()
    assert f'full-path="{PIECE_ID}.xml"' in container
    assert 'media-type="application/vnd.recordare.musicxml+xml"' in container


def test_xml_entry_is_readable() -> None:
    """The MusicXML entry must be valid UTF-8 and contain score content."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml_content = zf.read(f"{PIECE_ID}.xml").decode("utf-8")
    assert "<score-partwise" in xml_content


def test_doctype_is_stripped_from_xml_entry() -> None:
    """DOCTYPE must be absent from the stored XML entry."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml_content = zf.read(f"{PIECE_ID}.xml").decode("utf-8")
    assert "<!DOCTYPE" not in xml_content


def test_uses_deflate_compression() -> None:
    """Both entries should use DEFLATE (compress_type=8)."""
    data = _make_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED, (
                f"{info.filename}: expected DEFLATE, got compress_type={info.compress_type}"
            )


def test_local_header_has_correct_compressed_size() -> None:
    """Local file header at offset 18 must match actual compressed data size.

    The worker's extractXmlFromMxl reads compressedSize from offset 18 to
    slice the compressed bytes. Python's writestr() (unlike streaming writes)
    compresses in memory first and writes the correct size — no data descriptor.
    This test verifies that invariant by parsing the raw bytes directly.
    """
    data = _make_zip()
    offset = 0

    seen: list[tuple[str, int, int]] = []

    while offset + 30 <= len(data):
        sig = int.from_bytes(data[offset:offset + 4], "little")
        if sig != 0x04034b50:
            break

        flags = int.from_bytes(data[offset + 6:offset + 8], "little")
        method = int.from_bytes(data[offset + 8:offset + 10], "little")
        compressed_size = int.from_bytes(data[offset + 18:offset + 22], "little")
        fname_len = int.from_bytes(data[offset + 26:offset + 28], "little")
        extra_len = int.from_bytes(data[offset + 28:offset + 30], "little")

        fname = data[offset + 30:offset + 30 + fname_len].decode()
        data_start = offset + 30 + fname_len + extra_len

        assert flags & 8 == 0, (
            f"{fname}: data descriptor flag is set — writestr() should not produce this"
        )
        assert method == 8, f"{fname}: expected DEFLATE method 8, got {method}"

        actual_compressed = len(data[data_start:data_start + compressed_size])
        assert actual_compressed == compressed_size, (
            f"{fname}: header says {compressed_size} bytes but only "
            f"{actual_compressed} available — worker parser would read garbage"
        )

        seen.append((fname, compressed_size, data_start))
        offset = data_start + compressed_size

    assert len(seen) == 2, f"Expected 2 local file entries, found {len(seen)}: {seen}"
    fnames = {name for name, _, _ in seen}
    assert "META-INF/container.xml" in fnames
    assert f"{PIECE_ID}.xml" in fnames


def test_already_zip_passthrough() -> None:
    """wrap_as_mxl_zip must return existing ZIPs unchanged."""
    original = _make_zip()
    result = wrap_as_mxl_zip(original, PIECE_ID)
    assert result == original


def test_plain_xml_without_doctype() -> None:
    """Plain XML without DOCTYPE wraps correctly."""
    xml = b"<score-partwise version='3.1'><part-list/></score-partwise>"
    data = wrap_as_mxl_zip(xml, "bach.preludes.1")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        content = zf.read("bach.preludes.1.xml")
    assert b"<score-partwise" in content


# -- _zip_xml_entry -----------------------------------------------------------


def test_zip_xml_entry_returns_inner_xml() -> None:
    """Extracts the MusicXML bytes from a properly formed MXL ZIP."""
    data = _make_zip()
    xml = _zip_xml_entry(data)
    assert xml is not None
    assert b"<score-partwise" in xml


def test_zip_xml_entry_skips_meta_inf() -> None:
    """META-INF/container.xml is not returned."""
    data = _make_zip()
    xml = _zip_xml_entry(data)
    assert xml is not None
    assert b"rootfile" not in xml  # container.xml content, not the score


def test_zip_xml_entry_returns_none_on_bad_zip() -> None:
    """Returns None rather than raising on invalid ZIP bytes."""
    assert _zip_xml_entry(b"not a zip") is None


def test_zip_xml_entry_doctype_detection() -> None:
    """Caller can detect DOCTYPE by inspecting the returned bytes."""
    data = _make_zip()
    xml = _zip_xml_entry(data)
    assert xml is not None
    # wrap_as_mxl_zip strips DOCTYPE, so the extracted entry must be clean.
    assert b"<!DOCTYPE" not in xml


def test_rewrap_zip_with_doctype() -> None:
    """A ZIP whose XML entry still has DOCTYPE can be re-wrapped correctly.

    This simulates the R2 migration case: old uploads that skipped DOCTYPE
    stripping. The fix is to extract the XML from the existing ZIP and pass
    the raw XML bytes (not the ZIP bytes) to wrap_as_mxl_zip so _strip_doctype
    runs on them.
    """
    xml_with_doctype = SAMPLE_XML  # has DOCTYPE
    assert b"<!DOCTYPE" in xml_with_doctype

    # Simulate an old upload: build a ZIP without stripping DOCTYPE.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/container.xml", "")
        zf.writestr(f"{PIECE_ID}.xml", xml_with_doctype)
    old_zip = buf.getvalue()

    # Verify the old ZIP has DOCTYPE in its XML entry.
    inner = _zip_xml_entry(old_zip)
    assert inner is not None and b"<!DOCTYPE" in inner

    # Re-wrap: extract inner XML bytes and pass to wrap_as_mxl_zip.
    rewrapped = wrap_as_mxl_zip(inner, PIECE_ID)
    with zipfile.ZipFile(io.BytesIO(rewrapped)) as zf:
        fixed_xml = zf.read(f"{PIECE_ID}.xml")
    assert b"<!DOCTYPE" not in fixed_xml
    assert b"<score-partwise" in fixed_xml
