"""Upload score library to R2 and generate D1 seed SQL."""
from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

from score_library.discover import derive_piece_id

_ZIP_MAGIC = b"PK\x03\x04"


def _strip_doctype(xml_bytes: bytes) -> bytes:
    """Remove DOCTYPE declarations from XML bytes.

    Verovio's WASM XML parser may throw on external DTD references
    (e.g. the MusicXML 1.1 DTD at musicxml.org) in sandboxed environments.
    Stripping DOCTYPE is safe: Verovio does not use DTD validation.
    """
    import re
    text = xml_bytes.decode("utf-8", errors="replace")
    cleaned = re.sub(r"<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>", "", text)
    return cleaned.encode("utf-8")


def wrap_as_mxl_zip(xml_bytes: bytes, piece_id: str) -> bytes:
    """Wrap plain MusicXML bytes in a standard MXL ZIP container.

    R2 object format (scores/v1/{piece_id}.mxl):
      - ZIP archive (starts with PK\\x03\\x04 magic)
      - Two entries, written in this order:
          META-INF/container.xml  — rootfiles declaration per MusicXML 3.0+ spec
          {piece_id}.xml          — DOCTYPE-stripped MusicXML content
      - Both entries use DEFLATE compression (method 8)
      - Local file headers carry correct compressedSize (no data descriptor / flag bit 3)
        because writestr() compresses in memory before writing the header.
        The web worker's extractXmlFromMxl() relies on this invariant to slice
        compressed bytes without reading the central directory.

    If xml_bytes already starts with the ZIP magic bytes it is returned
    unchanged, so this function is safe to call on any input.
    """
    if xml_bytes[:4] == _ZIP_MAGIC:
        return xml_bytes

    xml_bytes = _strip_doctype(xml_bytes)

    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<container>\n"
        "  <rootfiles>\n"
        f'    <rootfile full-path="{piece_id}.xml"'
        ' media-type="application/vnd.recordare.musicxml+xml"/>\n'
        "  </rootfiles>\n"
        "</container>\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr(f"{piece_id}.xml", xml_bytes)
    return buf.getvalue()


def generate_d1_seed(source_dir: Path, output_path: Path | None = None) -> Path:
    """Generate INSERT SQL from parsed score JSON files.

    Reads each JSON, extracts catalog fields, and writes a batch
    INSERT OR REPLACE statement for the D1 scores table.
    """
    if output_path is None:
        output_path = source_dir / "seed.sql"

    json_files = sorted(
        f for f in source_dir.glob("*.json")
        if f.name not in ("titles.json", "seed.sql")
    )
    if not json_files:
        raise FileNotFoundError(f"No score JSON files found in {source_dir}")

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        piece_id = data["piece_id"]
        composer = data["composer"]
        title = data["title"].replace("'", "''")
        key_signature = data.get("key_signature")
        total_bars = data["total_bars"]

        # Time signature from the first entry
        time_sigs = data.get("time_signatures", [])
        if time_sigs:
            ts = time_sigs[0]
            time_signature = f"{ts['numerator']}/{ts['denominator']}"
        else:
            time_signature = None
        has_time_sig_changes = len(time_sigs) > 1

        # Tempo from the first entry
        tempo_markings = data.get("tempo_markings", [])
        tempo_bpm = int(tempo_markings[0]["bpm"]) if tempo_markings else None
        has_tempo_changes = len(tempo_markings) > 1

        # Duration = last note's onset_seconds + duration_seconds
        duration_seconds = None
        for bar in reversed(data.get("bars", [])):
            if bar["notes"]:
                last_note = max(bar["notes"], key=lambda n: n["onset_seconds"] + n["duration_seconds"])
                duration_seconds = round(last_note["onset_seconds"] + last_note["duration_seconds"], 2)
                break

        # Note count and pitch range across all bars
        total_notes = 0
        pitch_low = None
        pitch_high = None
        for bar in data.get("bars", []):
            total_notes += bar["note_count"]
            pr = bar.get("pitch_range", [])
            if len(pr) == 2:
                if pitch_low is None or pr[0] < pitch_low:
                    pitch_low = pr[0]
                if pitch_high is None or pr[1] > pitch_high:
                    pitch_high = pr[1]

        # Build SQL value tokens
        key_sig_sql = f"'{key_signature}'" if key_signature else "NULL"
        time_sig_sql = f"'{time_signature}'" if time_signature else "NULL"
        tempo_sql = str(tempo_bpm) if tempo_bpm is not None else "NULL"
        duration_sql = str(duration_seconds) if duration_seconds is not None else "NULL"
        pitch_low_sql = str(pitch_low) if pitch_low is not None else "NULL"
        pitch_high_sql = str(pitch_high) if pitch_high is not None else "NULL"
        has_ts_sql = "1" if has_time_sig_changes else "0"
        has_tc_sql = "1" if has_tempo_changes else "0"

        row = (
            f"  ('{piece_id}', '{composer}', '{title}', {key_sig_sql}, "
            f"{time_sig_sql}, {tempo_sql}, {total_bars}, {duration_sql}, "
            f"{total_notes}, {pitch_low_sql}, {pitch_high_sql}, "
            f"{has_ts_sql}, {has_tc_sql}, 'asap')"
        )
        rows.append(row)

    sql = (
        "INSERT OR REPLACE INTO pieces (\n"
        "  piece_id, composer, title, key_signature,\n"
        "  time_signature, tempo_bpm, bar_count, duration_seconds,\n"
        "  note_count, pitch_range_low, pitch_range_high,\n"
        "  has_time_sig_changes, has_tempo_changes, source\n"
        ") VALUES\n"
    )
    sql += ",\n".join(rows) + ";\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(sql)

    print(f"Generated D1 seed SQL: {output_path} ({len(rows)} rows)")
    return output_path


def upload_to_r2(source_dir: Path, version: str = "v1") -> None:
    """Upload score JSON files to Cloudflare R2 via S3-compatible API.

    Requires environment variables:
        R2_ACCOUNT_ID
        R2_ACCESS_KEY_ID
        R2_SECRET_ACCESS_KEY
    """
    import boto3

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    bucket = "crescendai-bucket"
    json_files = sorted(
        f for f in source_dir.glob("*.json")
        if f.name not in ("titles.json", "seed.sql")
    )
    if not json_files:
        raise FileNotFoundError(f"No score JSON files found in {source_dir}")

    uploaded = 0
    for jf in json_files:
        piece_id = jf.stem
        key = f"scores/{version}/{piece_id}.json"
        s3.upload_file(str(jf), bucket, key, ExtraArgs={"ContentType": "application/json"})
        uploaded += 1

    if uploaded != len(json_files):
        raise RuntimeError(f"Upload count mismatch: uploaded {uploaded}, expected {len(json_files)}")

    print(f"Uploaded {uploaded} scores to R2 (bucket={bucket}, prefix=scores/{version}/)")


def _zip_xml_entry(zip_bytes: bytes) -> bytes | None:
    """Return the raw XML bytes of the MusicXML entry inside an MXL ZIP.

    Returns None if no matching entry is found or the ZIP is invalid.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name.startswith("META-INF") and name.endswith(".xml"):
                    return zf.read(name)
    except zipfile.BadZipFile:
        pass
    return None


def reupload_plain_xml_in_r2(version: str = "v1", dry_run: bool = False) -> None:
    """Re-wrap .mxl objects in R2 that are plain XML or contain a DOCTYPE declaration.

    R2 objects at ``scores/{version}/*.mxl`` must be proper MXL ZIP archives
    with DOCTYPE stripped from the inner XML entry. This function:
      1. Downloads each object and checks the ZIP magic bytes.
      2. For plain-XML files: wraps as MXL ZIP (strips DOCTYPE in the process).
      3. For existing ZIPs: extracts the XML entry and checks for DOCTYPE.
         If found, re-wraps the extracted XML (stripping DOCTYPE).

    Idempotent: objects that are already correct ZIPs with no DOCTYPE are skipped.

    Args:
        version: R2 key version prefix (default: ``v1``).
        dry_run: If True, print what would be re-uploaded without doing it.

    Requires environment variables:
        R2_ACCOUNT_ID
        R2_ACCESS_KEY_ID
        R2_SECRET_ACCESS_KEY
    """
    import boto3

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    bucket = "crescendai-bucket"
    prefix = f"scores/{version}/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    # Map key -> raw downloaded bytes for files that need re-wrapping.
    # Value is the bytes to pass to wrap_as_mxl_zip (XML bytes, not the ZIP itself).
    needs_rewrap: dict[str, bytes] = {}
    clean: int = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".mxl"):
                continue
            response = s3.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()

            if data[:4] != _ZIP_MAGIC:
                # Plain XML — pass directly, wrap_as_mxl_zip will strip DOCTYPE.
                needs_rewrap[key] = data
            else:
                xml_bytes = _zip_xml_entry(data)
                if xml_bytes is None or b"<!DOCTYPE" in xml_bytes:
                    # ZIP with missing/bad XML entry or DOCTYPE present — re-wrap.
                    needs_rewrap[key] = xml_bytes if xml_bytes is not None else data
                else:
                    clean += 1

    total = clean + len(needs_rewrap)
    print(f"Scanned {total} .mxl objects in R2")
    print(f"  Clean (ZIP, no DOCTYPE): {clean}")
    print(f"  Needs re-wrap:           {len(needs_rewrap)}")

    if not needs_rewrap:
        print("Nothing to do.")
        return

    if dry_run:
        print("\nDry-run — would re-upload:")
        for key in needs_rewrap:
            print(f"  [dry] {key}")
        return

    reuploaded = 0
    errors = 0
    for key, xml_bytes in needs_rewrap.items():
        piece_id = key.removeprefix(prefix).removesuffix(".mxl")
        try:
            mxl_bytes = wrap_as_mxl_zip(xml_bytes, piece_id)
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=mxl_bytes,
                ContentType="application/vnd.recordare.musicxml+zip",
            )
            reuploaded += 1
            if reuploaded % 10 == 0:
                print(f"  Re-uploaded {reuploaded}/{len(needs_rewrap)}...")
        except Exception as e:
            print(f"  ERROR {key}: {e}")
            errors += 1

    print(f"\nDone. Re-uploaded {reuploaded} files.")
    if errors:
        print(f"Errors: {errors}")


def upload_mxl_to_r2(asap_dir: Path, version: str = "v1", dry_run: bool = False) -> None:
    """Upload MusicXML score files from the ASAP dataset to Cloudflare R2.

    Walks the ASAP directory tree looking for ``xml_score.musicxml`` files
    (one per piece directory). Each file is wrapped into a standard MXL ZIP
    container before upload (see ``wrap_as_mxl_zip``), so all objects stored
    in R2 are proper ``.mxl`` ZIP files regardless of source format.

    R2 key: ``scores/{version}/{piece_id}.mxl``
    Content-Type: ``application/vnd.recordare.musicxml+zip``

    Args:
        asap_dir: Root of the cloned ASAP dataset (e.g. ``data/raw/asap``).
        version: R2 key version prefix (default: ``v1``).
        dry_run: If True, print what would be uploaded without doing it.

    Requires environment variables (when not dry_run):
        R2_ACCOUNT_ID
        R2_ACCESS_KEY_ID
        R2_SECRET_ACCESS_KEY
    """
    if not asap_dir.exists():
        raise FileNotFoundError(f"ASAP directory not found: {asap_dir}")

    musicxml_files: list[tuple[str, Path]] = []
    for mxl_path in sorted(asap_dir.rglob("xml_score.musicxml")):
        piece_dir = mxl_path.parent
        piece_id = derive_piece_id(piece_dir, asap_dir)
        musicxml_files.append((piece_id, mxl_path))

    if not musicxml_files:
        raise FileNotFoundError(
            f"No xml_score.musicxml files found in {asap_dir}. "
            "Check that the ASAP dataset is checked out with: "
            "git sparse-checkout set '**/xml_score.musicxml'"
        )

    print(f"Found {len(musicxml_files)} xml_score.musicxml files (will wrap each as MXL ZIP)")

    if dry_run:
        for piece_id, path in musicxml_files[:10]:
            print(f"  [dry] scores/{version}/{piece_id}.mxl <- {path.relative_to(asap_dir)}")
        if len(musicxml_files) > 10:
            print(f"  ... and {len(musicxml_files) - 10} more")
        return

    bucket = "crescendai-bucket"
    uploaded = 0
    skipped = 0

    for piece_id, mxl_path in musicxml_files:
        object_path = f"{bucket}/scores/{version}/{piece_id}.mxl"
        try:
            xml_bytes = mxl_path.read_bytes()
            mxl_bytes = wrap_as_mxl_zip(xml_bytes, piece_id)
            result = subprocess.run(
                [
                    "wrangler", "r2", "object", "put", object_path,
                    "--pipe", "--remote",
                    "--content-type", "application/vnd.recordare.musicxml+zip",
                ],
                input=mxl_bytes,
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode().strip())
            uploaded += 1
            if uploaded % 50 == 0:
                print(f"  Uploaded {uploaded}/{len(musicxml_files)}...")
        except Exception as e:
            print(f"  SKIP {piece_id}: {e}")
            skipped += 1

    print(f"\nDone. Uploaded {uploaded} MXL ZIP files to R2 (bucket={bucket}, prefix=scores/{version}/)")
    if skipped:
        print(f"Skipped {skipped} files due to errors")
