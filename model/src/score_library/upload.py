"""Upload score library to R2 and generate D1 seed SQL."""
from __future__ import annotations

import json
import os
from pathlib import Path


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
        "INSERT OR REPLACE INTO scores (\n"
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
