#!/usr/bin/env python3
"""
Build a JSONL manifest for an audio dataset directory.

Each line is a JSON object with fields:
- path: absolute file path
- rel_path: path relative to the source dir
- sha1: SHA1 hex digest of file content
- duration_sec: float duration in seconds (computed at target_sr)
- license: license string provided via CLI
- source: free-form dataset/source name or note
- performer_id: optional string (heuristic extraction if --with-performer)
- notes: optional string

Usage (examples):
- uv run python -m src.tools.build_manifest --source data/raw/percepiano --out data/manifests/percepiano.jsonl --license "CC-BY-4.0" --source-note "PercePiano" --with-performer
- uv run python -m src.tools.build_manifest --source data/raw/maestro-v3.0.0 --out data/manifests/maestro.jsonl --license "CC-BY-NC-4.0" --source-note "MAESTRO v3"

Explicit exceptions: any failure to read/parse an audio file is logged and raised at the end unless --skip-errors is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Use the project's canonical loader (enforces mono 22050)
try:
    from src.data.audio_io import load_audio_mono_22050
except Exception as e:  # explicit exception handling per user preference
    raise ImportError(
        "Failed to import src.data.audio_io.load_audio_mono_22050. Ensure the project environment is set up (uv sync)."
    ) from e

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".aac"}
TARGET_SR = 22050


def sha1_of_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_audio_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                yield p


@dataclass
class Args:
    source: Path
    out: Path
    license: str
    source_note: str
    with_performer: bool
    notes: Optional[str]
    skip_errors: bool
    max_files: Optional[int]


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Build a JSONL manifest for an audio dataset directory.")
    ap.add_argument("--source", type=Path, help="Root directory to scan for audio files")
    ap.add_argument("--root", type=Path, help="Alias for --source")
    ap.add_argument("--out", required=True, type=Path, help="Output JSONL manifest path")
    ap.add_argument("--license", required=True, type=str, help="License string to record in manifest entries")
    ap.add_argument("--source-note", default="", type=str, help="Short note identifying the dataset/source")
    ap.add_argument("--with-performer", action="store_true", help="Attempt heuristic performer_id extraction from path/filename")
    ap.add_argument("--notes", default=None, type=str, help="Optional global notes to include per entry")
    ap.add_argument("--skip-errors", action="store_true", help="Skip files that fail to parse instead of raising at end")
    ap.add_argument("--max-files", type=int, default=None, help="Optional limit for number of files to index (for pilots)")
    ap.add_argument("--file-list", type=Path, default=None, help="Optional text file with relative paths to include (one per line)")
    ns = ap.parse_args()
    # Resolve source: prefer --root if provided
    src = ns.root if ns.root is not None else ns.source
    if src is None:
        ap.error("--source or --root is required")
    return Args(
        source=src.resolve(),
        out=ns.out.resolve(),
        license=ns.license,
        source_note=ns.source_note,
        with_performer=ns.with_performer,
        notes=ns.notes,
        skip_errors=ns.skip_errors,
        max_files=ns.max_files,
    )


def extract_performer_id(path: Path) -> Optional[str]:
    # Best-effort heuristic: take parent directory if it looks like a performer bucket
    # Users can post-process manifests to refine this mapping.
    parts = list(path.parts)
    if len(parts) >= 2:
        parent = path.parent.name
        # Simple filters; adjust as needed
        if any(tok in parent.lower() for tok in ("performer", "pianist", "artist", "player")):
            return parent
        # Otherwise, if parent contains alphabetic characters and not generic names
        if any(c.isalpha() for c in parent) and parent.lower() not in {"audio", "wav", "mp3", "flac", "data"}:
            return parent
    # Fallback: stem prefix up to first '-' or '_' if it seems name-like
    stem = path.stem
    for sep in ("-", "_"):
        if sep in stem:
            cand = stem.split(sep)[0]
            if cand and any(c.isalpha() for c in cand):
                return cand
    return None


def main() -> None:
    args = parse_args()

    if not args.source.exists() or not args.source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_audio_files(args.source, AUDIO_EXTS))
    if not files:
        raise FileNotFoundError(f"No audio files found under: {args.source}")
    # Optional filter by file-list (relative paths)
    import sys
    file_list_path = None
    for i, a in enumerate(sys.argv):
        if a == "--file-list" and i + 1 < len(sys.argv):
            file_list_path = sys.argv[i + 1]
            break
    if file_list_path:
        try:
            with open(file_list_path, "r", encoding="utf-8") as f:
                wanted = {line.strip() for line in f if line.strip()}
        except Exception as e:
            raise RuntimeError(f"Failed to read --file-list {file_list_path}: {e}")
        # Keep only files whose relative path to source is in list
        kept = []
        for p in files:
            try:
                rel = str(p.resolve().relative_to(args.source))
            except Exception:
                rel = p.name
            if rel in wanted:
                kept.append(p)
        files = kept
    if args.max_files is not None:
        files = files[: max(0, args.max_files)]
    failures: List[str] = []

    with args.out.open("w", encoding="utf-8") as fout:
        for p in files:
            try:
                sha1 = sha1_of_file(p)
                # Use canonical loader to enforce mono 22050 (duration computed at TARGET_SR)
                y = load_audio_mono_22050(p, target_sr=TARGET_SR)
                if y is None or not isinstance(y, np.ndarray) or y.size == 0:
                    raise ValueError(f"Loaded empty audio: {p}")
                duration = float(len(y) / float(TARGET_SR))
                rel = str(p.relative_to(args.source)) if p.is_relative_to(args.source) else p.name
                entry = {
                    "path": str(p.resolve()),
                    "rel_path": rel,
                    "sha1": sha1,
                    "duration_sec": duration,
                    "license": args.license,
                    "source": args.source_note or "",
                }
                if args.with_performer:
                    perf = extract_performer_id(p)
                    if perf is not None:
                        entry["performer_id"] = perf
                if args.notes is not None:
                    entry["notes"] = args.notes
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                failures.append(f"{p}: {e}")
                if args.skip_errors:
                    continue

    if failures and not args.skip_errors:
        msg = "\n".join(failures[:10])
        raise RuntimeError(
            f"Encountered {len(failures)} file errors during manifest build. First few errors:\n{msg}"
        )


if __name__ == "__main__":
    main()
