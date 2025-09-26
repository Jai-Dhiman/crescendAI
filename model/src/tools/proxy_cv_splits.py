#!/usr/bin/env python3
"""
Create piece-aware CV folds for window-level proxies using MAESTRO CSV metadata.

Inputs:
- proxies JSONL from src.tools.build_proxies (must contain audio_path and start_frame)
- MAESTRO CSV path (to map rel audio_filename -> (composer, title))
- dataset root path (to compute relpath = audio_path relative to root)

Outputs:
- JSONL with added field "fold" per proxy record (written to --out-jsonl)
- Summary JSON with fold counts and piece stats (written to --out-summary)

Usage:
uv run python -m src.tools.proxy_cv_splits \
  --proxies /content/data/proxies/maestro_train_half_windows.jsonl \
  --maestro-csv /content/data/maestro-v3.0.0/maestro-v3.0.0.csv \
  --root /content/data/maestro-v3.0.0 \
  --out-jsonl /content/data/proxies/maestro_windows_cv.jsonl \
  --out-summary /content/data/reports/maestro_cv_summary.json \
  --folds 5
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Args:
    proxies: Path
    maestro_csv: Path
    root: Path
    out_jsonl: Path
    out_summary: Path
    folds: int


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Piece-aware CV folds for proxies using MAESTRO CSV")
    ap.add_argument("--proxies", required=True, type=Path)
    ap.add_argument("--maestro-csv", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ns = ap.parse_args()
    return Args(
        proxies=ns.proxies.resolve(),
        maestro_csv=ns.maestro_csv.resolve(),
        root=ns.root.resolve(),
        out_jsonl=ns.out_jsonl.resolve(),
        out_summary=ns.out_summary.resolve(),
        folds=int(ns.folds),
    )


def load_maestro_map(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """Return mapping relpath -> (composer, title)."""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        m: Dict[str, Tuple[str, str]] = {}
        for r in dr:
            rel = (r.get("audio_filename") or "").strip()
            comp = (r.get("canonical_composer") or r.get("composer") or "Unknown").strip()
            title = (r.get("canonical_title") or r.get("title") or "Unknown").strip()
            if rel:
                m[rel] = (comp, title)
    if not m:
        raise RuntimeError("Empty MAESTRO CSV mapping; check columns and path")
    return m


def relpath_from_root(abs_path: str, root: Path) -> str:
    p = Path(abs_path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        # Fallback: string prefix removal
        s_root = str(root.resolve())
        s_abs = str(p.resolve())
        if s_abs.startswith(s_root):
            r = s_abs[len(s_root) :].lstrip("/")
            return r
        return p.name


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    rel_to_meta = load_maestro_map(args.maestro_csv)

    # Read proxies JSONL and attach piece_id
    items: List[dict] = []
    with args.proxies.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            apath = obj.get("audio_path") or obj.get("path")
            if not apath:
                continue
            rel = relpath_from_root(apath, args.root)
            comp, title = rel_to_meta.get(rel, ("Unknown", "Unknown"))
            piece_id = f"{comp}::{title}"
            obj["relpath"] = rel
            obj["piece_id"] = piece_id
            items.append(obj)

    if not items:
        raise RuntimeError("No proxies found or none mapped to MAESTRO CSV")

    # Assign folds by piece_id (round-robin deterministic)
    unique_pieces = sorted({obj["piece_id"] for obj in items})
    k = max(2, int(args.folds))
    piece_to_fold: Dict[str, int] = {}
    for idx, pid in enumerate(unique_pieces):
        piece_to_fold[pid] = idx % k

    # Write JSONL with fold field
    fold_counts = [0] * k
    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        for obj in items:
            f = piece_to_fold[obj["piece_id"]]
            obj_out = dict(obj)
            obj_out["fold"] = int(f)
            fout.write(json.dumps(obj_out) + "\n")
            fold_counts[f] += 1

    # Summary JSON
    summary = {
        "folds": k,
        "num_items": len(items),
        "num_pieces": len(unique_pieces),
        "fold_counts": fold_counts,
    }
    with args.out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Wrote:", args.out_jsonl)
    print("Summary:", args.out_summary)


if __name__ == "__main__":
    main()
