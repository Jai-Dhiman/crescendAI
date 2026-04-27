# scripts/compile_playbook.py
"""Compile shared/teacher-style/playbook.yaml -> apps/api/src/lib/playbook.json."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "shared" / "teacher-style" / "playbook.yaml"
DST = REPO_ROOT / "apps" / "api" / "src" / "lib" / "playbook.json"


def _serialize() -> str:
    data = yaml.safe_load(SRC.read_text())
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = _serialize()
    if args.check:
        if not DST.exists() or DST.read_text() != text:
            print(f"DRIFT: {DST} stale. Run: python scripts/compile_playbook.py", file=sys.stderr)
            return 1
        return 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(text)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
