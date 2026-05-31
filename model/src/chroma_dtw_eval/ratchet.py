"""Human-invoked CLI to update the committed baseline from a verify sidecar.

Refuses to write if the sidecar shows any regression — the user must first
investigate the regression and decide whether to accept it (rare, requires
manual edit of the sidecar's `regressed` field) or revert the change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.ratchet")
    parser.add_argument("--from", dest="src", required=True, type=Path)
    parser.add_argument("--to", dest="dst", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.src.exists():
        raise FileNotFoundError(f"sidecar not found: {args.src}")
    data = json.loads(args.src.read_text())
    if data.get("regressed"):
        print(f"refusing to ratchet: regressed={data['regressed']}", file=sys.stderr)
        return 2
    out = {"primary": data["primary"], "guards": data["guards"]}
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
