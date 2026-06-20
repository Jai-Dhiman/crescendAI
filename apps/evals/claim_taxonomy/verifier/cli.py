from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_DEFAULT_TAXONOMY = Path(__file__).resolve().parents[1] / "claim_taxonomy.json"


def _cmd_verify(args: argparse.Namespace) -> int:
    from claim_taxonomy.verifier.orchestrator import verify
    from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

    claim = json.loads(Path(args.claim).read_text())
    bundle = json.loads(Path(args.bundle).read_text())
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else _DEFAULT_TAXONOMY
    taxonomy = json.loads(taxonomy_path.read_text())

    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    print(json.dumps(dataclasses.asdict(result)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_taxonomy.verifier.cli")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify", help="Verify a single claim against a bundle")
    verify_parser.add_argument("--claim", required=True, help="Path to claim JSON file")
    verify_parser.add_argument("--bundle", required=True, help="Path to bundle JSON file")
    verify_parser.add_argument("--taxonomy", default=None,
                               help="Path to claim_taxonomy.json (default: committed taxonomy)")

    args = parser.parse_args(argv)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
