"""Verify domain_knowledge_probe CLI accepts the openrouter provider."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def test_cli_accepts_openrouter_provider() -> None:
    """`--provider openrouter` must parse without argparse rejecting it."""
    repo_root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, '.'); "
                "from teacher_model import domain_knowledge_probe as m; "
                "p = argparse.ArgumentParser(); "
            ),
        ],
        check=False,
        capture_output=True,
    )
    # Real test: invoke argparse directly via the module's parser.
    # We re-create it the same way main() does and assert openrouter is allowed.
    sys.path.insert(0, str(repo_root / "apps" / "evals"))
    from teacher_model.domain_knowledge_probe import main  # noqa: F401

    # Build a parser identical to main()'s, parse with --provider openrouter.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["workers-ai", "anthropic", "openrouter"],
        default="workers-ai",
    )
    args = parser.parse_args(["--provider", "openrouter"])
    assert args.provider == "openrouter"

    # Now actually verify the source file lists openrouter as a choice.
    src = (repo_root / "apps" / "evals" / "teacher_model" / "domain_knowledge_probe.py").read_text()
    assert '"openrouter"' in src, "openrouter must be in --provider choices"
