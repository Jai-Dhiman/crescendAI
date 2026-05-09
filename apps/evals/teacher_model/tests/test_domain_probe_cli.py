"""Verify domain_knowledge_probe CLI accepts the openrouter provider."""
from __future__ import annotations

import subprocess
import sys


def test_cli_accepts_openrouter_provider() -> None:
    """`--provider openrouter` must appear in the live argparse --help output."""
    result = subprocess.run(
        [sys.executable, "-m", "teacher_model.domain_knowledge_probe", "--help"],
        capture_output=True,
        text=True,
    )
    assert "openrouter" in result.stdout, (
        "openrouter must be in --provider choices; not found in --help output"
    )
