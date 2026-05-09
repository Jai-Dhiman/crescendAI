"""Verify domain_knowledge_probe CLI accepts the openrouter provider."""
from __future__ import annotations

from pathlib import Path


def test_cli_accepts_openrouter_provider() -> None:
    """`--provider openrouter` must be listed in the source choices."""
    repo_root = Path(__file__).resolve().parents[4]
    src = (repo_root / "apps" / "evals" / "teacher_model" / "domain_knowledge_probe.py").read_text()
    assert '"openrouter"' in src, "openrouter must be in --provider choices"
