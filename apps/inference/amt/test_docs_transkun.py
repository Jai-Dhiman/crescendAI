"""Guard: transcriber docs name Transkun. Run: cd apps/inference/amt && \
        uv run --with pytest pytest test_docs_transkun.py"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # apps/inference/amt/<file> -> repo root
DOCS = [
    "docs/apps/07-evaluation.md",
    "docs/model/01-data.md",
    "docs/architecture.md",
]


def test_docs_mention_transkun():
    missing = [d for d in DOCS if "ranskun" not in (REPO / d).read_text()]
    assert missing == [], f"docs not updated to Transkun: {missing}"
