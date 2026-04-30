"""Asserts that SESSION_SYNTHESIS_SYSTEM in prompts.ts matches synthesis_system.txt.

If this test fails, run:
    cp apps/shared/teacher-style/synthesis_system.txt <updated content>
and update the SESSION_SYNTHESIS_SYSTEM constant in apps/api/src/services/prompts.ts to match.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_PATH = REPO_ROOT / "apps" / "shared" / "teacher-style" / "synthesis_system.txt"
TS_PATH = REPO_ROOT / "apps" / "api" / "src" / "services" / "prompts.ts"

_PATTERN = re.compile(
    r"export const SESSION_SYNTHESIS_SYSTEM\s*=\s*`(.*?)`;",
    re.DOTALL,
)


def _extract_ts_constant() -> str:
    ts_content = TS_PATH.read_text()
    match = _PATTERN.search(ts_content)
    assert match, f"Could not find SESSION_SYNTHESIS_SYSTEM in {TS_PATH}"
    return match.group(1).strip()


def test_shared_file_exists() -> None:
    assert SHARED_PATH.exists(), f"Shared prompt file missing at {SHARED_PATH}"


def test_synthesis_prompt_in_sync() -> None:
    shared = SHARED_PATH.read_text().strip()
    ts = _extract_ts_constant()
    assert shared == ts, (
        "SESSION_SYNTHESIS_SYSTEM drift detected!\n"
        f"  Shared: {SHARED_PATH}\n"
        f"  TS:     {TS_PATH}\n"
        "Update one to match the other."
    )
