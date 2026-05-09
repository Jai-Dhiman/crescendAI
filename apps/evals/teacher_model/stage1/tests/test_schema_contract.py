import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TOOL_PROCESSOR_TS = REPO_ROOT / "apps/api/src/services/tool-processor.ts"
FINGERPRINT_PATH = (
    REPO_ROOT
    / "apps/evals/teacher_model/stage1/data/tool_processor_fingerprint.json"
)

_TOOL_NAMES = (
    "create_exercise",
    "score_highlight",
    "keyboard_guide",
    "show_session_data",
    "reference_browser",
    "search_catalog",
)


def _slice_for_tool(source: str, tool_name: str) -> str:
    pattern = rf"// Tool: {re.escape(tool_name)}\b"
    match = re.search(pattern, source)
    if match is None:
        raise AssertionError(
            f"Tool section marker missing for {tool_name} in tool-processor.ts. "
            f"Either the section was renamed or the tool was removed; "
            f"update both stage1/schema.py and the fingerprint snapshot."
        )
    start = match.start()
    end = len(source)
    for other in _TOOL_NAMES:
        if other == tool_name:
            continue
        other_match = re.search(rf"// Tool: {re.escape(other)}\b", source[start + 1 :])
        if other_match is not None:
            end = min(end, start + 1 + other_match.start())
    return source[start:end]


def test_tool_processor_fingerprint_matches_snapshot():
    source = TOOL_PROCESSOR_TS.read_text()
    fingerprint: dict[str, str] = {}
    for tool_name in _TOOL_NAMES:
        section = _slice_for_tool(source, tool_name)
        fingerprint[tool_name] = hashlib.sha256(section.encode()).hexdigest()

    if not FINGERPRINT_PATH.exists():
        pytest.fail(
            f"Snapshot missing at {FINGERPRINT_PATH}. "
            f"Initial content (commit alongside Pydantic mirror):\n"
            f"{json.dumps(fingerprint, indent=2)}"
        )

    snapshot = json.loads(FINGERPRINT_PATH.read_text())
    drift = {
        name: (snapshot.get(name), fingerprint[name])
        for name in _TOOL_NAMES
        if snapshot.get(name) != fingerprint[name]
    }
    if drift:
        msg_lines = [
            "tool-processor.ts diverged from the Stage 1 Pydantic mirror snapshot.",
            "If this is intentional, update apps/evals/teacher_model/stage1/schema.py "
            "to match, then update the fingerprint:",
            json.dumps(fingerprint, indent=2),
            "Drifted tools:",
        ]
        for name, (old, new) in drift.items():
            msg_lines.append(f"  {name}: {old} -> {new}")
        pytest.fail("\n".join(msg_lines))
