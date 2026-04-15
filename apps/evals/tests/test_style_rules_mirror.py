from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_PATH = REPO_ROOT / "apps" / "evals" / "shared" / "data" / "style_rules.json"
TS_PATH = REPO_ROOT / "apps" / "api" / "src" / "lib" / "style-rules.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mirror_file_exists() -> None:
    assert TS_PATH.exists(), f"TS mirror missing at {TS_PATH}"


def test_python_and_ts_style_rules_are_byte_identical() -> None:
    assert _sha256(PY_PATH) == _sha256(TS_PATH), (
        f"Style rules drift detected!\n"
        f"  Python: {PY_PATH}\n"
        f"  TS:     {TS_PATH}\n"
        f"Run: cp {PY_PATH} {TS_PATH}"
    )
