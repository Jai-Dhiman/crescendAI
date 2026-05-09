"""Stage 0 needs transformers + jsonschema in a dedicated extras group."""
from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def test_teacher_model_stage0_extras_present() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    extras = data.get("project", {}).get("optional-dependencies", {})
    assert "teacher-model-stage0" in extras
    deps_str = " ".join(extras["teacher-model-stage0"])
    assert "transformers" in deps_str
    assert "jsonschema" in deps_str
