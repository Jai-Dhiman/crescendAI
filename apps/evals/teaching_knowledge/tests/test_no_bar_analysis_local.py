# apps/evals/teaching_knowledge/tests/test_no_bar_analysis_local.py
from __future__ import annotations

from pathlib import Path


def test_bar_analysis_local_module_deleted() -> None:
    here = Path(__file__).resolve().parents[1]  # teaching_knowledge/
    assert not (here / "bar_analysis_local.py").exists()


def test_bar_analysis_local_not_importable() -> None:
    import importlib

    try:
        importlib.import_module("teaching_knowledge.bar_analysis_local")
    except ModuleNotFoundError:
        return
    raise AssertionError("teaching_knowledge.bar_analysis_local should not be importable")
