# apps/evals/teacher_model/stage0/tests/test_no_run_synthesis.py
from __future__ import annotations

import importlib
from pathlib import Path


def test_stage0_run_synthesis_module_deleted() -> None:
    here = Path(__file__).resolve().parents[1]  # stage0/
    assert not (here / "run_synthesis.py").exists()


def test_stage0_run_synthesis_not_importable() -> None:
    try:
        importlib.import_module("teacher_model.stage0.run_synthesis")
    except ModuleNotFoundError:
        return
    raise AssertionError("teacher_model.stage0.run_synthesis should not be importable")
