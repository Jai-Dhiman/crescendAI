# apps/evals/teacher_model/calibration/tests/test_regen_calibration_repointed.py
from __future__ import annotations

from pathlib import Path

import teacher_model.calibration.regen_calibration_baseline as regen


def test_regen_no_longer_uses_thin_framing_symbols() -> None:
    text = Path(regen.__file__).read_text()
    assert "build_synthesis_user_msg" not in text
    assert "extract_teacher_response" not in text


def test_regen_builds_row_through_build_do_row() -> None:
    # The repointed regen routes synthesis through build_do_row (the DO path),
    # so the module must reference it.
    text = Path(regen.__file__).read_text()
    assert "build_do_row" in text
