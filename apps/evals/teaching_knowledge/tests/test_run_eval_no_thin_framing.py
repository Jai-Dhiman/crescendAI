# apps/evals/teaching_knowledge/tests/test_run_eval_no_thin_framing.py
from __future__ import annotations

import teaching_knowledge.run_eval as run_eval


def test_thin_framing_builder_is_gone() -> None:
    assert not hasattr(run_eval, "build_synthesis_user_msg")


def test_module_imports_without_bar_analysis_local() -> None:
    # Importing run_eval must not require the deleted bar_analysis_local module.
    import importlib

    mod = importlib.reload(run_eval)
    src = mod.__file__
    text = open(src).read()
    assert "bar_analysis_local" not in text
    assert "piece_score_map" not in text
