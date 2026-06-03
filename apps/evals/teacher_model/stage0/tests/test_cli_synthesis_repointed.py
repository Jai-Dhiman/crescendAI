# apps/evals/teacher_model/stage0/tests/test_cli_synthesis_repointed.py
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock


def test_cli_imports_without_run_synthesis() -> None:
    import importlib

    mod = importlib.import_module("teacher_model.stage0.cli")
    importlib.reload(mod)
    text = Path(mod.__file__).read_text()
    assert "run_synthesis" not in text
    assert "run_do_baseline" in text


def test_synthesis_subcommand_calls_run_do_baseline_with_judge_extended(tmp_path: Path) -> None:
    import teacher_model.stage0.cli as cli
    from teacher_model.stage0.judge_extended import judge_extended

    out = tmp_path / "synth.jsonl"
    argv = ["cli", "synthesis", "--provider", "anthropic",
            "--model", "claude-sonnet-4-6", "--out", str(out)]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch("teaching_knowledge.run_eval.run_do_baseline") as m:
        cli.main()
    assert m.called
    kwargs = m.call_args.kwargs
    assert kwargs["out_path"] == out
    # BLOCKER 2: preserve the 9-dim stage0 judge; do NOT swap to judge_synthesis_v2.
    assert kwargs["judge_fn"] is judge_extended
