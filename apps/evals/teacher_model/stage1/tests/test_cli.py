import subprocess
import sys

import pytest


@pytest.mark.parametrize("subcmd", ["holdout", "distill", "coverage", "render", "harness"])
def test_cli_subcommand_help(subcmd):
    result = subprocess.run(
        [sys.executable, "-m", "teacher_model.stage1", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert subcmd in result.stdout.lower() or "usage" in result.stdout.lower()
