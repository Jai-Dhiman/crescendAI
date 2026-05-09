# tests/unit/test_cli.py
"""Verifies CLI exposes a tick command callable from typer's runner."""
from typer.testing import CliRunner
from content_engine.cli import app


def test_tick_command_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTENT_ENGINE_DB", str(tmp_path / "e.sqlite"))
    runner = CliRunner()
    result = runner.invoke(app, ["tick", "--dry-run"])
    assert result.exit_code == 0


def test_help_lists_known_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tick" in result.output
