"""Smoke tests for shared/local_session.py — importable, helpers work without services."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

import pytest

from shared.local_session import read_eval_secret, SessionCapture, drive


def test_session_capture_importable():
    """SessionCapture must be importable from local_session (re-exported from score.py)."""
    cap = SessionCapture(
        session_id="test",
        recording=Path("dummy.wav"),
        piece_slug="fur_elise",
        teaching_moments=[],
        baselines={},
        piece_identification=None,
        piece_resolved=False,
        dominant_dimension=None,
        prescribed_exercise=None,
        synthesis_text="",
    )
    assert cap.session_id == "test"


def test_read_eval_secret_raises_on_missing_file(tmp_path: Path):
    """read_eval_secret raises FileNotFoundError when .dev.vars does not exist."""
    missing = tmp_path / "nonexistent.vars"
    with pytest.raises(FileNotFoundError, match="apps/api/.dev.vars not found"):
        read_eval_secret(dev_vars=missing)


def test_read_eval_secret_raises_on_missing_key(tmp_path: Path):
    """read_eval_secret raises KeyError when EVAL_SHARED_SECRET is not in file."""
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text("OTHER_VAR=something\n")
    with pytest.raises(KeyError, match="EVAL_SHARED_SECRET not present"):
        read_eval_secret(dev_vars=dev_vars)


def test_read_eval_secret_raises_on_empty_value(tmp_path: Path):
    """read_eval_secret raises ValueError when EVAL_SHARED_SECRET is present but empty."""
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text('EVAL_SHARED_SECRET=\n')
    with pytest.raises(ValueError, match="EVAL_SHARED_SECRET is empty"):
        read_eval_secret(dev_vars=dev_vars)


def test_read_eval_secret_returns_value(tmp_path: Path):
    """read_eval_secret returns the secret when present and non-empty."""
    dev_vars = tmp_path / ".dev.vars"
    dev_vars.write_text('EVAL_SHARED_SECRET="my-secret-value"\n')
    assert read_eval_secret(dev_vars=dev_vars) == "my-secret-value"


def test_drive_callable():
    """drive is importable and callable (checking signature only — not running it)."""
    import inspect
    sig = inspect.signature(drive)
    assert "recording" in sig.parameters
    assert "piece_slug" in sig.parameters
