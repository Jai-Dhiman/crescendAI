from __future__ import annotations

from pathlib import Path

import pytest

from teaching_knowledge.run_eval import _assert_models_compatible


def test_run_raises_when_teacher_and_judge_same_family() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        _assert_models_compatible("claude-sonnet-4-6", "claude-sonnet-4-6")


def test_run_raises_when_teacher_and_openrouter_judge_same_family() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        _assert_models_compatible("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


def test_run_allows_cross_family() -> None:
    _assert_models_compatible("claude-sonnet-4-6", "@cf/google/gemma-4-26b-a4b-it")
    _assert_models_compatible("claude-sonnet-4-6", "openai/gpt-5.4-mini")
