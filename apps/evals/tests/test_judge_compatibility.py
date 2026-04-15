from __future__ import annotations

import pytest

from shared.judge_compatibility import assert_judge_compatible, model_family


def test_anthropic_native_same_family_raises() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        assert_judge_compatible("claude-sonnet-4-6", "claude-sonnet-4-6")


def test_anthropic_native_vs_openrouter_anthropic_same_family_raises() -> None:
    with pytest.raises(ValueError, match="anthropic"):
        assert_judge_compatible("claude-sonnet-4-6", "anthropic/claude-sonnet-4-6")


def test_anthropic_teacher_vs_google_workers_ai_judge_passes() -> None:
    assert_judge_compatible("claude-sonnet-4-6", "@cf/google/gemma-4-26b-a4b-it")


def test_anthropic_teacher_vs_openrouter_openai_passes() -> None:
    assert_judge_compatible("claude-sonnet-4-6", "openai/gpt-5.4-mini")


def test_workers_ai_openai_prefix_resolves_to_openai_family() -> None:
    with pytest.raises(ValueError, match="openai"):
        assert_judge_compatible("@cf/openai/gpt-oss-120b", "openai/gpt-5.4-mini")


def test_qwen_teacher_vs_sonnet_judge_passes() -> None:
    assert_judge_compatible("qwen3-27b-finetune", "claude-sonnet-4-6")


def test_model_family_unknown_raises_to_surface_typos() -> None:
    with pytest.raises(ValueError, match="unknown model family"):
        model_family("some-random-nonsense-model")
