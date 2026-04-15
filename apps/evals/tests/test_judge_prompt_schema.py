from __future__ import annotations

from shared.judge import load_prompt


def test_judge_prompt_mentions_process_and_outcome_keys() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    assert '"process"' in prompt
    assert '"outcome"' in prompt


def test_judge_prompt_defines_process_vs_outcome() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    lowered = prompt.lower()
    # must explain what each signal means
    assert "process" in lowered and "notice" in lowered
    assert "outcome" in lowered and ("correct" in lowered or "accurate" in lowered)


def test_judge_prompt_retains_seven_dimension_numbered_list() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    for i in range(1, 8):
        assert f"{i}." in prompt


def test_judge_prompt_retains_zero_to_three_scale() -> None:
    prompt = load_prompt("synthesis_quality_judge_v2.txt")
    assert "0-3" in prompt or "0\u20113" in prompt or "0\u20133" in prompt
