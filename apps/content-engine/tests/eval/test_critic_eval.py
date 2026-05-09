"""Eval gate: critic_truthfulness false-negative rate on golden set.

False-negative = letting a false observation through as PASS. This is the
brand-safety failure mode and is treated as a deploy gate at FN < 5%.
"""
import json
import os
from pathlib import Path
import pytest
from content_engine.adapters.llm_gateway import LlmGateway
from content_engine.agents.critic_truthfulness import CriticTruthfulness
from content_engine.agents.observation_selector import Observation


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EVAL") != "1",
    reason="eval-set tests run only when RUN_EVAL=1 (use real LLM + real clips)",
)


def test_critic_false_negative_rate_below_5_percent():
    base = Path(__file__).parent
    cases = json.loads((base / "golden_critic.json").read_text())
    gw = LlmGateway(
        cf_gateway_url=os.environ["CF_AI_GATEWAY_URL"],
        cf_token=os.environ["CF_API_TOKEN"],
        claude_bin=os.environ.get("CLAUDE_CODE_BIN", "/usr/local/bin/claude"),
    )
    critic = CriticTruthfulness(llm=gw)

    false_observations = [c for c in cases if not c["expected_passed"]]
    false_negatives = 0
    for case in false_observations:
        clip = base / case["clip"]
        if not clip.exists():
            pytest.skip(f"clip not present: {clip}")
        obs = Observation(
            dimension=case["observation"]["dimension"],
            time_range=tuple(case["observation"]["time_range"]),
            plain_english=case["observation"]["plain_english"],
        )
        verdict = critic.verify(clip, obs)
        if verdict.passed:
            false_negatives += 1

    fn_rate = false_negatives / max(1, len(false_observations))
    assert fn_rate < 0.05, f"critic FN rate {fn_rate:.2%} above 5% deploy gate"
