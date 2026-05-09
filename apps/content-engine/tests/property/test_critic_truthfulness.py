"""Verifies CriticTruthfulness parses CLI verdicts correctly."""
from pathlib import Path
from content_engine.adapters.llm_gateway import LlmResponse, LlmMode
from content_engine.agents.critic_truthfulness import CriticTruthfulness, Verdict
from content_engine.agents.observation_selector import Observation


class FakeLlm:
    def __init__(self, text: str):
        self._text = text
        self.calls = []

    def complete(self, prompt, mode, schema=None):
        self.calls.append((prompt, mode))
        return LlmResponse(text=self._text)


def _obs() -> Observation:
    return Observation(dimension="phrasing", time_range=(5.2, 7.1), plain_english="Phrasing peak arrives early.")


def test_pass_verdict_when_response_says_pass(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"x")
    fake = FakeLlm("VERDICT: PASS\nReason: The phrasing peak is audibly early at 5.6s.")
    critic = CriticTruthfulness(llm=fake)
    v = critic.verify(clip, _obs())
    assert v.passed is True
    assert "audibly" in v.reason


def test_kill_verdict_when_response_says_kill(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"x")
    fake = FakeLlm("VERDICT: KILL\nReason: No audible deviation in the cited range.")
    critic = CriticTruthfulness(llm=fake)
    v = critic.verify(clip, _obs())
    assert v.passed is False
    assert "No audible" in v.reason


def test_critic_invokes_llm_in_critic_mode(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"x")
    fake = FakeLlm("VERDICT: PASS\nReason: ok")
    critic = CriticTruthfulness(llm=fake)
    critic.verify(clip, _obs())
    _, mode = fake.calls[0]
    assert mode == LlmMode.CRITIC
