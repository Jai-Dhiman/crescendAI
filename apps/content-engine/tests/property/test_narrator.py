"""Verifies Narrator produces script within length budget and CTA-phase-aware."""
from content_engine.adapters.llm_gateway import LlmResponse, LlmMode
from content_engine.agents.narrator import Narrator
from content_engine.agents.observation_selector import Observation
from content_engine.render.templates import CtaTemplate


class FakeLlm:
    def __init__(self, text: str):
        self._text = text
        self.calls = []

    def complete(self, prompt, mode, schema=None):
        self.calls.append((prompt, mode, schema))
        return LlmResponse(text=self._text)


def test_narrator_truncates_to_word_budget():
    long_text = " ".join(["word"] * 200)
    fake = FakeLlm(long_text)
    narrator = Narrator(llm=fake)
    obs = Observation(
        dimension="phrasing",
        time_range=(5.2, 7.1),
        plain_english="Phrasing peak arrives early.",
    )
    script = narrator.write_script(obs, CtaTemplate.for_phase("A"), style_examples=[])
    assert script.word_count <= 120
    assert script.text.split() == script.text.split()[:script.word_count]


def test_narrator_passes_phase_c_cta_into_prompt():
    fake = FakeLlm("Hook. Observation. crescend.ai/submit.")
    narrator = Narrator(llm=fake)
    obs = Observation(
        dimension="timing",
        time_range=(1.0, 3.0),
        plain_english="Tempo dips.",
    )
    narrator.write_script(obs, CtaTemplate.for_phase("C"), style_examples=[])

    prompt, mode, _ = fake.calls[0]
    assert mode == LlmMode.NARRATOR
    assert "crescend.ai/submit" in prompt


def test_narrator_phase_a_prompt_has_no_spoken_cta():
    fake = FakeLlm("Hook. Observation. End.")
    narrator = Narrator(llm=fake)
    obs = Observation(
        dimension="dynamics",
        time_range=(2.0, 4.0),
        plain_english="Dynamic peak misplaced.",
    )
    narrator.write_script(obs, CtaTemplate.for_phase("A"), style_examples=[])

    prompt, _, _ = fake.calls[0]
    assert "no spoken cta" in prompt.lower() or "do not include" in prompt.lower()
