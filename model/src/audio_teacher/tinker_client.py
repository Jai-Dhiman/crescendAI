"""Real Tinker sampling client for the Gate 0 Inkling probe.

Wraps the Inkling audio flow
(https://tinker-docs.thinkingmachines.ai/cookbook/inkling/audio): local
WAV files are referenced with tml_renderers.chat.AudioPointer and encoded
client-side into DMel tokens; remote URLs are unsupported, which is why
the manifest loader guarantees clips exist locally before any call.

Never imported by tests except the not-installed error path; all scoring
tests use audio_teacher.client.RecordedResponseClient. The SDK trio is
deliberately NOT in pyproject dependencies -- verify current package
names on tinker-docs.thinkingmachines.ai when funding the probe run.
"""
from __future__ import annotations

from audio_teacher.audio import validate_wav
from audio_teacher.client import ProbeResponse
from audio_teacher.manifest import ContrastPair
from audio_teacher.prompts import build_question

INKLING_MODEL = "thinkingmachines/Inkling"

# Conservative DMel-tokens-per-second estimate for PRE-CALL cost projection.
# Overestimating trips the budget cap earlier, never later -- the safe
# direction. Actual recorded cost uses real token counts from the response.
AUDIO_TOKENS_PER_SECOND_ESTIMATE = 100

# Text scaffold (question + chat template) allowance for pre-call estimates.
TEXT_TOKENS_ESTIMATE = 500


class TinkerNotInstalledError(RuntimeError):
    """The Tinker SDK trio is not installed in this environment."""


class TinkerProbeClient:
    """ProbeClient implementation that samples thinkingmachines/Inkling."""

    def __init__(
        self,
        sample_rate: int,
        usd_per_1m_input_tokens: float,
        usd_per_1m_output_tokens: float,
        max_tokens: int = 256,
    ):
        try:
            import tinker
            from tinker_cookbook import model_info
            from tinker_cookbook.renderers import get_renderer, get_text_content
            from tinker_cookbook.tokenizer_utils import get_tokenizer
            from tml_renderers import chat
        except ImportError as exc:
            raise TinkerNotInstalledError(
                "Tinker SDK not installed; the live probe needs it. Install with:\n"
                "    cd model && uv add tinker tinker-cookbook\n"
                "(verify current package names on tinker-docs.thinkingmachines.ai; "
                "tests never need this -- offline scoring uses RecordedResponseClient)"
            ) from exc
        self._tinker = tinker
        self._chat = chat
        self._get_text_content = get_text_content
        self._sample_rate = sample_rate
        self._in_rate = usd_per_1m_input_tokens
        self._out_rate = usd_per_1m_output_tokens
        self._max_tokens = max_tokens
        self._renderer = get_renderer(
            model_info.get_recommended_renderer_name(INKLING_MODEL),
            get_tokenizer(INKLING_MODEL),
        )
        self._sampling = tinker.ServiceClient().create_sampling_client(
            base_model=INKLING_MODEL
        )

    def _messages(self, pair: ContrastPair):
        chat = self._chat
        user = chat.Author(chat.AuthorKind.User)

        def clip_message(path):
            info = validate_wav(path, expected_sample_rate=self._sample_rate)
            return chat.Message(
                content=chat.AudioPointer(
                    location=str(path),
                    format=chat.AudioFormat.Wav,
                    num_frames=info.num_frames,
                    sample_rate=info.sample_rate,
                ),
                author=user,
            )

        # The Inkling renderer's build_generation_prompt consumes a
        # chat.MessageList (not a bare Python list) -- match the cookbook
        # sample_audio.py flow exactly.
        return chat.MessageList([
            chat.Message(content=chat.Text(build_question(pair.axis)), author=user),
            chat.Message(content=chat.Text("Clip A:"), author=user),
            clip_message(pair.clip_a),
            chat.Message(content=chat.Text("Clip B:"), author=user),
            clip_message(pair.clip_b),
        ])

    def estimate_cost_usd(self, pair: ContrastPair) -> float:
        seconds = 0.0
        for path in (pair.clip_a, pair.clip_b):
            info = validate_wav(path, expected_sample_rate=self._sample_rate)
            seconds += info.duration_seconds
        est_input = seconds * AUDIO_TOKENS_PER_SECOND_ESTIMATE + TEXT_TOKENS_ESTIMATE
        return (est_input / 1e6) * self._in_rate + (
            self._max_tokens / 1e6
        ) * self._out_rate

    def ask(self, pair: ContrastPair) -> ProbeResponse:
        prompt = self._renderer.build_generation_prompt(self._messages(pair))
        result = self._sampling.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=self._max_tokens,
                # Greedy: a discrimination probe wants the model's single
                # most-likely A/B judgment, not a sample from its posterior --
                # temperature noise would add variance to a 20-pair accuracy.
                temperature=0.0,
                stop=self._renderer.get_stop_sequences(),
            ),
        ).result()
        tokens = result.sequences[0].tokens
        message, _termination = self._renderer.parse_response(tokens)
        # parse_response returns a renderers.Message (a TypedDict whose
        # "content" is a str or a list of content parts), NOT an object with
        # .content.text -- get_text_content is the cookbook accessor and it
        # strips reasoning parts, leaving just the answer text to parse.
        text = self._get_text_content(message)
        # Loud on API drift: if ModelInput stops exposing length, this raises.
        input_tokens = prompt.length
        output_tokens = len(tokens)
        cost = (input_tokens / 1e6) * self._in_rate + (
            output_tokens / 1e6
        ) * self._out_rate
        return ProbeResponse(
            pair_id=pair.pair_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
