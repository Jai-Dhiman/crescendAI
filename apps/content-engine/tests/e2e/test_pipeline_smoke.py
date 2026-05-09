"""End-to-end smoke: episode walks the full pipeline with mocked externals."""
from datetime import datetime, timezone
from pathlib import Path
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.pipeline.orchestrator import Orchestrator
from content_engine.store.episode_store import EpisodeStore
from content_engine.store.config_store import ConfigStore
from content_engine.adapters.model_runner import ModelOutput
from content_engine.adapters.scheduler import PostResult
from content_engine.agents.observation_selector import Observation
from content_engine.agents.narrator import ScriptText
from content_engine.agents.critic_truthfulness import Verdict


class _Runner:
    def run(self, clip):
        return ModelOutput(
            scores={d: [0.5] for d in ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]},
            duration_sec=15.0, raw={},
        )


class _Selector:
    def select(self, mo, meta):
        return Observation(dimension="phrasing", time_range=(5.0, 7.0), plain_english="rushed peak")


class _Narrator:
    def write_script(self, obs, cta, style_examples):
        return ScriptText(text="Hook. Obs. Close.", word_count=3)


class _Critic:
    def verify(self, clip, obs):
        return Verdict(passed=True, reason="audible")


class _Renderer:
    def __init__(self, dir): self._dir = dir
    def render(self, ep, cta):
        out = self._dir / f"{ep.id}.mp4"
        out.write_bytes(b"\x00" * 16)
        return out


class _Scheduler:
    def schedule(self, asset_path, when, platforms, caption, description_link):
        return [PostResult(platform=p, post_id=f"{p}_id", status="scheduled") for p in platforms]


def test_episode_reaches_scheduled_through_full_pipeline(tmp_path):
    es = EpisodeStore(db_path=tmp_path / "e.sqlite")
    cs = ConfigStore(db_path=tmp_path / "c.sqlite")
    cs.create_version("cta", {"phase": "A"})

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x")
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    es.save(Episode(
        id="ep_e2e",
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=State.CURATED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    ))

    orch = Orchestrator(
        episode_store=es,
        model_runner=_Runner(),
        clip_paths={"ep_e2e": clip},
        observation_selector=_Selector(),
        narrator=_Narrator(),
        critic=_Critic(),
        renderer=_Renderer(tmp_path),
        scheduler=_Scheduler(),
        config_store=cs,
    )

    # CURATED -> ANALYZED -> OBSERVATION_SELECTED -> SCRIPT_DRAFTED -> CRITIC_PASSED
    for _ in range(4):
        orch.tick()
    assert es.get("ep_e2e").state is State.CRITIC_PASSED

    # Simulate Jai recording voiceover, transitioning to RECORDED
    ep = es.get("ep_e2e")
    vo = tmp_path / "vo.wav"
    vo.write_bytes(b"x")
    ep.voiceover_path = str(vo)
    es.save(ep)
    es.transition("ep_e2e", State.RECORDED)

    # RECORDED -> RENDERED -> SCHEDULED
    orch.tick()
    orch.tick()

    final = es.get("ep_e2e")
    assert final.state is State.SCHEDULED
    assert final.posts is not None
    assert {"youtube", "tiktok", "instagram"} <= set(final.posts.keys())
