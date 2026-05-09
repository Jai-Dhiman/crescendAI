# apps/content-engine/tests/unit/test_orchestrator_progression.py
"""Verifies Orchestrator advances episode through full pipeline to SCHEDULED."""
from datetime import datetime, timezone
from pathlib import Path
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.pipeline.orchestrator import Orchestrator
from content_engine.store.episode_store import EpisodeStore
from content_engine.adapters.model_runner import ModelOutput
from content_engine.adapters.scheduler import PostResult
from content_engine.agents.observation_selector import Observation
from content_engine.agents.narrator import ScriptText
from content_engine.agents.critic_truthfulness import Verdict


class FakeRunner:
    def run(self, clip):
        return ModelOutput(
            scores={d: [0.5] for d in ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]},
            duration_sec=15.0, raw={},
        )


class FakeSelector:
    def select(self, model_output, metadata):
        return Observation(dimension="phrasing", time_range=(5.0, 7.0), plain_english="rushed")


class FakeNarrator:
    def write_script(self, observation, cta_template, style_examples):
        return ScriptText(text="Hook. Obs. End.", word_count=3)


class FakeCritic:
    def verify(self, clip_path, observation):
        return Verdict(passed=True, reason="audible")


class FakeRenderer:
    def render(self, episode, cta_template):
        return Path("/tmp/fake_render.mp4")


class FakeScheduler:
    def schedule(self, asset_path, when, platforms, caption, description_link):
        return [PostResult(platform=p, post_id=f"{p}_id", status="scheduled") for p in platforms]


def _seed(store: EpisodeStore, state: State, eid: str = "ep1", **fields) -> Episode:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    ep = Episode(
        id=eid,
        candidate_url="x",
        source_type="youtube_amateur",
        state=state,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
        **fields,
    )
    store.save(ep)
    return ep


def test_episode_advances_through_each_pipeline_stage(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x")
    _seed(store, State.CURATED, eid="ep1")

    orch = Orchestrator(
        episode_store=store,
        model_runner=FakeRunner(),
        clip_paths={"ep1": clip},
        observation_selector=FakeSelector(),
        narrator=FakeNarrator(),
        critic=FakeCritic(),
        renderer=FakeRenderer(),
        scheduler=FakeScheduler(),
    )

    # tick #1: CURATED -> ANALYZED
    orch.tick()
    assert store.get("ep1").state is State.ANALYZED

    # tick #2: ANALYZED -> OBSERVATION_SELECTED
    orch.tick()
    assert store.get("ep1").state is State.OBSERVATION_SELECTED

    # tick #3: OBSERVATION_SELECTED -> SCRIPT_DRAFTED
    orch.tick()
    assert store.get("ep1").state is State.SCRIPT_DRAFTED

    # tick #4: SCRIPT_DRAFTED -> CRITIC_PASSED
    orch.tick()
    assert store.get("ep1").state is State.CRITIC_PASSED

    # human step: Jai records voiceover (simulated)
    ep = store.get("ep1")
    ep.voiceover_path = str(tmp_path / "vo.wav")
    Path(ep.voiceover_path).write_bytes(b"x")
    store.save(ep)
    store.transition("ep1", State.RECORDED)

    # tick #5: RECORDED -> RENDERED
    orch.tick()
    assert store.get("ep1").state is State.RENDERED

    # tick #6: RENDERED -> SCHEDULED
    orch.tick()
    assert store.get("ep1").state is State.SCHEDULED
    assert store.get("ep1").posts is not None
    assert "youtube" in store.get("ep1").posts


class FakeCriticKill:
    def verify(self, clip_path, observation):
        return Verdict(passed=False, reason="not audible in cited range")


def test_critic_kill_routes_to_killed_truthfulness(tmp_path):
    """Brand-safety failure path: a KILL verdict must land in KILLED_TRUTHFULNESS,
    NOT CRITIC_PASSED. This is the engine's highest-stakes invariant per spec."""
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x")
    _seed(store, State.CURATED, eid="ep_kill")

    orch = Orchestrator(
        episode_store=store,
        model_runner=FakeRunner(),
        clip_paths={"ep_kill": clip},
        observation_selector=FakeSelector(),
        narrator=FakeNarrator(),
        critic=FakeCriticKill(),
        renderer=FakeRenderer(),
        scheduler=FakeScheduler(),
    )

    # CURATED -> ANALYZED -> OBSERVATION_SELECTED -> SCRIPT_DRAFTED -> KILLED_TRUTHFULNESS
    for _ in range(4):
        orch.tick()

    final = store.get("ep_kill")
    assert final.state is State.KILLED_TRUTHFULNESS
    assert final.posts is None
    # Subsequent ticks must NOT advance a killed episode
    orch.tick()
    assert store.get("ep_kill").state is State.KILLED_TRUTHFULNESS


class FakeRunnerRaises:
    def run(self, clip):
        raise RuntimeError("unexpected upstream bug")


def test_unexpected_exception_propagates_not_swallowed(tmp_path):
    """Per spec error policy: unknown exceptions must propagate, not silently
    transition to FAILED_*. Sentry needs to see real bugs."""
    import pytest
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x")
    _seed(store, State.CURATED, eid="ep_bug")

    orch = Orchestrator(
        episode_store=store,
        model_runner=FakeRunnerRaises(),
        clip_paths={"ep_bug": clip},
        observation_selector=FakeSelector(),
        narrator=FakeNarrator(),
        critic=FakeCritic(),
        renderer=FakeRenderer(),
        scheduler=FakeScheduler(),
    )

    with pytest.raises(RuntimeError, match="unexpected upstream bug"):
        orch.tick()
    # Episode stays in CURATED -- no silent failure marker
    assert store.get("ep_bug").state is State.CURATED
