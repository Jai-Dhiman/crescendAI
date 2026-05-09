# apps/content-engine/tests/unit/test_orchestrator_dispatch.py
"""Verifies Orchestrator dispatches CURATED -> model_runner -> ANALYZED."""
from datetime import datetime, timezone
from pathlib import Path
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.pipeline.orchestrator import Orchestrator
from content_engine.store.episode_store import EpisodeStore
from content_engine.adapters.model_runner import ModelOutput


class FakeModelRunner:
    def __init__(self):
        self.calls = []

    def run(self, clip_path):
        self.calls.append(clip_path)
        return ModelOutput(
            scores={d: [0.5] for d in ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]},
            duration_sec=15.0,
            raw={},
        )


def _seed_curated(store: EpisodeStore, eid: str = "ep1") -> Episode:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    ep = Episode(
        id=eid,
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=State.CURATED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    )
    store.save(ep)
    return ep


def test_tick_dispatches_curated_episode_to_model_runner_and_advances_to_analyzed(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_curated(store, "ep1")
    runner = FakeModelRunner()

    clip_paths = {"ep1": tmp_path / "clip.wav"}
    (tmp_path / "clip.wav").write_bytes(b"x")

    orch = Orchestrator(
        episode_store=store,
        model_runner=runner,
        clip_paths=clip_paths,
        observation_selector=None,
        narrator=None,
        critic=None,
        renderer=None,
        scheduler=None,
    )
    orch.tick()

    updated = store.get("ep1")
    assert updated.state is State.ANALYZED
    assert updated.model_output is not None
    assert updated.model_output["duration_sec"] == 15.0
    assert len(runner.calls) == 1


def test_tick_does_nothing_when_no_curated_episodes(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    runner = FakeModelRunner()

    orch = Orchestrator(
        episode_store=store,
        model_runner=runner,
        clip_paths={},
        observation_selector=None,
        narrator=None,
        critic=None,
        renderer=None,
        scheduler=None,
    )
    orch.tick()

    assert len(runner.calls) == 0
