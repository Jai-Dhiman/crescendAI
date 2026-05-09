"""Verifies invalid state transitions are rejected at the store boundary."""
from datetime import datetime, timezone
import pytest
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore, InvalidTransitionError


def _seed_episode(store: EpisodeStore, state: State) -> Episode:
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    ep = Episode(
        id="ep_test",
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=state,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    )
    store.save(ep)
    return ep


def test_invalid_transition_raises(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "t.sqlite")
    _seed_episode(store, State.CANDIDATE)
    with pytest.raises(InvalidTransitionError):
        store.transition("ep_test", State.PUBLISHED)


def test_valid_transition_persists_new_state(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "t.sqlite")
    _seed_episode(store, State.CANDIDATE)
    updated = store.transition("ep_test", State.CURATED)
    assert updated.state is State.CURATED
    assert store.get("ep_test").state is State.CURATED


def test_transition_on_missing_episode_raises_keyerror(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "t.sqlite")
    with pytest.raises(KeyError):
        store.transition("does_not_exist", State.CURATED)
