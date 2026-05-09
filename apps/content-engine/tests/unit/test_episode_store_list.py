"""Verifies list_by_state filters and orders correctly."""
from datetime import datetime, timezone, timedelta
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore


def _ep(eid: str, state: State, t_offset_min: int) -> Episode:
    base = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    t = base + timedelta(minutes=t_offset_min)
    return Episode(
        id=eid,
        candidate_url=f"https://yt.example/{eid}",
        source_type="youtube_amateur",
        state=state,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=t,
        updated_at=t,
    )


def test_list_by_state_returns_only_matching_episodes_in_order(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "t.sqlite")
    store.save(_ep("a", State.CANDIDATE, 0))
    store.save(_ep("b", State.CURATED, 5))
    store.save(_ep("c", State.CANDIDATE, 10))
    store.save(_ep("d", State.ANALYZED, 15))

    candidates = store.list_by_state(State.CANDIDATE)
    assert [e.id for e in candidates] == ["a", "c"]

    curated = store.list_by_state(State.CURATED)
    assert [e.id for e in curated] == ["b"]


def test_list_by_state_returns_empty_when_none_match(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "t.sqlite")
    store.save(_ep("a", State.CANDIDATE, 0))
    assert store.list_by_state(State.PUBLISHED) == []
