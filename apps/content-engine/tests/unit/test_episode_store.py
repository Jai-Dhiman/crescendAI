"""Verifies EpisodeStore round-trips episodes through SQLite."""
from datetime import datetime, timezone
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore


def _make_episode(eid: str = "ep_001") -> Episode:
    now = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    return Episode(
        id=eid,
        candidate_url="https://youtube.com/watch?v=abc",
        source_type="youtube_amateur",
        state=State.CANDIDATE,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    )


def test_save_then_get_returns_equal_episode(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "test.sqlite")
    ep = _make_episode()
    store.save(ep)
    retrieved = store.get(ep.id)
    assert retrieved == ep
