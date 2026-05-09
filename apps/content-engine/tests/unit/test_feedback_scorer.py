"""Verifies FeedbackScorer adjusts source_type weights based on install conversion."""
from datetime import datetime, timezone, timedelta
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore
from content_engine.store.config_store import ConfigStore
from content_engine.feedback.scorer import FeedbackScorer


def _seed_episode(store: EpisodeStore, eid: str, source_type: str, views: int, installs: int, age_days: int):
    now = datetime(2026, 5, 8, tzinfo=timezone.utc) - timedelta(days=age_days)
    ep = Episode(
        id=eid,
        candidate_url=f"https://yt.example/{eid}",
        source_type=source_type,
        state=State.MEASURED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
        analytics={"views": views, "installs": installs},
    )
    store.save(ep)


def test_higher_converting_source_type_gets_higher_weight(tmp_path):
    es = EpisodeStore(db_path=tmp_path / "ep.sqlite")
    cs = ConfigStore(db_path=tmp_path / "cfg.sqlite")
    cs.create_version("ranking_weights", {"youtube_amateur": 1.0, "youtube_competition": 1.0})

    _seed_episode(es, "a1", "youtube_amateur", views=1000, installs=20, age_days=3)
    _seed_episode(es, "a2", "youtube_amateur", views=1000, installs=15, age_days=4)
    _seed_episode(es, "c1", "youtube_competition", views=1000, installs=2, age_days=3)
    _seed_episode(es, "c2", "youtube_competition", views=1000, installs=3, age_days=4)

    scorer = FeedbackScorer(episode_store=es, config_store=cs)
    new_version = scorer.update_weights(since=datetime(2026, 5, 1, tzinfo=timezone.utc))

    new_weights = cs.get("ranking_weights", version=new_version).value
    assert new_weights["youtube_amateur"] > new_weights["youtube_competition"]


def test_no_episodes_in_window_keeps_weights_unchanged(tmp_path):
    es = EpisodeStore(db_path=tmp_path / "ep.sqlite")
    cs = ConfigStore(db_path=tmp_path / "cfg.sqlite")
    initial = cs.create_version("ranking_weights", {"youtube_amateur": 1.0})

    scorer = FeedbackScorer(episode_store=es, config_store=cs)
    new_version = scorer.update_weights(since=datetime(2026, 5, 1, tzinfo=timezone.utc))

    assert new_version == initial
