# tests/unit/test_ui_server.py
"""Verifies UI server swipe-approve transitions an episode."""
from datetime import datetime, timezone
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State
from content_engine.store.episode_store import EpisodeStore
from content_engine.ui.server import build_app


def _seed_candidate(store: EpisodeStore, eid: str) -> None:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    store.save(Episode(
        id=eid,
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=State.CANDIDATE,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    ))


def test_post_swipe_approve_transitions_candidate_to_curated(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_candidate(store, "ep1")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/swipe/ep1/approve")

    assert resp.status_code == 200
    assert store.get("ep1").state is State.CURATED


def test_post_swipe_reject_does_not_advance(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_candidate(store, "ep2")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/swipe/ep2/reject")

    assert resp.status_code == 200
    assert store.get("ep2").state is State.CANDIDATE
