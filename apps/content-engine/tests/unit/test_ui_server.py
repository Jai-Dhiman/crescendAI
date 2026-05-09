# tests/unit/test_ui_server.py
"""Verifies UI server swipe-approve and record-complete transitions."""
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


def test_post_swipe_reject_returns_404_for_unknown_episode(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/swipe/nonexistent/reject")

    assert resp.status_code == 404


def _seed_critic_passed(store: EpisodeStore, eid: str) -> None:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    store.save(Episode(
        id=eid,
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=State.CRITIC_PASSED,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    ))


def test_record_complete_transitions_critic_passed_to_recorded(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_critic_passed(store, "ep3")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post(
        "/record/ep3/complete",
        json={"voiceover_path": "/tmp/vo_ep3.wav"},
        content_type="application/json",
    )

    assert resp.status_code == 200
    updated = store.get("ep3")
    assert updated.state is State.RECORDED
    assert updated.voiceover_path == "/tmp/vo_ep3.wav"


def test_record_complete_missing_voiceover_path_returns_400(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_critic_passed(store, "ep4")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/record/ep4/complete", json={}, content_type="application/json")

    assert resp.status_code == 400
    assert store.get("ep4").state is State.CRITIC_PASSED


def _seed_killed(store: EpisodeStore, eid: str) -> None:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    store.save(Episode(
        id=eid,
        candidate_url="https://yt.example/x",
        source_type="youtube_amateur",
        state=State.KILLED_TRUTHFULNESS,
        config_versions={"cta": 1, "source_criteria": 1, "ranking_weights": 1},
        created_at=now,
        updated_at=now,
    ))


def test_override_critic_transitions_killed_to_critic_passed(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_killed(store, "ep5")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/swipe/ep5/override-critic")

    assert resp.status_code == 200
    assert store.get("ep5").state is State.CRITIC_PASSED


def test_override_critic_on_wrong_state_returns_409(tmp_path):
    store = EpisodeStore(db_path=tmp_path / "e.sqlite")
    _seed_candidate(store, "ep6")
    app = build_app(episode_store=store)
    client = app.test_client()

    resp = client.post("/swipe/ep6/override-critic")

    assert resp.status_code == 409
    assert store.get("ep6").state is State.CANDIDATE
