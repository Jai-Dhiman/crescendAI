"""Verifies Episode dataclass JSON round-trip preserves all fields."""
from datetime import datetime, timezone
from content_engine.pipeline.episode import Episode
from content_engine.pipeline.states import State


def test_episode_round_trip_preserves_all_fields():
    original = Episode(
        id="ep_001",
        candidate_url="https://youtube.com/watch?v=abc",
        source_type="youtube_amateur",
        model_output={"phrasing": [0.4, 0.5, 0.6]},
        observation={"dimension": "phrasing", "time_range": [5.2, 7.1], "plain_english": "rushed"},
        script_text="Hook... observation... close.",
        voiceover_path="data/voiceovers/ep_001.wav",
        render_path="data/renders/ep_001.mp4",
        posts={"youtube": "yt_xyz", "tiktok": "tt_abc"},
        analytics={"views": 1234, "installs": 7},
        state=State.MEASURED,
        config_versions={"cta": 1, "source_criteria": 2, "ranking_weights": 3},
        created_at=datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 8, 14, 30, tzinfo=timezone.utc),
    )
    restored = Episode.from_dict(original.to_dict())
    assert restored == original
