"""Verifies ClipScout filters candidates by source_type criteria."""
from content_engine.agents.clip_scout import ClipScout, SourceCriteria, Candidate


class FakeYouTubeBackend:
    """Test double standing in for the YouTube data fetch -- provides fixed candidates."""

    def search(self, query: str, max_results: int) -> list[Candidate]:
        return [
            Candidate(url="https://yt.example/a", source_type="youtube_amateur", duration_sec=18, title="A"),
            Candidate(url="https://yt.example/b", source_type="youtube_label", duration_sec=15, title="B"),
            Candidate(url="https://yt.example/c", source_type="youtube_amateur", duration_sec=22, title="C"),
            Candidate(url="https://yt.example/d", source_type="youtube_competition", duration_sec=19, title="D"),
        ]


def test_search_filters_by_source_type():
    scout = ClipScout(youtube_backend=FakeYouTubeBackend(), tiktok_backend=None)
    crit = SourceCriteria(
        source_types=["youtube_amateur"],
        max_duration_sec=20,
        weights={},
    )
    results = scout.search(criteria=crit, count=10)
    urls = {c.url for c in results}
    assert urls == {"https://yt.example/a"}  # only amateur AND duration <= 20


def test_search_filters_by_max_duration():
    scout = ClipScout(youtube_backend=FakeYouTubeBackend(), tiktok_backend=None)
    crit = SourceCriteria(
        source_types=["youtube_amateur", "youtube_competition"],
        max_duration_sec=20,
        weights={},
    )
    results = scout.search(criteria=crit, count=10)
    urls = {c.url for c in results}
    assert urls == {"https://yt.example/a", "https://yt.example/d"}
