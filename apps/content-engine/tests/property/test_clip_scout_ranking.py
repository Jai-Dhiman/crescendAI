"""Verifies ClipScout ranks by source_type weights."""
from content_engine.agents.clip_scout import ClipScout, SourceCriteria, Candidate


class FakeBackend:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results):
        return self._results[:max_results]


def test_higher_weighted_source_type_ranks_first():
    scout = ClipScout(
        youtube_backend=FakeBackend([
            Candidate(url="amateur", source_type="youtube_amateur", duration_sec=15, title="A"),
            Candidate(url="comp", source_type="youtube_competition", duration_sec=15, title="C"),
        ]),
        tiktok_backend=None,
    )
    crit = SourceCriteria(
        source_types=["youtube_amateur", "youtube_competition"],
        max_duration_sec=20,
        weights={"youtube_competition": 2.0, "youtube_amateur": 0.5},
    )
    results = scout.search(criteria=crit, count=10)
    assert [c.url for c in results] == ["comp", "amateur"]
