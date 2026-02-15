import json
import tempfile
from pathlib import Path

from masterclass_experiments.data import Moment, load_moments


def _write_moments(path: Path, moments: list[dict]) -> None:
    with open(path, "w") as f:
        for m in moments:
            f.write(json.dumps(m) + "\n")


SAMPLE_MOMENT = {
    "moment_id": "abc123",
    "video_id": "7FTdGbVCPyQ",
    "video_title": "Test Masterclass",
    "teacher": "Test Teacher",
    "stop_timestamp": 619.6,
    "feedback_start": 619.6,
    "feedback_end": 649.6,
    "playing_before_start": 533.28,
    "playing_before_end": 549.90,
    "transcript_text": "Some feedback",
    "feedback_summary": "Summary",
    "musical_dimension": "tone_color",
    "secondary_dimensions": ["interpretation"],
    "severity": "moderate",
    "feedback_type": "suggestion",
    "piece": "Chopin Ballade No. 1",
    "composer": "Chopin",
    "passage_description": None,
    "student_level": None,
    "stop_order": 1,
    "total_stops": 16,
    "time_spent_seconds": 30.0,
    "demonstrated": False,
    "extracted_at": "2026-02-15T05:07:20.257864+00:00",
    "extraction_model": "gpt-4o",
    "confidence": 0.7,
}


def test_load_moments_parses_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "moments.jsonl"
        _write_moments(path, [SAMPLE_MOMENT])

        moments = load_moments(path)

        assert len(moments) == 1
        m = moments[0]
        assert m.moment_id == "abc123"
        assert m.video_id == "7FTdGbVCPyQ"
        assert m.playing_before_start == 533.28
        assert m.playing_before_end == 549.90
        assert m.feedback_end == 649.6
        assert m.musical_dimension == "tone_color"


def test_load_moments_sorts_by_video_and_timestamp():
    m1 = {**SAMPLE_MOMENT, "moment_id": "a", "video_id": "vid1", "stop_timestamp": 200.0}
    m2 = {**SAMPLE_MOMENT, "moment_id": "b", "video_id": "vid1", "stop_timestamp": 100.0}
    m3 = {**SAMPLE_MOMENT, "moment_id": "c", "video_id": "vid2", "stop_timestamp": 50.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "moments.jsonl"
        _write_moments(path, [m1, m2, m3])

        moments = load_moments(path)

        assert [m.moment_id for m in moments] == ["b", "a", "c"]
