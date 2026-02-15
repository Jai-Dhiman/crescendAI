import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from masterclass_experiments.data import (
    Moment,
    Segment,
    extract_audio_segments,
    identify_segments,
    load_moments,
)


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


def test_identify_segments_creates_stop_segments():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        )
    ]

    segments = identify_segments(moments)

    stops = [s for s in segments if s.label == "stop"]
    assert len(stops) == 1
    assert stops[0].start_time == 80.0
    assert stops[0].end_time == 100.0
    assert stops[0].video_id == "vid1"


def test_identify_segments_creates_continue_between_moments():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid1",
            teacher="T",
            stop_timestamp=200.0,
            playing_before_start=180.0,
            playing_before_end=200.0,
            feedback_start=200.0,
            feedback_end=230.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 1
    assert continues[0].start_time == 130.0
    assert continues[0].end_time == 180.0


def test_identify_segments_skips_short_continue_gaps():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid1",
            teacher="T",
            stop_timestamp=133.0,
            playing_before_start=131.0,
            playing_before_end=133.0,
            feedback_start=133.0,
            feedback_end=140.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments, min_continue_duration=5.0)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 0


def test_identify_segments_no_continue_across_videos():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid2",
            teacher="T",
            stop_timestamp=200.0,
            playing_before_start=180.0,
            playing_before_end=200.0,
            feedback_start=200.0,
            feedback_end=230.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 0


def test_identify_segments_deduplicates_stop_windows():
    # Two moments with same playing window should produce only one STOP segment
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid1",
            teacher="T",
            stop_timestamp=120.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=120.0,
            feedback_end=150.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments)

    stops = [s for s in segments if s.label == "stop"]
    assert len(stops) == 1


def test_extract_audio_segments_creates_wav_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_dir = Path(tmpdir) / "audio"
        wav_dir.mkdir()
        out_dir = Path(tmpdir) / "segments"
        out_dir.mkdir()

        # Create a 10-second mono WAV at 16kHz
        sr = 16000
        audio = np.random.randn(sr * 10).astype(np.float32)
        sf.write(wav_dir / "vid1.wav", audio, sr)

        segments = [
            Segment(
                segment_id="stop_0000",
                video_id="vid1",
                label="stop",
                start_time=1.0,
                end_time=3.0,
                moment_id="a",
            ),
            Segment(
                segment_id="cont_0001",
                video_id="vid1",
                label="continue",
                start_time=5.0,
                end_time=8.0,
            ),
        ]

        extract_audio_segments(segments, wav_dir, out_dir)

        # Check files were created
        assert (out_dir / "stop_0000.wav").exists()
        assert (out_dir / "cont_0001.wav").exists()

        # Check durations
        data0, sr0 = sf.read(out_dir / "stop_0000.wav")
        assert len(data0) == sr * 2  # 2 seconds

        data1, sr1 = sf.read(out_dir / "cont_0001.wav")
        assert len(data1) == sr * 3  # 3 seconds
