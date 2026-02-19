import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.competition import (
    segment_and_embed_competition,
    CompetitionRecord,
)


class TestSegmentAndEmbedCompetition:
    @pytest.fixture
    def setup_cache(self, tmp_path):
        """Create a mock competition cache with audio files and recording metadata."""
        cache_dir = tmp_path / "chopin2021"
        audio_dir = cache_dir / "audio"
        audio_dir.mkdir(parents=True)

        # Create synthetic WAV files (90s each = 3 segments of 30s)
        import soundfile as sf
        sr = 24000
        for i in range(2):
            audio = np.random.randn(sr * 90).astype(np.float32)
            sf.write(str(audio_dir / f"recording_{i}.wav"), audio, sr)

        # Create recordings metadata
        recordings = [
            {
                "recording_id": "recording_0",
                "competition": "chopin",
                "edition": 2021,
                "round": "stage2",
                "placement": 1,
                "performer": "Test Performer A",
                "piece": "Ballade No. 1",
                "audio_path": "audio/recording_0.wav",
                "duration_seconds": 90.0,
                "source_url": "https://youtube.com/watch?v=test0",
                "country": "Test",
            },
            {
                "recording_id": "recording_1",
                "competition": "chopin",
                "edition": 2021,
                "round": "stage2",
                "placement": 2,
                "performer": "Test Performer B",
                "piece": "Ballade No. 1",
                "audio_path": "audio/recording_1.wav",
                "duration_seconds": 90.0,
                "source_url": "https://youtube.com/watch?v=test1",
                "country": "Test",
            },
        ]
        import jsonlines
        with jsonlines.open(cache_dir / "recordings.jsonl", mode="w") as writer:
            for r in recordings:
                writer.write(r)

        return cache_dir

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_creates_segment_metadata(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        # Mock MuQ to return fake embeddings
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        n = segment_and_embed_competition(cache_dir, segment_duration=30.0)

        metadata_path = cache_dir / "metadata.jsonl"
        assert metadata_path.exists()

        import jsonlines
        with jsonlines.open(metadata_path) as reader:
            segments = list(reader)

        # 2 recordings * 3 segments each = 6 segments
        assert len(segments) == 6
        assert n == 6

        # Check segment metadata schema
        seg = segments[0]
        assert "segment_id" in seg
        assert "recording_id" in seg
        assert "performer" in seg
        assert "piece" in seg
        assert "round" in seg
        assert "placement" in seg
        assert "segment_start" in seg
        assert "segment_end" in seg

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_creates_per_segment_embeddings(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        segment_and_embed_competition(cache_dir, segment_duration=30.0)

        emb_dir = cache_dir / "muq_embeddings"
        assert emb_dir.exists()
        pt_files = list(emb_dir.glob("*.pt"))
        assert len(pt_files) == 6

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_segments_inherit_recording_placement(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        segment_and_embed_competition(cache_dir)

        import jsonlines
        with jsonlines.open(cache_dir / "metadata.jsonl") as reader:
            segments = list(reader)

        # All segments from recording_0 should have placement=1
        r0_segs = [s for s in segments if s["recording_id"] == "recording_0"]
        assert all(s["placement"] == 1 for s in r0_segs)

        # All segments from recording_1 should have placement=2
        r1_segs = [s for s in segments if s["recording_id"] == "recording_1"]
        assert all(s["placement"] == 2 for s in r1_segs)

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_is_idempotent(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        n1 = segment_and_embed_competition(cache_dir)
        n2 = segment_and_embed_competition(cache_dir)

        assert n1 == 6
        assert n2 == 0  # All already cached
