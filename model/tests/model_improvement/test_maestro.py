import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

import jsonlines

from model_improvement.maestro import (
    build_piece_performer_mapping,
    parse_maestro_audio_metadata,
    MaestroSegment,
    segment_and_embed_maestro,
)


class TestParseMaestroAudioMetadata:
    @pytest.fixture
    def maestro_dir(self, tmp_path):
        """Create a mock MAESTRO directory with metadata JSON."""
        maestro = tmp_path / "maestro-v3.0.0"
        maestro.mkdir()

        metadata = {
            "canonical_composer": {"0": "Chopin", "1": "Chopin", "2": "Beethoven"},
            "canonical_title": {
                "0": "Ballade No. 1",
                "1": "Ballade No. 1",
                "2": "Sonata No. 14",
            },
            "split": {"0": "train", "1": "train", "2": "validation"},
            "midi_filename": {
                "0": "2015/file_a.midi",
                "1": "2015/file_b.midi",
                "2": "2017/file_c.midi",
            },
            "audio_filename": {
                "0": "2015/file_a.wav",
                "1": "2015/file_b.wav",
                "2": "2017/file_c.wav",
            },
            "duration": {"0": 300.5, "1": 280.2, "2": 600.1},
        }

        with open(maestro / "maestro-v3.0.0.json", "w") as f:
            json.dump(metadata, f)

        return maestro

    def test_parses_all_records(self, maestro_dir):
        records = parse_maestro_audio_metadata(maestro_dir)
        assert len(records) == 3

    def test_record_schema(self, maestro_dir):
        records = parse_maestro_audio_metadata(maestro_dir)
        r = records[0]
        assert "audio_filename" in r
        assert "canonical_title" in r
        assert "canonical_composer" in r
        assert "split" in r
        assert "duration" in r

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            parse_maestro_audio_metadata(Path("/nonexistent"))

    def test_raises_on_missing_json(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            parse_maestro_audio_metadata(empty_dir)


class TestSegmentAndEmbedMaestro:
    @pytest.fixture
    def maestro_with_audio(self, tmp_path):
        """Create mock MAESTRO dir with metadata and audio files."""
        maestro = tmp_path / "maestro-v3.0.0"
        subdir = maestro / "2015"
        subdir.mkdir(parents=True)

        import soundfile as sf
        sr = 24000
        # Two recordings of the same piece (different performers)
        for fname in ["file_a.wav", "file_b.wav"]:
            audio = np.random.randn(sr * 65).astype(np.float32)  # 65s = 2 full + 1 short
            sf.write(str(subdir / fname), audio, sr)
        # One recording of a different piece
        audio = np.random.randn(sr * 35).astype(np.float32)
        sf.write(str(subdir / "file_c.wav"), audio, sr)

        metadata = {
            "canonical_composer": {"0": "Chopin", "1": "Chopin", "2": "Beethoven"},
            "canonical_title": {
                "0": "Ballade No. 1",
                "1": "Ballade No. 1",
                "2": "Sonata No. 14",
            },
            "split": {"0": "train", "1": "train", "2": "validation"},
            "midi_filename": {
                "0": "2015/file_a.midi",
                "1": "2015/file_b.midi",
                "2": "2015/file_c.midi",
            },
            "audio_filename": {
                "0": "2015/file_a.wav",
                "1": "2015/file_b.wav",
                "2": "2015/file_c.wav",
            },
            "duration": {"0": 65.0, "1": 65.0, "2": 35.0},
        }
        with open(maestro / "maestro-v3.0.0.json", "w") as f:
            json.dump(metadata, f)

        return maestro, tmp_path / "cache"

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_produces_segment_metadata(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n = segment_and_embed_maestro(maestro_dir, cache_dir, segment_duration=30.0)

        metadata_path = cache_dir / "metadata.jsonl"
        assert metadata_path.exists()

        with jsonlines.open(metadata_path) as reader:
            segments = list(reader)

        # file_a: 65s -> 2 full (30s) + 1 short (5s, meets min) = 3 segments
        # file_b: 65s -> 3 segments
        # file_c: 35s -> 1 full + 1 short (5s) = 2 segments
        assert len(segments) == 8
        assert n == 8

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_is_idempotent(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n1 = segment_and_embed_maestro(maestro_dir, cache_dir)
        n2 = segment_and_embed_maestro(maestro_dir, cache_dir)
        assert n1 > 0
        assert n2 == 0

    @patch("audio_experiments.extractors.muq.MuQExtractor")
    def test_contrastive_mapping(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        segment_and_embed_maestro(maestro_dir, cache_dir)
        mapping = build_piece_performer_mapping(cache_dir)

        # "Ballade No. 1" has 2 recordings -> included
        assert "Ballade No. 1" in mapping
        assert len(mapping["Ballade No. 1"]) == 6  # 3 segments from each recording

        # "Sonata No. 14" has only 1 recording -> excluded
        assert "Sonata No. 14" not in mapping
