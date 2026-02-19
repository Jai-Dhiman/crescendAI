import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.maestro import (
    parse_maestro_audio_metadata,
    MaestroSegment,
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
