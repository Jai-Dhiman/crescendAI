import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.youtube_piano import (
    load_channel_list,
    discover_channel_videos,
    download_piano_audio,
)


class TestLoadChannelList:
    def test_loads_yaml(self, tmp_path):
        channels_yaml = tmp_path / "channels.yaml"
        channels_yaml.write_text(
            "channels:\n"
            "  - url: https://youtube.com/@channel1\n"
            "    name: Channel One\n"
            "    category: recital\n"
            "  - url: https://youtube.com/@channel2\n"
            "    name: Channel Two\n"
            "    category: conservatory\n"
        )
        channels = load_channel_list(channels_yaml)
        assert len(channels) == 2
        assert channels[0]["name"] == "Channel One"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_channel_list(Path("/nonexistent/channels.yaml"))


class TestDiscoverChannelVideos:
    @patch("model_improvement.youtube_piano.subprocess")
    def test_returns_video_list(self, mock_subprocess):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            '{"id": "abc123", "title": "Chopin Ballade No 1", "duration": 600}\n'
            '{"id": "def456", "title": "Beethoven Sonata No 14", "duration": 900}\n'
        )
        mock_subprocess.run.return_value = mock_result

        videos = discover_channel_videos("https://youtube.com/@channel1", max_videos=10)
        assert len(videos) == 2
        assert videos[0]["id"] == "abc123"

    @patch("model_improvement.youtube_piano.subprocess")
    def test_handles_yt_dlp_failure(self, mock_subprocess):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error"
        mock_subprocess.run.return_value = mock_result

        videos = discover_channel_videos("https://youtube.com/@bad", max_videos=10)
        assert videos == []


class TestDownloadPianoAudio:
    @patch("model_improvement.youtube_piano._download_audio_yt_dlp")
    def test_downloads_and_writes_metadata(self, mock_download, tmp_path):
        cache_dir = tmp_path / "youtube_piano_cache"

        # Mock download to create a fake WAV file
        def fake_download(url, output_path):
            import soundfile as sf
            audio = np.random.randn(24000 * 60).astype(np.float32)
            sf.write(str(output_path), audio, 24000)

        mock_download.side_effect = fake_download

        videos = [
            {"id": "abc123", "title": "Chopin Ballade", "duration": 600,
             "channel": "TestChannel", "url": "https://youtube.com/watch?v=abc123"},
        ]

        records = download_piano_audio(videos, cache_dir)
        assert len(records) == 1
        assert (cache_dir / "audio" / "abc123.wav").exists()

        # Metadata should be written
        import jsonlines
        with jsonlines.open(cache_dir / "recordings.jsonl") as reader:
            metadata = list(reader)
        assert len(metadata) == 1
        assert metadata[0]["video_id"] == "abc123"
