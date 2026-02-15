import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from masterclass_experiments.data import Segment
from masterclass_experiments.features import extract_muq_features, stats_pool


def test_stats_pool_produces_correct_shape():
    embeddings = torch.randn(10, 1024)
    pooled = stats_pool(embeddings)
    assert pooled.shape == (2048,)


def test_stats_pool_values():
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pooled = stats_pool(embeddings)
    # mean = [2.0, 3.0], std = [sqrt(2), sqrt(2)] ~ [1.414, 1.414]
    assert torch.allclose(pooled[:2], torch.tensor([2.0, 3.0]))
    assert pooled.shape == (4,)


def test_extract_muq_features_returns_dict_of_pooled_vectors():
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_dir = Path(tmpdir) / "segments"
        seg_dir.mkdir()
        cache_dir = Path(tmpdir) / "muq_cache"

        # Create a dummy WAV
        sr = 24000
        audio = np.random.randn(sr * 2).astype(np.float32)
        sf.write(seg_dir / "stop_0000.wav", audio, sr)

        segments = [
            Segment(
                segment_id="stop_0000",
                video_id="vid1",
                label="stop",
                start_time=0.0,
                end_time=2.0,
                moment_id="a",
            )
        ]

        # Mock MuQExtractor to avoid loading real model
        fake_embedding = torch.randn(10, 1024)  # [T, 1024]
        with patch("masterclass_experiments.features.MuQExtractor") as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract_from_file.return_value = fake_embedding

            features = extract_muq_features(segments, seg_dir, cache_dir)

        assert "stop_0000" in features
        assert features["stop_0000"].shape == (2048,)  # mean + std
