"""Feature extraction for masterclass segments."""

from __future__ import annotations

from pathlib import Path

import torch

from audio_experiments.extractors.muq import MuQExtractor
from masterclass_experiments.data import Segment


def stats_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean + std pooling over time dimension.

    Args:
        embeddings: [T, D] tensor.

    Returns:
        [2*D] tensor (mean concatenated with std).
    """
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    return torch.cat([mean, std])


def extract_muq_features(
    segments: list[Segment],
    segment_dir: Path,
    cache_dir: Path,
) -> dict[str, torch.Tensor]:
    """Extract stats-pooled MuQ embeddings for each segment.

    Args:
        segments: List of segments to process.
        segment_dir: Directory containing segment WAV files.
        cache_dir: Directory to cache raw MuQ embeddings.

    Returns:
        Dict mapping segment_id to [2048] pooled embedding tensor.
    """
    extractor = MuQExtractor(cache_dir=cache_dir)
    features: dict[str, torch.Tensor] = {}

    for seg in segments:
        wav_path = segment_dir / f"{seg.segment_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Segment WAV not found: {wav_path}")

        # MuQExtractor handles caching internally
        raw = extractor.extract_from_file(wav_path)  # [T, 1024]
        pooled = stats_pool(raw)  # [2048]
        features[seg.segment_id] = pooled

    return features
