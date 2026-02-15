"""Feature extraction for masterclass segments."""

from __future__ import annotations

from pathlib import Path

import torch

from audio_experiments.extractors.muq import MuQExtractor
from audio_experiments.models.muq_models import MuQStatsModel
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


@torch.no_grad()
def extract_quality_scores(
    raw_embeddings: dict[str, torch.Tensor],
    checkpoint_paths: Path | list[Path],
) -> dict[str, torch.Tensor]:
    """Run PercePiano model inference to get 19-dim quality scores.

    When multiple checkpoint paths are provided, predictions are averaged
    across all models (ensemble).

    Args:
        raw_embeddings: Dict mapping segment_id to [T, 1024] raw MuQ embeddings.
        checkpoint_paths: Single checkpoint path or list of paths to average.

    Returns:
        Dict mapping segment_id to [19] quality score tensor.
    """
    if isinstance(checkpoint_paths, Path):
        checkpoint_paths = [checkpoint_paths]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load all fold models
    models = []
    for ckpt in checkpoint_paths:
        model = MuQStatsModel.load_from_checkpoint(ckpt)
        model.eval()
        model = model.to(device)
        models.append(model)

    scores: dict[str, torch.Tensor] = {}

    for seg_id, emb in raw_embeddings.items():
        x = emb.unsqueeze(0).to(device)  # [1, T, 1024]
        mask = torch.ones(1, emb.shape[0], device=device)  # [1, T]

        # Average predictions across all fold models
        preds = []
        for model in models:
            pooled = model.pool(x, mask)  # [1, 2048]
            pred = model.clf(pooled)  # [1, 19]
            preds.append(pred)

        avg_pred = torch.stack(preds).mean(dim=0)  # [1, 19]
        scores[seg_id] = avg_pred.squeeze(0).cpu()

    return scores
