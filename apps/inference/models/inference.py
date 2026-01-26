"""M1c MuQ L9-12 inference - MuQ embedding extraction and prediction."""

import numpy as np
import torch

from constants import MODEL_CONFIG
from models.loader import ModelCache


@torch.no_grad()
def extract_muq_embeddings(
    audio: np.ndarray,
    cache: ModelCache,
    layer_start: int = None,
    layer_end: int = None,
    max_frames: int = None,
) -> torch.Tensor:
    """Extract MuQ embeddings from audio waveform.

    Averages hidden states from layers 9-12 (best performing range for M1c).

    Args:
        audio: Audio waveform at 24kHz
        cache: Model cache with loaded MuQ model
        layer_start: Start layer (inclusive), default 9
        layer_end: End layer (exclusive), default 13
        max_frames: Maximum frames to keep

    Returns:
        Embeddings tensor [T, 1024] where T is number of frames
    """
    layer_start = layer_start or MODEL_CONFIG["muq_layer_start"]
    layer_end = layer_end or MODEL_CONFIG["muq_layer_end"]
    max_frames = max_frames or MODEL_CONFIG["max_frames"]

    # MuQ expects [B, samples] tensor
    wavs = torch.tensor(audio).unsqueeze(0).to(cache.device)

    # Get hidden states from all layers
    outputs = cache.muq_model(wavs, output_hidden_states=True)

    # Average layers 9-12 (indices in hidden_states tuple)
    # hidden_states is tuple of [B, T, D] tensors
    hidden_states = outputs.hidden_states[layer_start:layer_end]
    embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0)

    if embeddings.shape[0] > max_frames:
        embeddings = embeddings[:max_frames]

    return embeddings


@torch.no_grad()
def stats_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """Apply statistical pooling (mean + std) to frame embeddings.

    Args:
        embeddings: Frame embeddings [T, D]

    Returns:
        Pooled representation [D*2] (mean concatenated with std)
    """
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0)


@torch.no_grad()
def predict_with_ensemble(
    embeddings: torch.Tensor,
    cache: ModelCache,
) -> np.ndarray:
    """Get predictions from 4-fold ensemble of MuQ heads.

    Args:
        embeddings: Frame embeddings [T, D] from MuQ
        cache: Model cache with loaded heads

    Returns:
        Averaged predictions [19] across all folds
    """
    if not cache.muq_heads:
        raise RuntimeError("No MuQ heads loaded in cache")

    # Apply stats pooling
    pooled = stats_pool(embeddings)

    # Get predictions from each fold
    predictions = []
    for head in cache.muq_heads:
        pred = head(pooled).cpu().numpy()
        predictions.append(pred)

    return np.mean(predictions, axis=0)
