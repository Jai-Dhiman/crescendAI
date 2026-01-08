"""MERT embedding extraction and inference."""

from typing import Optional

import numpy as np
import torch

from ..constants import MERT_CONFIG
from .loader import ModelCache


@torch.no_grad()
def extract_mert_embeddings(
    audio: np.ndarray,
    cache: ModelCache,
    layer_start: int = MERT_CONFIG["layer_start"],
    layer_end: int = MERT_CONFIG["layer_end"],
    max_frames: int = MERT_CONFIG["max_frames"],
) -> torch.Tensor:
    """Extract MERT embeddings from audio waveform.

    Args:
        audio: Audio waveform at 24kHz sample rate
        cache: Model cache with loaded MERT model
        layer_start: First hidden layer to include (0-indexed)
        layer_end: Last hidden layer (exclusive)
        max_frames: Maximum sequence length to return

    Returns:
        Tensor of shape (T, 1024) where T <= max_frames
    """
    # Process audio through MERT processor
    inputs = cache.mert_processor(
        audio,
        sampling_rate=MERT_CONFIG["target_sr"],
        return_tensors="pt",
    )
    inputs = {k: v.to(cache.device) for k, v in inputs.items()}

    # Get hidden states from MERT
    outputs = cache.mert_model(**inputs)

    # Average layers in specified range (13-24 works best)
    hidden_states = outputs.hidden_states[layer_start:layer_end]
    embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0)

    # Truncate if too long
    if embeddings.shape[0] > max_frames:
        embeddings = embeddings[:max_frames]

    return embeddings  # Shape: (T, 1024)


@torch.no_grad()
def predict_with_mert_ensemble(
    embeddings: torch.Tensor,
    cache: ModelCache,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Get predictions from 4-fold ensemble of MERT MLP heads.

    Args:
        embeddings: MERT embeddings of shape (T, 1024)
        cache: Model cache with loaded MLP heads
        mask: Optional attention mask

    Returns:
        Array of shape (19,) with averaged predictions
    """
    if not cache.mert_heads:
        raise RuntimeError("No MERT heads loaded in cache")

    # Mean pool the sequence (matching training)
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (embeddings * mask_expanded).sum(0) / mask_expanded.sum(0).clamp(min=1)
    else:
        pooled = embeddings.mean(dim=0)  # Shape: (1024,)

    pooled = pooled.unsqueeze(0)  # Shape: (1, 1024)

    # Get predictions from each fold
    predictions = []
    for head in cache.mert_heads:
        pred = head(pooled).cpu().numpy()
        predictions.append(pred)

    # Average ensemble predictions
    return np.mean(predictions, axis=0).squeeze()  # Shape: (19,)
