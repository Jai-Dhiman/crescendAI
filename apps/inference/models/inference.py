"""A1-Max MuQ inference - MuQ embedding extraction and prediction."""

import numpy as np
import torch
from constants import MODEL_CONFIG

from models.loader import A1MaxInferenceHeadGaussian, ModelCache


@torch.no_grad()
def extract_muq_embeddings(
    audio: np.ndarray,
    cache: ModelCache,
    layer_start: int = None,
    layer_end: int = None,
    max_frames: int = None,
) -> torch.Tensor:
    """Extract MuQ embeddings from audio waveform.

    Averages hidden states from layers 9-12 (best performing range).

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
def predict_with_ensemble(
    embeddings: torch.Tensor,
    cache: ModelCache,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Get predictions from 4-fold ensemble of A1-Max heads.

    Args:
        embeddings: Frame embeddings [T, D] from MuQ.
        cache: Model cache with loaded heads.

    Returns:
        Scalar heads: averaged predictions [6] across all folds.
        Gaussian heads: (mu [6], sigma [6]) averaged across all folds.
    """
    if not cache.muq_heads:
        raise RuntimeError("No A1-Max heads loaded in cache")

    if isinstance(cache.muq_heads[0], A1MaxInferenceHeadGaussian):
        all_mu, all_sigma = [], []
        for head in cache.muq_heads:
            mu, sigma = head(embeddings)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
        return np.mean(all_mu, axis=0), np.mean(all_sigma, axis=0)

    predictions = []
    for head in cache.muq_heads:
        pred = head(embeddings).cpu().numpy()
        predictions.append(pred)
    return np.mean(predictions, axis=0)
