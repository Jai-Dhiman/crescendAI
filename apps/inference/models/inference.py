"""D9c AsymmetricGatedFusion inference - MERT and MuQ embedding extraction."""

import numpy as np
import torch

from constants import MODEL_CONFIG
from models.loader import ModelCache


@torch.no_grad()
def extract_mert_embeddings(
    audio: np.ndarray,
    cache: ModelCache,
    layer_start: int = None,
    layer_end: int = None,
    max_frames: int = None,
) -> torch.Tensor:
    """Extract MERT embeddings from audio waveform.

    Averages hidden states from layers 13-24 (best performing range).
    """
    layer_start = layer_start or MODEL_CONFIG["mert_layer_start"]
    layer_end = layer_end or MODEL_CONFIG["mert_layer_end"]
    max_frames = max_frames or MODEL_CONFIG["max_frames"]

    inputs = cache.mert_processor(
        audio,
        sampling_rate=MODEL_CONFIG["target_sr"],
        return_tensors="pt",
    )
    inputs = {k: v.to(cache.device) for k, v in inputs.items()}

    outputs = cache.mert_model(**inputs)

    hidden_states = outputs.hidden_states[layer_start:layer_end]
    embeddings = torch.stack(hidden_states, dim=0).mean(dim=0).squeeze(0)

    if embeddings.shape[0] > max_frames:
        embeddings = embeddings[:max_frames]

    return embeddings


@torch.no_grad()
def extract_muq_embeddings(
    audio: np.ndarray,
    cache: ModelCache,
    max_frames: int = None,
) -> torch.Tensor:
    """Extract MuQ embeddings from audio waveform."""
    max_frames = max_frames or MODEL_CONFIG["max_frames"]

    wavs = torch.tensor(audio).unsqueeze(0).to(cache.device)
    outputs = cache.muq_model(wavs, output_hidden_states=True)
    embeddings = outputs.last_hidden_state.squeeze(0)

    if embeddings.shape[0] > max_frames:
        embeddings = embeddings[:max_frames]

    return embeddings


@torch.no_grad()
def predict_with_fusion_ensemble(
    mert_embeddings: torch.Tensor,
    muq_embeddings: torch.Tensor,
    cache: ModelCache,
) -> np.ndarray:
    """Get predictions from 4-fold ensemble of AsymmetricGatedFusion heads."""
    if not cache.fusion_heads:
        raise RuntimeError("No fusion heads loaded in cache")

    predictions = []
    for head in cache.fusion_heads:
        pred = head(mert_embeddings, muq_embeddings).cpu().numpy()
        predictions.append(pred)

    return np.mean(predictions, axis=0)
