"""Aria-medium adapter: the thin seam between the Backbone protocol and the
real Aria weight-loading/inference call in model_improvement.aria_embeddings.
Loading real weights is the human-lit GPU boundary -- this class is tested by
monkeypatching extract_embedding, never by loading a real checkpoint."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class AriaBackbone:
    """Backbone protocol implementation over the existing 512-dim
    TransformerEMB embedding path."""

    def embed(self, midi_path: Path) -> dict:
        from model_improvement.aria_embeddings import extract_embedding
        vec = extract_embedding(midi_path, variant="embedding")
        return {"embedding": vec.detach().cpu().numpy().astype(np.float32)}
