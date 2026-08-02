"""The seam: a narrow interface both real backbones and test fakes implement,
so extraction and its tests never depend on Aria or MoonBeam internals."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np


class Backbone(Protocol):
    def embed(self, midi_path: Path) -> dict:
        """Return one or more named pooled embedding vectors for one MIDI file."""
        ...


class FakeBackbone:
    """Deterministic, weight-free stand-in: each pooling name maps to a
    fixed-length vector derived from a hash of the MIDI path, so different
    paths get different (but reproducible) vectors and tests never touch a
    real model."""

    def __init__(self, pooling_names: tuple[str, ...] = ("embedding",), dim: int = 8):
        self.pooling_names = pooling_names
        self.dim = dim

    def embed(self, midi_path: Path) -> dict:
        seed = int(hashlib.sha256(str(midi_path).encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return {name: rng.random(self.dim).astype(np.float32) for name in self.pooling_names}
