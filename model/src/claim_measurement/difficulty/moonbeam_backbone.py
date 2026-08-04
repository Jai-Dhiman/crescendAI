"""MoonBeam-839M adapter -- an INTEGRATION SPIKE (issue #138 design). MoonBeam's
pooling API is undocumented and untested until the human GPU run against the
real 839M checkpoint; this class only owns the pooling MATH (mean-over-tokens
vs. last-token), injected behind a `loader` callable so the class is fully
testable without the transformers_minimal fork or moonbeam_839M.pt installed.

The real loader (checkpoint + tokenizer + forward pass) lives in
moonbeam_extract_script.py, which runs under an ISOLATED uv-managed Python
3.12 venv -- see that file's module docstring for the setup recipe. Importing
THIS module never requires that venv; only calling MoonBeamBackbone with a
real loader does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


class MoonBeamBackbone:
    """Backbone protocol implementation. `loader(midi_path) -> np.ndarray`
    must return raw per-token hidden states, shape (seq_len, hidden_dim);
    this class only does the pooling, never the checkpoint/tokenizer call."""

    def __init__(self, loader: Callable[[Path], np.ndarray] | None = None):
        if loader is None:
            raise ValueError(
                "MoonBeamBackbone requires an explicit `loader` (real checkpoint "
                "inference lives in moonbeam_extract_script.py, run under the "
                "isolated MoonBeam venv -- see that file's docstring for setup)."
            )
        self._loader = loader

    def embed(self, midi_path: Path) -> dict:
        hidden_states = np.asarray(self._loader(midi_path), dtype=np.float32)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"loader must return (seq_len, hidden_dim) hidden states, "
                f"got shape {hidden_states.shape}"
            )
        return {
            "mean_pool": hidden_states.mean(axis=0),
            "last_token": hidden_states[-1],
        }
