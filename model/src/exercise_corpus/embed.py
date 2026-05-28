"""Thin adapter over aria_embeddings for exercise corpus embedding extraction.

Pins variant="embedding" (512-dim EOS-pooled) so the rest of the pipeline
never references aria_embeddings directly.
"""

from pathlib import Path

import torch

from model_improvement.aria_embeddings import extract_all_embeddings


def embed_primitives(midi_dir: Path) -> dict[str, torch.Tensor]:
    """Extract 512-dim Aria embeddings from all MIDI files in midi_dir.

    Args:
        midi_dir: directory containing .mid files (one per primitive).

    Returns:
        Dict mapping filename stem (primitive_id) to 512-dim float32 tensor.

    Raises:
        FileNotFoundError: if midi_dir contains no .mid files or does not exist.
    """
    return extract_all_embeddings(Path(midi_dir), variant="embedding")
