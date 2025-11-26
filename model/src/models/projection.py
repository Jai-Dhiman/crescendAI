"""
Projection heads for aligning MERT and MIDIBert representations.

Maps different encoder outputs to a shared embedding space before fusion.
This addresses the representation misalignment problem identified in the research:
MERT (768-dim, pre-trained) and MIDIBert (256-dim, scratch) have incompatible
embedding geometries that cause cross-attention to fail.

Reference: CLIP, MuLan, CLaMP approaches for multi-modal alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProjectionHead(nn.Module):
    """
    Single projection head for mapping encoder output to shared space.

    Architecture: Linear -> LayerNorm -> GELU -> Linear -> L2 normalize
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize projection head.

        Args:
            input_dim: Input dimension from encoder
            output_dim: Output dimension (shared space)
            hidden_dim: Hidden layer dimension (default: output_dim)
            dropout: Dropout probability
            normalize: Whether to L2-normalize output
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim

        self.normalize = normalize

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to shared space.

        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]

        Returns:
            Projected tensor [batch, seq_len, output_dim] or [batch, output_dim]
        """
        projected = self.projection(x)

        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected


class DualProjection(nn.Module):
    """
    Dual projection heads for audio and MIDI encoders.

    Maps both modalities to a shared 512-dimensional space for fusion.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        midi_dim: int = 256,
        shared_dim: int = 512,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize dual projection heads.

        Args:
            audio_dim: MERT output dimension (768 for MERT-95M)
            midi_dim: MIDIBert output dimension (256)
            shared_dim: Shared embedding space dimension
            dropout: Dropout probability
            normalize: Whether to L2-normalize outputs
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.midi_dim = midi_dim
        self.shared_dim = shared_dim

        # Audio projection: 768 -> 512 (skip if audio_dim=0 for MIDI-only mode)
        if audio_dim > 0:
            self.audio_proj = ProjectionHead(
                input_dim=audio_dim,
                output_dim=shared_dim,
                dropout=dropout,
                normalize=normalize,
            )
        else:
            self.audio_proj = None

        # MIDI projection: 256 -> 512 (skip if midi_dim=0 for audio-only mode)
        if midi_dim > 0:
            self.midi_proj = ProjectionHead(
                input_dim=midi_dim,
                output_dim=shared_dim,
                dropout=dropout,
                normalize=normalize,
            )
        else:
            self.midi_proj = None

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        midi_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Project both modalities to shared space.

        Args:
            audio_features: Audio encoder output [batch, seq_len, 768]
            midi_features: MIDI encoder output [batch, seq_len, 256]

        Returns:
            Tuple of (audio_projected, midi_projected) each [batch, seq_len, 512]
        """
        audio_proj = None
        midi_proj = None

        if audio_features is not None and self.audio_proj is not None:
            audio_proj = self.audio_proj(audio_features)

        if midi_features is not None and self.midi_proj is not None:
            midi_proj = self.midi_proj(midi_features)

        return audio_proj, midi_proj

    def get_pooled_embeddings(
        self,
        audio_features: Optional[torch.Tensor] = None,
        midi_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get pooled (mean) embeddings for contrastive learning.

        Args:
            audio_features: Audio encoder output [batch, seq_len, 768]
            midi_features: MIDI encoder output [batch, seq_len, 256]

        Returns:
            Tuple of (audio_pooled, midi_pooled) each [batch, 512]
        """
        audio_proj, midi_proj = self.forward(audio_features, midi_features)

        audio_pooled = None
        midi_pooled = None

        if audio_proj is not None:
            audio_pooled = audio_proj.mean(dim=1)  # [batch, 512]
            audio_pooled = F.normalize(audio_pooled, p=2, dim=-1)

        if midi_proj is not None:
            midi_pooled = midi_proj.mean(dim=1)  # [batch, 512]
            midi_pooled = F.normalize(midi_pooled, p=2, dim=-1)

        return audio_pooled, midi_pooled

    def get_output_dim(self) -> int:
        """Get shared embedding dimension."""
        return self.shared_dim


if __name__ == "__main__":
    print("Projection heads module loaded successfully")
    print("- Audio projection: 768 -> 512")
    print("- MIDI projection: 256 -> 512")
    print("- L2-normalized outputs for contrastive learning")
