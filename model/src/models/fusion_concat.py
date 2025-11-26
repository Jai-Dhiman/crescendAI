"""
Simple concatenation fusion baseline for Phase 2 diagnostic experiments.

Implements straightforward concatenation of audio and MIDI features
as a simpler alternative to cross-attention fusion.

Used to test whether fusion complexity is helping or hurting performance
in TRAINING_PLAN_v2.md Phase 2.

Reference: TRAINING_PLAN_v2.md Phase 2 - Experiment 3
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion module.

    Concatenates audio and MIDI features and projects to fusion_dim.
    No attention mechanism - just a simple linear projection.

    This serves as a simpler baseline to compare against cross-attention fusion.
    """

    def __init__(
        self,
        audio_dim: int = 768,
        midi_dim: int = 256,
        fusion_dim: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize concatenation fusion module.

        Args:
            audio_dim: Audio encoder output dimension
            midi_dim: MIDI encoder output dimension
            fusion_dim: Output dimension after fusion
            dropout: Dropout probability
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.midi_dim = midi_dim
        self.fusion_dim = fusion_dim

        # Projection layer for concatenated features
        # Input: audio_dim + midi_dim
        # Output: fusion_dim
        self.projection = nn.Sequential(
            nn.Linear(audio_dim + midi_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate projections for single-modal modes
        self.audio_only_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.midi_only_projection = nn.Sequential(
            nn.Linear(midi_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        midi_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass - concatenate and project features.

        Args:
            audio_features: Audio features [batch, audio_seq_len, audio_dim]
            midi_features: MIDI features [batch, midi_seq_len, midi_dim]
            audio_mask: Audio attention mask (not used, for API compatibility)
            midi_mask: MIDI attention mask (not used, for API compatibility)

        Returns:
            Tuple of:
                - fused_features: Fused features [batch, seq_len, fusion_dim]
                - attention: None (no attention in concatenation fusion)
        """
        # Handle single-modal cases
        if audio_features is None and midi_features is None:
            raise ValueError("At least one of audio_features or midi_features must be provided")

        if audio_features is None:
            # MIDI-only mode
            fused_features = self.midi_only_projection(midi_features)
            return fused_features, None

        if midi_features is None:
            # Audio-only mode
            fused_features = self.audio_only_projection(audio_features)
            return fused_features, None

        # Multi-modal fusion: concatenate features
        # Both features have shape [batch, seq_len, dim]

        # Ensure both sequences have same length (use shorter one)
        min_seq_len = min(audio_features.shape[1], midi_features.shape[1])

        audio_truncated = audio_features[:, :min_seq_len, :]
        midi_truncated = midi_features[:, :min_seq_len, :]

        # Concatenate along feature dimension
        concatenated = torch.cat([audio_truncated, midi_truncated], dim=-1)
        # [batch, seq_len, audio_dim + midi_dim]

        # Project to fusion_dim
        fused_features = self.projection(concatenated)
        # [batch, seq_len, fusion_dim]

        # Return None for attention (no attention mechanism)
        return fused_features, None


if __name__ == "__main__":
    print("Concatenation fusion module loaded successfully")
    print("- Simple concatenation + projection fusion")
    print("- No attention mechanism (simpler baseline)")
    print("- For Phase 2 diagnostic experiments")
