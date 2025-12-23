"""
Gated Multimodal Unit (GMU) fusion for audio and MIDI modalities.

Uses learned gating to determine how much to trust each modality per-sample.
This is more robust than cross-attention when encoders have misaligned
representation spaces.

Reference: "Gated Multimodal Units for Information Fusion" (Arevalo et al., 2017)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated Multimodal Unit for audio-MIDI fusion.

    Architecture:
        gate = sigmoid(W_g @ concat(audio, midi) + b_g)
        fused = gate * transform_a(audio) + (1 - gate) * transform_m(midi)

    The gate learns which modality to trust for each sample/timestep.
    """

    def __init__(
        self,
        audio_dim: int = 512,
        midi_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize gated fusion.

        Args:
            audio_dim: Audio projection output dimension
            midi_dim: MIDI projection output dimension
            output_dim: Fused output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.midi_dim = midi_dim
        self.output_dim = output_dim

        # Gating network: learns modality weighting
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + midi_dim, output_dim),
            nn.Sigmoid(),
        )

        # Transform networks for each modality
        self.audio_transform = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.midi_transform = nn.Sequential(
            nn.Linear(midi_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Single-modality fallbacks
        self.audio_only_proj = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.midi_only_proj = nn.Sequential(
            nn.Linear(midi_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        midi_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through gated fusion.

        Args:
            audio_features: Audio projections [batch, seq_len, audio_dim]
            midi_features: MIDI projections [batch, seq_len, midi_dim]
            audio_mask: Audio attention mask (unused, for API compatibility)
            midi_mask: MIDI attention mask (unused, for API compatibility)

        Returns:
            Tuple of:
                - fused: Fused features [batch, seq_len, output_dim]
                - gate_info: Dict with gate values for diagnostics
        """
        # Handle single-modality cases
        if audio_features is None and midi_features is None:
            raise ValueError(
                "At least one of audio_features or midi_features must be provided"
            )

        if audio_features is None:
            # MIDI-only mode
            fused = self.midi_only_proj(midi_features)
            return fused, {"gate_values": None, "mode": "midi_only"}

        if midi_features is None:
            # Audio-only mode
            fused = self.audio_only_proj(audio_features)
            return fused, {"gate_values": None, "mode": "audio_only"}

        # Multi-modal fusion
        # Align sequence lengths (use shorter)
        min_seq_len = min(audio_features.shape[1], midi_features.shape[1])
        audio_aligned = audio_features[:, :min_seq_len, :]
        midi_aligned = midi_features[:, :min_seq_len, :]

        # Concatenate for gating
        concat = torch.cat([audio_aligned, midi_aligned], dim=-1)

        # Compute gate values
        gate_values = self.gate(concat)  # [batch, seq_len, output_dim]

        # Transform each modality
        audio_transformed = self.audio_transform(audio_aligned)
        midi_transformed = self.midi_transform(midi_aligned)

        # Gated combination
        # gate * audio + (1 - gate) * midi
        fused = gate_values * audio_transformed + (1 - gate_values) * midi_transformed

        # Output projection
        fused = self.output_proj(fused)

        gate_info = {
            "gate_values": gate_values,
            "gate_mean": gate_values.mean().item(),
            "mode": "fusion",
        }

        return fused, gate_info

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) fusion.

    MIDI features modulate audio processing via learned scale and shift:
        output = gamma(midi) * audio + beta(midi)

    This is appropriate when MIDI should guide what audio features matter.

    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
    """

    def __init__(
        self,
        audio_dim: int = 512,
        midi_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize FiLM fusion.

        Args:
            audio_dim: Audio projection dimension
            midi_dim: MIDI projection dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.midi_dim = midi_dim
        self.output_dim = output_dim

        # MIDI -> modulation parameters (gamma, beta)
        self.film_generator = nn.Sequential(
            nn.Linear(midi_dim, output_dim * 2),
        )

        # Audio transform (before modulation)
        self.audio_transform = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Single-modality fallbacks
        self.audio_only_proj = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.midi_only_proj = nn.Sequential(
            nn.Linear(midi_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        midi_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through FiLM fusion.

        Args:
            audio_features: Audio projections [batch, seq_len, audio_dim]
            midi_features: MIDI projections [batch, seq_len, midi_dim]
            audio_mask: Unused
            midi_mask: Unused

        Returns:
            Tuple of (fused_features, modulation_info)
        """
        if audio_features is None and midi_features is None:
            raise ValueError("At least one modality required")

        if audio_features is None:
            return self.midi_only_proj(midi_features), {"mode": "midi_only"}

        if midi_features is None:
            return self.audio_only_proj(audio_features), {"mode": "audio_only"}

        # Align sequences
        min_seq_len = min(audio_features.shape[1], midi_features.shape[1])
        audio_aligned = audio_features[:, :min_seq_len, :]
        midi_aligned = midi_features[:, :min_seq_len, :]

        # Generate modulation parameters from MIDI
        film_params = self.film_generator(midi_aligned)
        gamma, beta = film_params.chunk(2, dim=-1)

        # Transform audio
        audio_transformed = self.audio_transform(audio_aligned)

        # Apply FiLM modulation
        modulated = gamma * audio_transformed + beta

        # Output projection
        fused = self.output_proj(modulated)

        modulation_info = {
            "gamma_mean": gamma.mean().item(),
            "beta_mean": beta.mean().item(),
            "mode": "fusion",
        }

        return fused, modulation_info

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


if __name__ == "__main__":
    print("Gated fusion module loaded successfully")
    print("- GatedFusion: GMU-style gating")
    print("- FiLMFusion: Feature-wise Linear Modulation")
    print("- Both support single-modality fallback")
