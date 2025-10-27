import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for audio and MIDI modalities.

    Architecture:
    - Audio queries attend to MIDI keys/values (what notes are written?)
    - MIDI queries attend to audio keys/values (how are notes performed?)
    - Concatenate fused representations
    - Both audio and MIDI inputs are required (multi-modal mandatory)
    """

    def __init__(
        self,
        audio_dim: int = 768,
        midi_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_relative_position: bool = True,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            audio_dim: Audio feature dimension (MERT output)
            midi_dim: MIDI feature dimension (MIDIBert output)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_relative_position: Use relative positional encoding
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.midi_dim = midi_dim
        self.num_heads = num_heads
        self.use_relative_position = use_relative_position

        # Audio-to-MIDI cross-attention
        self.audio_to_midi_attn = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            kdim=midi_dim,
            vdim=midi_dim,
            dropout=dropout,
            batch_first=True,
        )

        # MIDI-to-Audio cross-attention
        self.midi_to_audio_attn = nn.MultiheadAttention(
            embed_dim=midi_dim,
            num_heads=num_heads,
            kdim=audio_dim,
            vdim=audio_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.audio_norm1 = nn.LayerNorm(audio_dim)
        self.midi_norm1 = nn.LayerNorm(midi_dim)

        # Feed-forward networks
        self.audio_ffn = nn.Sequential(
            nn.Linear(audio_dim, audio_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(audio_dim * 4, audio_dim),
            nn.Dropout(dropout),
        )

        self.midi_ffn = nn.Sequential(
            nn.Linear(midi_dim, midi_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(midi_dim * 4, midi_dim),
            nn.Dropout(dropout),
        )

        self.audio_norm2 = nn.LayerNorm(audio_dim)
        self.midi_norm2 = nn.LayerNorm(midi_dim)

        # Output dimension after concatenation
        self.output_dim = audio_dim + midi_dim

    def forward(
        self,
        audio_features: torch.Tensor,
        midi_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        midi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through cross-attention fusion.

        Args:
            audio_features: Audio embeddings [batch, audio_time, audio_dim]
            midi_features: MIDI embeddings [batch, midi_events, midi_dim] or None
            audio_mask: Audio attention mask [batch, audio_time]
            midi_mask: MIDI attention mask [batch, midi_events]

        Returns:
            Tuple of:
                - fused: Fused features [batch, audio_time, audio_dim + midi_dim]
                - attention_weights: Dict of attention weights (optional)
        """
        assert midi_features is not None, "MIDI features are required for multi-modal fusion"

        batch_size, audio_len, _ = audio_features.shape

        # Audio-to-MIDI cross-attention
        # Audio queries attend to MIDI
        audio_attended, audio_attn_weights = self.audio_to_midi_attn(
            query=audio_features,
            key=midi_features,
            value=midi_features,
            key_padding_mask=midi_mask if midi_mask is not None else None,
            need_weights=True,
        )

        # Residual connection + normalization
        audio_out = self.audio_norm1(audio_features + audio_attended)

        # Feed-forward
        audio_out = self.audio_norm2(audio_out + self.audio_ffn(audio_out))

        # MIDI-to-Audio cross-attention
        # MIDI queries attend to Audio
        midi_attended, midi_attn_weights = self.midi_to_audio_attn(
            query=midi_features,
            key=audio_features,
            value=audio_features,
            key_padding_mask=audio_mask if audio_mask is not None else None,
            need_weights=True,
        )

        # Residual connection + normalization
        midi_out = self.midi_norm1(midi_features + midi_attended)

        # Feed-forward
        midi_out = self.midi_norm2(midi_out + self.midi_ffn(midi_out))

        # Align MIDI features to audio time dimension
        if midi_out.shape[1] != audio_len:
            midi_out = self._align_sequences(midi_out, audio_len)

        # Concatenate audio and MIDI features
        fused = torch.cat([audio_out, midi_out], dim=-1)

        # Collect attention weights for visualization
        attention_weights = {
            "audio_to_midi": audio_attn_weights,
            "midi_to_audio": midi_attn_weights,
        }

        return fused, attention_weights

    def _audio_only_forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Process audio features without MIDI fusion.

        Args:
            audio_features: Audio embeddings [batch, time, audio_dim]

        Returns:
            Processed audio features
        """
        # Just apply feed-forward network
        audio_out = self.audio_norm1(audio_features)
        audio_out = self.audio_norm2(audio_out + self.audio_ffn(audio_out))
        return audio_out

    def _align_sequences(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Align sequence to target length using interpolation.

        Args:
            sequence: Input sequence [batch, seq_len, dim]
            target_length: Target sequence length

        Returns:
            Aligned sequence [batch, target_length, dim]
        """
        batch_size, seq_len, dim = sequence.shape

        if seq_len == target_length:
            return sequence

        # Transpose for interpolation: [batch, dim, seq_len]
        sequence = sequence.transpose(1, 2)

        # Interpolate
        aligned = torch.nn.functional.interpolate(
            sequence,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

        # Transpose back: [batch, target_length, dim]
        aligned = aligned.transpose(1, 2)

        return aligned

    def get_output_dim(self) -> int:
        """Get output dimension after fusion."""
        return self.output_dim


if __name__ == "__main__":
    print("Cross-attention fusion module loaded successfully")
    print("PRODUCTION: Multi-modal fusion only (Audio + MIDI required)")
    print("- Bidirectional cross-attention (audio↔MIDI)")
    print("- 8 attention heads for rich interaction")
    print("- Relative positional encoding")
    print("- Attention weight visualization support")
