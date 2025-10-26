import torch
import torch.nn as nn
from typing import Optional, Tuple


class MIDIBertEncoder(nn.Module):
    """
    MIDI encoder using BERT-style architecture for OctupleMIDI tokens.

    Architecture:
    - Embedding layer for each OctupleMIDI dimension
    - Transformer encoder (6 layers, 256 hidden dim)
    - Positional encoding
    - Output: Event-level embeddings

    Input: OctupleMIDI tokens [batch, events, 8]
    Output: Event embeddings [batch, events, 256]
    """

    def __init__(
        self,
        vocab_sizes: Optional[dict] = None,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
    ):
        """
        Initialize MIDI encoder.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes for each dimension
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        if vocab_sizes is None:
            # Default vocabulary sizes from OctupleMIDI
            vocab_sizes = {
                'type': 5,
                'beat': 16,
                'position': 16,
                'pitch': 88,  # 88 piano keys
                'duration': 128,
                'velocity': 128,
                'instrument': 1,
                'bar': 512,
            }

        self.vocab_sizes = vocab_sizes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        # Embedding layers for each OctupleMIDI dimension
        self.type_embed = nn.Embedding(vocab_sizes['type'], hidden_size // 8)
        self.beat_embed = nn.Embedding(vocab_sizes['beat'], hidden_size // 8)
        self.position_embed = nn.Embedding(vocab_sizes['position'], hidden_size // 8)
        self.pitch_embed = nn.Embedding(vocab_sizes['pitch'], hidden_size // 4)
        self.duration_embed = nn.Embedding(vocab_sizes['duration'], hidden_size // 8)
        self.velocity_embed = nn.Embedding(vocab_sizes['velocity'], hidden_size // 8)
        self.instrument_embed = nn.Embedding(vocab_sizes['instrument'], hidden_size // 16)
        self.bar_embed = nn.Embedding(vocab_sizes['bar'], hidden_size // 16)

        # Calculate total embedding dimension
        total_embed_dim = (
            hidden_size // 8 +  # type
            hidden_size // 8 +  # beat
            hidden_size // 8 +  # position
            hidden_size // 4 +  # pitch (larger - most important)
            hidden_size // 8 +  # duration
            hidden_size // 8 +  # velocity
            hidden_size // 16 +  # instrument
            hidden_size // 16   # bar
        )

        # Project concatenated embeddings to hidden_size
        self.embed_projection = nn.Linear(total_embed_dim, hidden_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_size)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through MIDI encoder.

        Args:
            midi_tokens: OctupleMIDI tokens [batch, events, 8]
            attention_mask: Optional attention mask [batch, events]

        Returns:
            Event embeddings [batch, events, hidden_size]
        """
        batch_size, num_events, num_dims = midi_tokens.shape
        assert num_dims == 8, f"Expected 8 dimensions, got {num_dims}"

        # Embed each dimension
        type_emb = self.type_embed(midi_tokens[:, :, 0])
        beat_emb = self.beat_embed(midi_tokens[:, :, 1])
        position_emb = self.position_embed(midi_tokens[:, :, 2])
        pitch_emb = self.pitch_embed(midi_tokens[:, :, 3])
        duration_emb = self.duration_embed(midi_tokens[:, :, 4])
        velocity_emb = self.velocity_embed(midi_tokens[:, :, 5])
        instrument_emb = self.instrument_embed(midi_tokens[:, :, 6])
        bar_emb = self.bar_embed(midi_tokens[:, :, 7])

        # Concatenate all embeddings
        combined = torch.cat([
            type_emb,
            beat_emb,
            position_emb,
            pitch_emb,
            duration_emb,
            velocity_emb,
            instrument_emb,
            bar_emb,
        ], dim=-1)

        # Project to hidden size
        embeddings = self.embed_projection(combined)

        # Add positional encoding
        seq_len = num_events
        if seq_len <= self.max_seq_length:
            embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        else:
            # Truncate or handle longer sequences
            embeddings = embeddings + self.positional_encoding[:, :self.max_seq_length, :]

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = mask out)
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None

        # Pass through transformer
        output = self.transformer(
            embeddings,
            src_key_padding_mask=transformer_mask,
        )

        # Layer normalization
        output = self.layer_norm(output)

        return output

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_size


class SimplifiedMIDIEncoder(nn.Module):
    """
    Simplified MIDI encoder using just embeddings + LSTM.

    For testing when transformer is too heavy.
    """

    def __init__(
        self,
        vocab_sizes: Optional[dict] = None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize simplified MIDI encoder.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes
            hidden_size: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        if vocab_sizes is None:
            vocab_sizes = {
                'type': 5,
                'beat': 16,
                'position': 16,
                'pitch': 88,
                'duration': 128,
                'velocity': 128,
                'instrument': 1,
                'bar': 512,
            }

        self.hidden_size = hidden_size

        # Simple embeddings for pitch and velocity (most important)
        self.pitch_embed = nn.Embedding(vocab_sizes['pitch'], hidden_size // 2)
        self.velocity_embed = nn.Embedding(vocab_sizes['velocity'], hidden_size // 2)

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # Project bidirectional LSTM output back to hidden_size
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        midi_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            midi_tokens: OctupleMIDI tokens [batch, events, 8]
            attention_mask: Optional mask (not used)

        Returns:
            Event embeddings [batch, events, hidden_size]
        """
        # Extract pitch and velocity
        pitch = midi_tokens[:, :, 3]
        velocity = midi_tokens[:, :, 5]

        # Embed
        pitch_emb = self.pitch_embed(pitch)
        velocity_emb = self.velocity_embed(velocity)

        # Concatenate
        combined = torch.cat([pitch_emb, velocity_emb], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(combined)

        # Project
        output = self.projection(lstm_out)

        return output

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_size


def create_midi_encoder(
    use_transformer: bool = True,
    hidden_size: int = 256,
    num_layers: int = 6,
    **kwargs
) -> nn.Module:
    """
    Factory function to create MIDI encoder.

    Args:
        use_transformer: If True, use transformer; else use LSTM
        hidden_size: Hidden dimension
        num_layers: Number of layers
        **kwargs: Additional arguments

    Returns:
        MIDI encoder module
    """
    if use_transformer:
        return MIDIBertEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )
    else:
        return SimplifiedMIDIEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )


if __name__ == "__main__":
    print("MIDI encoder module loaded successfully")
    print("- MIDIBert-style transformer encoder")
    print("- Simplified LSTM encoder (fallback)")
    print("- OctupleMIDI tokenization support")
