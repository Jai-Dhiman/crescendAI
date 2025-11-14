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

        # Clamp all tokens to valid vocabulary ranges (defensive programming)
        # This protects against malformed data that bypassed tokenizer validation
        midi_tokens = midi_tokens.clone()  # Don't modify input in-place
        midi_tokens[:, :, 0] = torch.clamp(midi_tokens[:, :, 0], 0, self.vocab_sizes['type'] - 1)
        midi_tokens[:, :, 1] = torch.clamp(midi_tokens[:, :, 1], 0, self.vocab_sizes['beat'] - 1)
        midi_tokens[:, :, 2] = torch.clamp(midi_tokens[:, :, 2], 0, self.vocab_sizes['position'] - 1)
        midi_tokens[:, :, 3] = torch.clamp(midi_tokens[:, :, 3], 0, self.vocab_sizes['pitch'] - 1)
        midi_tokens[:, :, 4] = torch.clamp(midi_tokens[:, :, 4], 0, self.vocab_sizes['duration'] - 1)
        midi_tokens[:, :, 5] = torch.clamp(midi_tokens[:, :, 5], 0, self.vocab_sizes['velocity'] - 1)
        midi_tokens[:, :, 6] = torch.clamp(midi_tokens[:, :, 6], 0, self.vocab_sizes['instrument'] - 1)
        midi_tokens[:, :, 7] = torch.clamp(midi_tokens[:, :, 7], 0, self.vocab_sizes['bar'] - 1)

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


if __name__ == "__main__":
    print("MIDI encoder module loaded successfully")
    print("PRODUCTION: MIDIBert-style transformer encoder only")
    print("- OctupleMIDI tokenization (8-tuple per event)")
    print("- Transformer encoder (6 layers, 256 hidden)")
    print("- Positional encoding for temporal awareness")
