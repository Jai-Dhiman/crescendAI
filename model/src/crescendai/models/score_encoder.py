"""
Score encoder module for encoding score alignment features.

Encodes score alignment deviation features into a fixed-size representation
that can be fused with MIDI embeddings for performance evaluation.

Following PercePiano approach:
- Note-level deviation features -> Transformer encoder
- Global statistics -> MLP encoder
- Combined representation for fusion with MIDI features
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoteFeatureEncoder(nn.Module):
    """
    Encodes per-note deviation features using a transformer.

    Input: Note-level features [batch, num_notes, num_features]
    Output: Encoded sequence [batch, num_notes, hidden_dim]
    """

    def __init__(
        self,
        num_features: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_notes: int = 2048,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Project input features to hidden dimension
        self.input_projection = nn.Linear(num_features, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_notes, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        note_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode note-level deviation features.

        Args:
            note_features: [batch, num_notes, num_features]
            attention_mask: [batch, num_notes] (1 = valid, 0 = padding)

        Returns:
            Encoded features [batch, num_notes, hidden_dim]
        """
        batch_size, num_notes, _ = note_features.shape

        # Project to hidden dimension
        x = self.input_projection(note_features)

        # Add positional encoding
        if num_notes <= self.positional_encoding.size(1):
            x = x + self.positional_encoding[:, :num_notes, :]
        else:
            # Handle longer sequences by truncating
            x = x + self.positional_encoding[:, : self.positional_encoding.size(1), :]

        # Create transformer mask (True = mask out)
        if attention_mask is not None:
            transformer_mask = ~attention_mask.bool()
        else:
            transformer_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=transformer_mask)

        # Layer normalization
        x = self.layer_norm(x)

        return x


class GlobalFeatureEncoder(nn.Module):
    """
    Encodes global aggregated statistics.

    Input: Global features [batch, num_global_features]
    Output: Encoded features [batch, hidden_dim]
    """

    def __init__(
        self,
        num_features: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        Encode global statistics.

        Args:
            global_features: [batch, num_global_features]

        Returns:
            Encoded features [batch, hidden_dim]
        """
        return self.encoder(global_features)


class TempoCurveEncoder(nn.Module):
    """
    Encodes tempo curve using 1D convolutions.

    Input: Tempo ratios [batch, num_segments]
    Output: Encoded features [batch, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_segments: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_segments = max_segments

        # 1D convolutional encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),  # Fixed output size
            nn.Flatten(),
            nn.Linear(64 * 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, tempo_curve: torch.Tensor) -> torch.Tensor:
        """
        Encode tempo curve.

        Args:
            tempo_curve: [batch, num_segments]

        Returns:
            Encoded features [batch, hidden_dim]
        """
        # Add channel dimension
        x = tempo_curve.unsqueeze(1)  # [batch, 1, num_segments]

        return self.conv_layers(x)


class ScoreAlignmentEncoder(nn.Module):
    """
    Complete score alignment encoder combining all feature types.

    Encodes:
    - Per-note deviation features (onset, duration, velocity deviations)
    - Global aggregated statistics
    - Tempo curve

    Output can be:
    1. Sequence-level: For cross-attention fusion with MIDI embeddings
    2. Global-level: Single vector for concatenation with MIDI representation
    """

    def __init__(
        self,
        note_features: int = 6,
        global_features: int = 12,
        hidden_dim: int = 256,
        num_note_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_notes: int = 2048,
        output_mode: str = "both",  # 'sequence', 'global', or 'both'
    ):
        """
        Args:
            note_features: Number of per-note features
            global_features: Number of global statistic features
            hidden_dim: Hidden dimension for encoders
            num_note_layers: Number of transformer layers for note encoder
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_notes: Maximum number of notes
            output_mode: What to output ('sequence', 'global', or 'both')
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_mode = output_mode

        # Component encoders (each uses hidden_dim // 2)
        component_dim = hidden_dim // 2

        self.note_encoder = NoteFeatureEncoder(
            num_features=note_features,
            hidden_dim=component_dim,
            num_layers=num_note_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_notes=max_notes,
        )

        self.global_encoder = GlobalFeatureEncoder(
            num_features=global_features,
            hidden_dim=component_dim,
            dropout=dropout,
        )

        self.tempo_encoder = TempoCurveEncoder(
            hidden_dim=component_dim,
            dropout=dropout,
        )

        # Self-attention aggregation for sequence -> global (matching PercePiano)
        self.sequence_attention = nn.Sequential(
            nn.Linear(component_dim, component_dim // 2),
            nn.Tanh(),
            nn.Linear(component_dim // 2, 1, bias=False),
        )

        # Combine global and tempo into single global representation
        self.global_combiner = nn.Sequential(
            nn.Linear(component_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Project sequence for cross-attention with MIDI
        self.sequence_projection = nn.Linear(component_dim, hidden_dim)

    def forward(
        self,
        note_features: torch.Tensor,
        global_features: torch.Tensor,
        tempo_curve: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode score alignment features.

        Args:
            note_features: [batch, num_notes, 6] per-note deviations
            global_features: [batch, 12] aggregated statistics
            tempo_curve: [batch, num_segments] tempo ratios
            attention_mask: [batch, num_notes] (1 = valid, 0 = padding)

        Returns:
            Dictionary containing:
            - 'sequence': [batch, num_notes, hidden_dim] if output_mode in ['sequence', 'both']
            - 'global': [batch, hidden_dim] if output_mode in ['global', 'both']
        """
        outputs = {}

        # Encode note-level features
        note_encoded = self.note_encoder(note_features, attention_mask)  # [B, N, D/2]

        # Encode global statistics
        global_encoded = self.global_encoder(global_features)  # [B, D/2]

        # Encode tempo curve
        tempo_encoded = self.tempo_encoder(tempo_curve)  # [B, D/2]

        if self.output_mode in ["sequence", "both"]:
            # Project note sequence for cross-attention
            sequence_out = self.sequence_projection(note_encoded)  # [B, N, D]
            outputs["sequence"] = sequence_out

        if self.output_mode in ["global", "both"]:
            # Aggregate note sequence using attention
            attn_scores = self.sequence_attention(note_encoded).squeeze(-1)  # [B, N]
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    ~attention_mask.bool(), float("-inf")
                )
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]
            note_aggregated = torch.bmm(
                attn_weights.unsqueeze(1), note_encoded
            ).squeeze(1)  # [B, D/2]

            # Combine with global and tempo features
            combined_global = torch.cat(
                [
                    global_encoded + note_aggregated,  # Residual connection
                    tempo_encoded,
                ],
                dim=-1,
            )  # [B, D]

            global_out = self.global_combiner(combined_global)
            outputs["global"] = global_out

        return outputs

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_dim


class ScoreMIDIFusion(nn.Module):
    """
    Fuses score alignment features with MIDI embeddings.

    Supports multiple fusion strategies:
    1. Concatenation: Simple concat of global representations
    2. Cross-attention: Score attends to MIDI and vice versa
    3. Gated: Learned gates to weight contributions
    """

    def __init__(
        self,
        midi_dim: int = 768,
        score_dim: int = 256,
        output_dim: int = 768,
        fusion_type: str = "gated",  # 'concat', 'crossattn', or 'gated'
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.midi_dim = midi_dim
        self.score_dim = score_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(midi_dim + score_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(output_dim),
            )

        elif fusion_type == "crossattn":
            # Project score to midi dimension for attention
            self.score_projection = nn.Linear(score_dim, midi_dim)

            # Cross-attention: MIDI queries, score keys/values
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=midi_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(midi_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        elif fusion_type == "gated":
            # Project both to same dimension
            self.midi_projection = nn.Linear(midi_dim, output_dim)
            self.score_projection = nn.Linear(score_dim, output_dim)

            # Gating network
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )

            self.output_norm = nn.LayerNorm(output_dim)

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        midi_features: torch.Tensor,
        score_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse MIDI and score features.

        Args:
            midi_features: [batch, midi_dim] global MIDI representation
            score_features: [batch, score_dim] global score representation

        Returns:
            Fused features [batch, output_dim]
        """
        if self.fusion_type == "concat":
            combined = torch.cat([midi_features, score_features], dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == "crossattn":
            # Project score features
            score_proj = self.score_projection(score_features)

            # Add sequence dimension for attention
            midi_seq = midi_features.unsqueeze(1)  # [B, 1, D]
            score_seq = score_proj.unsqueeze(1)  # [B, 1, D]

            # Cross-attention
            attn_out, _ = self.cross_attention(
                query=midi_seq,
                key=score_seq,
                value=score_seq,
            )

            # Remove sequence dimension and project
            return self.output_projection(attn_out.squeeze(1))

        elif self.fusion_type == "gated":
            midi_proj = self.midi_projection(midi_features)
            score_proj = self.score_projection(score_features)

            # Compute gate
            combined = torch.cat([midi_proj, score_proj], dim=-1)
            gate = self.gate(combined)

            # Gated fusion
            fused = gate * midi_proj + (1 - gate) * score_proj

            return self.output_norm(fused)

        raise RuntimeError(f"Unknown fusion type: {self.fusion_type}")


class HierarchicalScoreEncoder(nn.Module):
    """
    Hierarchical score encoder using HAN (Hierarchical Attention Network).

    Implements the PercePiano-style hierarchical structure:
    - Note level: Process individual notes with Transformer/LSTM
    - Beat level: Aggregate notes within beats using attention
    - Measure level: Aggregate beats within measures using attention

    This captures musical structure at multiple temporal scales,
    critical for dimensions like tempo, phrasing, and overall interpretation.
    """

    def __init__(
        self,
        note_features: int = 20,
        global_features: int = 12,
        hidden_dim: int = 256,
        note_size: int = 128,
        beat_size: int = 64,
        measure_size: int = 64,
        num_note_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        use_voice_processing: bool = False,
        output_mode: str = "both",
    ):
        """
        Args:
            note_features: Number of per-note features (expanded to 20)
            global_features: Number of global statistic features
            hidden_dim: Final output hidden dimension
            note_size: Hidden size for note-level LSTM
            beat_size: Hidden size for beat-level LSTM
            measure_size: Hidden size for measure-level LSTM
            num_note_layers: Number of LSTM/Transformer layers at note level
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            use_voice_processing: Whether to use voice-level parallel processing
            output_mode: What to output ('sequence', 'global', or 'both')
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_mode = output_mode
        self.use_voice_processing = use_voice_processing

        # Import HAN components
        from .context_attention import ContextAttention
        from .han_encoder import HanEncoder, SimplifiedHanEncoder

        # Note-level projection
        self.note_projection = nn.Linear(note_features, note_size)

        # Hierarchical encoder (simplified or full based on voice processing)
        if use_voice_processing:
            self.han_encoder = HanEncoder(
                input_size=note_size,
                note_size=note_size,
                note_layers=num_note_layers,
                voice_size=beat_size,
                voice_layers=1,
                beat_size=beat_size,
                beat_layers=1,
                measure_size=measure_size,
                measure_layers=1,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
            han_output_dim = self.han_encoder.output_dim
        else:
            self.han_encoder = SimplifiedHanEncoder(
                input_size=note_size,
                hidden_size=note_size,
                num_layers=num_note_layers,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
            )
            han_output_dim = self.han_encoder.output_dim

        # Global feature encoder
        self.global_encoder = GlobalFeatureEncoder(
            num_features=global_features,
            hidden_dim=hidden_dim // 4,
            dropout=dropout,
        )

        # Tempo curve encoder
        self.tempo_encoder = TempoCurveEncoder(
            hidden_dim=hidden_dim // 4,
            dropout=dropout,
        )

        # Final attention aggregation for global output
        self.final_attention = ContextAttention(han_output_dim, num_attention_heads)

        # Combine hierarchical, global, and tempo features
        self.global_combiner = nn.Sequential(
            nn.Linear(han_output_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Project sequence for cross-attention with MIDI
        self.sequence_projection = nn.Linear(han_output_dim, hidden_dim)

    def forward(
        self,
        note_features: torch.Tensor,
        global_features: torch.Tensor,
        tempo_curve: torch.Tensor,
        note_locations: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode score features using hierarchical structure.

        Args:
            note_features: [batch, num_notes, note_features] per-note features
            global_features: [batch, 12] aggregated statistics
            tempo_curve: [batch, num_segments] tempo ratios
            note_locations: Dict with 'beat', 'measure', 'voice' tensors
            attention_mask: [batch, num_notes] (1 = valid, 0 = padding)

        Returns:
            Dictionary containing:
            - 'sequence': [batch, num_notes, hidden_dim] if output_mode in ['sequence', 'both']
            - 'global': [batch, hidden_dim] if output_mode in ['global', 'both']
            - 'hierarchical': Dict with note/beat/measure outputs
        """
        outputs = {}
        batch_size, num_notes, _ = note_features.shape

        # Project note features
        x = self.note_projection(note_features)

        # Create note_locations if not provided (default to flat structure)
        if note_locations is None:
            note_locations = self._create_default_note_locations(
                batch_size, num_notes, note_features.device
            )

        # Run hierarchical encoder
        han_outputs = self.han_encoder(x, note_locations)

        # Get hierarchical sequence (concatenated note/beat/measure)
        hierarchical_seq = han_outputs["total_note_cat"]  # [B, N, han_output_dim]

        outputs["hierarchical"] = han_outputs

        if self.output_mode in ["sequence", "both"]:
            # Project for cross-attention fusion
            sequence_out = self.sequence_projection(hierarchical_seq)
            outputs["sequence"] = sequence_out

        if self.output_mode in ["global", "both"]:
            # Aggregate sequence using context attention
            # Mask zero-padded positions
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hierarchical_seq)
                hierarchical_seq_masked = hierarchical_seq * mask_expanded
            else:
                hierarchical_seq_masked = hierarchical_seq

            hierarchical_global = self.final_attention(
                hierarchical_seq_masked
            )  # [B, han_output_dim]

            # Encode global statistics
            global_encoded = self.global_encoder(global_features)  # [B, hidden_dim/4]

            # Encode tempo curve
            tempo_encoded = self.tempo_encoder(tempo_curve)  # [B, hidden_dim/4]

            # Combine all global features
            combined = torch.cat(
                [
                    hierarchical_global,
                    global_encoded,
                    tempo_encoded,
                ],
                dim=-1,
            )

            global_out = self.global_combiner(combined)
            outputs["global"] = global_out

        return outputs

    def _create_default_note_locations(
        self,
        batch_size: int,
        num_notes: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Create default note_locations with linear beat/measure indices."""
        # Simple linear indexing - every 4 notes is a beat, every 16 notes is a measure
        # Use 1-based indexing (0 is reserved for padding in HAN encoder)
        note_indices = torch.arange(num_notes, device=device)

        beat_indices = note_indices // 4 + 1  # Start from 1
        measure_indices = note_indices // 16 + 1  # Start from 1
        voice_indices = torch.ones(
            num_notes, dtype=torch.long, device=device
        )  # Already 1-based

        return {
            "beat": beat_indices.unsqueeze(0).expand(batch_size, -1),
            "measure": measure_indices.unsqueeze(0).expand(batch_size, -1),
            "voice": voice_indices.unsqueeze(0).expand(batch_size, -1),
        }

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_dim


if __name__ == "__main__":
    print("Score encoder module loaded successfully")
    print("Components:")
    print("- NoteFeatureEncoder: Transformer for per-note deviations")
    print("- GlobalFeatureEncoder: MLP for aggregated statistics")
    print("- TempoCurveEncoder: 1D CNN for tempo ratios")
    print("- ScoreAlignmentEncoder: Combined encoder (flat)")
    print("- HierarchicalScoreEncoder: HAN-based encoder (hierarchical)")
    print("- ScoreMIDIFusion: Fusion strategies (concat, crossattn, gated)")
