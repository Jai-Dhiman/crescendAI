import torch
import torch.nn as nn
from typing import Optional, Tuple


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical temporal aggregation for piano performance evaluation.

    Three-level hierarchy:
    1. Frame-level: CQT frames (~11.6ms) - handled by MERT
    2. Phrase-level: 10-30s segments - BiLSTM + multi-head attention
    3. Piece-level: Full performance - Not implemented in MVP

    Input: Fused features [batch, time_frames, feature_dim]
    Output: Aggregated embedding [batch, output_dim]
    """

    def __init__(
        self,
        input_dim: int = 1024,  # audio_dim + midi_dim
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
        output_dim: int = 512,
    ):
        """
        Initialize hierarchical aggregator.

        Args:
            input_dim: Input feature dimension (from fusion)
            lstm_hidden: LSTM hidden units per direction
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads for pooling
            dropout: Dropout probability
            output_dim: Final output dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim

        # Bidirectional LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # LSTM output dimension (bidirectional)
        lstm_output_dim = lstm_hidden * 2

        # Multi-head attention pooling
        self.attention_heads = attention_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query for attention pooling
        self.attention_query = nn.Parameter(
            torch.randn(1, 1, lstm_output_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # Project to output dimension
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through hierarchical aggregator.

        Args:
            fused_features: Fused audio-MIDI features [batch, time, input_dim]
            mask: Optional attention mask [batch, time]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - aggregated: Aggregated features [batch, output_dim]
                - attention_weights: Attention weights if requested [batch, heads, 1, time]
        """
        batch_size, seq_len, _ = fused_features.shape

        # BiLSTM processing
        lstm_out, (h_n, c_n) = self.lstm(fused_features)
        # lstm_out: [batch, time, lstm_hidden * 2]

        # Multi-head attention pooling
        # Expand query for batch
        query = self.attention_query.expand(batch_size, -1, -1)  # [batch, 1, dim]

        # Attention pooling (query attends to all LSTM outputs)
        attn_out, attn_weights = self.attention(
            query=query,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=mask,
            need_weights=return_attention,
        )
        # attn_out: [batch, 1, lstm_hidden * 2]

        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)  # [batch, lstm_hidden * 2]

        # Layer normalization
        attn_out = self.layer_norm(attn_out)

        # Project to output dimension
        aggregated = self.projection(attn_out)  # [batch, output_dim]

        if return_attention:
            return aggregated, attn_weights
        else:
            return aggregated, None

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


class SimpleTemporalAggregator(nn.Module):
    """
    Simplified temporal aggregator using global pooling.

    For testing when LSTM is too heavy.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 512,
        pooling_mode: str = 'attention',  # 'mean', 'max', or 'attention'
    ):
        """
        Initialize simple aggregator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            pooling_mode: Pooling strategy
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        if pooling_mode == 'attention':
            # Attention-based pooling
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1),
            )

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            fused_features: Features [batch, time, input_dim]
            mask: Optional mask [batch, time]
            return_attention: Return attention weights

        Returns:
            Tuple of (aggregated features, attention weights)
        """
        if self.pooling_mode == 'mean':
            # Mean pooling
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (fused_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = fused_features.mean(dim=1)
            attn_weights = None

        elif self.pooling_mode == 'max':
            # Max pooling
            pooled, _ = fused_features.max(dim=1)
            attn_weights = None

        else:  # attention
            # Attention-weighted pooling
            # Compute attention scores
            scores = self.attention_weights(fused_features)  # [batch, time, 1]

            if mask is not None:
                # Mask out invalid positions
                scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

            # Softmax over time
            attn_weights = torch.softmax(scores, dim=1)

            # Weighted sum
            pooled = (fused_features * attn_weights).sum(dim=1)

        # Project to output dimension
        aggregated = self.projection(pooled)

        if return_attention and attn_weights is not None:
            return aggregated, attn_weights
        else:
            return aggregated, None

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


def create_aggregator(
    input_dim: int = 1024,
    output_dim: int = 512,
    use_lstm: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create temporal aggregator.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        use_lstm: If True, use LSTM-based; else use simple pooling
        **kwargs: Additional arguments

    Returns:
        Temporal aggregator module
    """
    if use_lstm:
        return HierarchicalAggregator(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        return SimpleTemporalAggregator(
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )


if __name__ == "__main__":
    print("Hierarchical aggregation module loaded successfully")
    print("- BiLSTM for sequential modeling")
    print("- Multi-head attention pooling")
    print("- Simple pooling fallback (mean/max/attention)")
    print("- Attention weight visualization support")
