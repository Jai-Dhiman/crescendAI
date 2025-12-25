"""
Context Attention module for hierarchical aggregation.

Ported from PercePiano's virtuoso/module.py (line 348-383).
Implements multi-head attention for aggregating sequences.

CRITICAL: The original PercePiano uses a SIMPLER attention mechanism:
- attention_net: Linear(size, num_head) - projects to num_head scores per position
- softmax over heads (not over sequence)
- BUT the multiplication x * attention has a shape mismatch in original code

After investigation, we implement a corrected version that:
1. Projects to num_head attention scores per position
2. Applies softmax over the SEQUENCE dimension (not heads)
3. Computes weighted sum properly
"""

from typing import Optional

import torch
import torch.nn as nn


class ContextAttention(nn.Module):
    """
    Multi-head context attention for sequence aggregation.

    This implementation is based on the original PercePiano but fixes the
    shape mismatch in the original code. Key design:
    - Projects input to num_head scores per position: [B, T, num_head]
    - Softmax over sequence (T) for each head
    - Weighted sum of input features

    The original PercePiano code has:
        attention = self.attention_net(x)  # [B, T, num_head]
        attention = self.softmax(attention)  # softmax over heads (dim=-1)
        upper = x * attention  # SHAPE MISMATCH: [B, T, D] * [B, T, H]

    We fix this by applying softmax over sequence (dim=1) and properly
    broadcasting the attention weights.
    """

    def __init__(self, size: int, num_head: int):
        """
        Args:
            size: Input feature dimension
            num_head: Number of attention heads
        """
        super().__init__()

        self.size = size
        self.num_head = num_head

        # Project to num_head attention scores per position
        # This matches original: nn.Linear(size, num_head)
        self.attention_net = nn.Linear(size, num_head)

        # Use softmax over sequence dimension for proper aggregation
        self.softmax = nn.Softmax(dim=1)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores (before softmax).

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Raw attention scores of shape (B, T, num_head)
        """
        return self.attention_net(x)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention and return weighted sum.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional mask of shape (B, T) where True = valid position

        Returns:
            Weighted sum of shape (B, D)
        """
        # Compute attention scores: [B, T, num_head]
        attention = self.attention_net(x)

        # Mask out invalid positions before softmax
        if mask is not None:
            # Expand mask to match attention shape
            mask_expanded = mask.unsqueeze(-1).expand_as(attention)
            attention = attention.masked_fill(~mask_expanded, -1e9)

        # Also mask zero-padded positions (backup for when mask not provided)
        is_zero_padded = (x.abs().sum(dim=-1) < 1e-8)  # [B, T]
        if is_zero_padded.any():
            zero_mask = is_zero_padded.unsqueeze(-1).expand_as(attention)
            attention = attention.masked_fill(zero_mask, -1e9)

        # Softmax over SEQUENCE dimension (not heads)
        # This gives attention weights that sum to 1 across positions for each head
        attention_weights = self.softmax(attention)  # [B, T, num_head]

        # Weighted sum: average over heads, then sum over sequence
        # Expand attention weights to match input: [B, T, num_head] -> [B, T, 1]
        # We average the attention weights across heads
        avg_attention = attention_weights.mean(dim=-1, keepdim=True)  # [B, T, 1]

        # Weighted sum over sequence
        weighted_x = x * avg_attention  # [B, T, D] * [B, T, 1] = [B, T, D]
        output = weighted_x.sum(dim=1)  # [B, D]

        return output


class SimpleAttention(nn.Module):
    """
    Simple attention without multiple heads or context vectors.

    Projects input to attention scores and computes weighted sum.
    """

    def __init__(self, size: int):
        """
        Args:
            size: Input/output feature dimension
        """
        super().__init__()
        self.attention_net = nn.Linear(size, size)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw attention scores."""
        return self.attention_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention and return weighted sum.

        Args:
            x: Input tensor of shape (N, T, C)

        Returns:
            Weighted sum of shape (N, C)
        """
        attention = self.attention_net(x)
        softmax_weight = torch.softmax(attention, dim=1)
        attention = softmax_weight * x
        sum_attention = torch.sum(attention, dim=1)
        return sum_attention
